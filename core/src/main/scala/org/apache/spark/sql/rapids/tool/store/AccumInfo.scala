/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids.tool.store

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.analysis.StatisticsMetrics

import org.apache.spark.scheduler.AccumulableInfo
import org.apache.spark.sql.rapids.tool.util.EventUtils
import org.apache.spark.sql.rapids.tool.util.EventUtils.parseAccumFieldToLong


/**
 * Maintains the accumulator information for a single accumulator.
 * This maintains following information:
 * 1. Statistical metrics for each stage including min, median, max, count and sum
 * 2. AccumMetaRef for the accumulator - a reference to the Meta information
 * @param infoRef - AccumMetaRef for the accumulator
 */
class AccumInfo(val infoRef: AccumMetaRef) {
  /**
   * Maps stageId to StatisticsMetrics which contains:
   * - min: Smallest value across the reporting task attempts
   * - med: Median value across the reporting task attempts
   * - max: Largest value across the reporting task attempts
   * - count: Number of task attempts in the stage that reported the metric
   * - total: Published total, which a stage completion event may raise
   * - sampleTotal: Sum of the task updates alone, the numerator behind the mean
   * - welfordSumSqDev: Sum of squared deviations over the same task updates
   */
  protected val stagesStatMap: mutable.HashMap[Int, StatisticsMetrics] =
    new mutable.HashMap[Int, StatisticsMetrics]()

  /**
   * Adds or updates accumulator information for a stage.
   * Called from StageCompleted event processing, without filtering on the stage attempt or on
   * whether the attempt succeeded. The map is keyed by stage id, so every attempt of a stage
   * shares one record.
   *
   * @param stageId The ID of the stage
   * @param accumulableInfo Accumulator information from the event
   * @param update Fallback when the event carries no parseable value; unused by callers
   */
  def addAccumToStage(stageId: Int,
      accumulableInfo: AccumulableInfo,
      update: Option[Long] = None): Unit = {
    val parsedValue = accumulableInfo.value.flatMap(parseValue)
    val existingEntry = stagesStatMap.getOrElse(stageId,
      StatisticsMetrics.ZERO_RECORD)
    val incomingValue = parsedValue match {
      case Some(v) => v
      case _ => update.getOrElse(0L)
    }
    // Out of order or duplicated stage events can carry a value lower than one already seen, so
    // keep the maximum. Only the published total moves: min, max, count, sampleTotal and
    // welfordSumSqDev describe the task updates and must keep describing the same population.
    stagesStatMap.put(stageId, existingEntry.copy(
      total = Math.max(existingEntry.total, incomingValue)))
  }

  /**
   * Processes task-level accumulator updates and updates stage-level statistics.
   * Called once per accumulable carried by a TaskEnd event, without filtering on the task
   * attempt or on whether it succeeded. Failed and retried attempts do carry accumulables, so
   * they contribute samples; the sampled population is task attempts rather than tasks.
   *
   * @param stageId The ID of the stage containing the task
   * @param accumulableInfo Accumulator information from the TaskEnd event
   */
  def addAccumToTask(stageId: Int, accumulableInfo: AccumulableInfo): Unit = {
    // 1. We first extract the incoming task update value
    // 2. Then allocate a new Statistic metric object with min,max as incoming update
    // 3. Use count to calculate rolling average
    // 4. Increment count by 1
    // 5. Add the update to both the published total and the sample total
    // 6. Create final object and update map
    // TODO: rename med, which holds a rolling average rather than a median. See issue 2140.
    val parsedUpdateValue = accumulableInfo.update.flatMap(parseValue)
    // we need to update the stageMap if the stageId does not exist in the map
    parsedUpdateValue.foreach { value =>
      val stats = stagesStatMap.getOrElse(stageId,
        StatisticsMetrics(value, 0L, value, 0, 0L))
      val newCount = stats.count + 1
      val newTotal = stats.total + value
      val newSampleTotal = stats.sampleTotal + value
      // Welford, with the mean derived from the sample total and count the record holds.
      val meanBefore = if (stats.count == 0L) 0.0 else stats.sampleTotal.toDouble / stats.count
      val meanAfter = newSampleTotal.toDouble / newCount
      val sqDev = (value.toDouble - meanBefore) * (value.toDouble - meanAfter)
      // A record a stage event created carries no samples, so the first task update sets min
      // and max instead of folding them against the placeholder zeros that record holds.
      val firstSample = stats.count == 0L
      val newStats = StatisticsMetrics(
        if (firstSample) value else Math.min(stats.min, value),
        (stats.med * stats.count + value) / ( stats.count + 1),
        if (firstSample) value else Math.max(stats.max, value),
        newCount,
        newTotal,
        stats.welfordSumSqDev + sqDev,
        newSampleTotal
      )
      stagesStatMap.put(stageId, newStats)
    }
  }

  /**
   * Parses a raw accumulable value, applying the fixed-point storage scale the metric catalog
   * declares for this metric. A value no branch can read is dropped and reported once per
   * process -- this used to be silent, which is how a Double-valued accumulator went unnoticed
   * while its metric published zeros.
   */
  private def parseValue(rawValue: Any): Option[Long] = {
    val parsed = parseAccumFieldToLong(rawValue, infoRef.storageScale)
    if (parsed.isEmpty) {
      EventUtils.reportUnparseableAccum(infoRef.getName(), rawValue)
    }
    parsed
  }

  // Getters for stage-specific metrics

  /**
   * Gets the total value for a specific stage
   */
  def getTotalForStage(stageId: Int): Option[Long] = {
    stagesStatMap.get(stageId).map(_.total)
  }

  /**
   * Gets the maximum value for a specific stage
   */
  def getMaxForStage(stageId: Int): Option[Long] = {
    stagesStatMap.get(stageId).map(_.max)
  }

  // Getters for cross-stage aggregates

  /**
   * Gets sum of values across all stages
   */
  def getTotalAcrossStages: Long = {
    stagesStatMap.values.map(_.total).sum
  }

  /**
   * Get max total across stages
   */
  def getMaxTotalAcrossStages: Option[Long] = {
    if (stagesStatMap.values.isEmpty) {
      None
    } else {
      Some(stagesStatMap.values.map(_.total).max)
    }
  }

  // Utility methods

  /**
   * Returns all stage IDs that have accumulator updates.
   *
   * @return Set of stage IDs
   */
  def getStageIds: Set[Int] = {
    stagesStatMap.keySet.toSet
  }

  /**
   * Returns the smallest stage ID in the accumulator data.
   *
   * @return Minimum stage ID
   */
  def getMinStageId: Int = {
    stagesStatMap.keys.min
  }

  /**
   * Calculates aggregate statistics across all stages for this accumulator
   */
  def calculateAccStats(): StatisticsMetrics = {
    val reduced_val = stagesStatMap.values.reduce { (a, b) =>
      val totalCount = a.count + b.count
      val medianValue = if (totalCount == 0) {
        0L
      } else {
        (a.med * a.count + b.med * b.count) / totalCount
      }
      val mergedSumSqDev = StatisticsMetrics.mergeSumSqDev(
        a.count, a.sampleTotal, a.welfordSumSqDev, b.count, b.sampleTotal, b.welfordSumSqDev)
      StatisticsMetrics(
        Math.min(a.min, b.min),
        medianValue,
        Math.max(a.max, b.max),
        totalCount,
        a.total + b.total,
        mergedSumSqDev,
        a.sampleTotal + b.sampleTotal
      )
    }
    readjustTotalStats(reduced_val)
  }

  /**
   * The unadjusted record for a stage, before `readjustTotalStats` masks `total`.
   *
   * Needed to compute a real arithmetic mean: `med` is a rolling mean recomputed with integer
   * division on every update, so it ratchets toward the floor and is badly wrong for
   * small-valued metrics. `sampleTotal / count` is one Double division, rounded once at
   * render rather than truncated on every update.
   */
  def getRawStatsForStage(stageId: Int): Option[StatisticsMetrics] = stagesStatMap.get(stageId)

  /**
   * Retrieves statistical metrics for a specific stage
   */
  def calculateAccStatsForStage(stageId: Int): Option[StatisticsMetrics] = {
    stagesStatMap.get(stageId).map { statValue =>
      readjustTotalStats(statValue)
    }
  }

  /**
   * Checks if stage exists in the map
   */
  def containsStage(stageId: Int): Boolean = {
    stagesStatMap.contains(stageId)
  }

  /**
   * Override this method to readjust the total value for the accumulator.
   * This is useful to do any final adjustments of the aggregations on the metric.
   */
  protected def readjustTotalStats(statsRec: StatisticsMetrics): StatisticsMetrics = {
    statsRec
  }
}

/**
 * Derived from AccumInfo, this class represents accumulators that are aggregated by the maximum
 * value. The implementation overrides the original class by enforcing the "max" value to be used
 * instead of the total
 * @param infoRef - AccumMetaRef for the accumulator
 */
class AccumInfoWithMaxAgg(override val infoRef: AccumMetaRef) extends AccumInfo(infoRef) {
  // The aggregate by max should not return the total field. instead, it enforces the max field.
  override protected def readjustTotalStats(statsRec: StatisticsMetrics): StatisticsMetrics = {
    statsRec.copy(total = statsRec.max)
  }
  /**
   * Get max total across stages. for that type of aggregates, the total should not be used.
   * Instead, use the max.
   */
  override def getMaxTotalAcrossStages: Option[Long] = {
    if (stagesStatMap.values.isEmpty) {
      None
    } else {
      Some(stagesStatMap.values.map(_.max).max)
    }
  }
}

object AccumInfo {
  def apply(infoRef: AccumMetaRef): AccumInfo = {
    infoRef.metricCategory match {
      case 1 => // max aggregate
        new AccumInfoWithMaxAgg(infoRef)
      case _ =>  // default
        new AccumInfo(infoRef)
    }
  }
}
