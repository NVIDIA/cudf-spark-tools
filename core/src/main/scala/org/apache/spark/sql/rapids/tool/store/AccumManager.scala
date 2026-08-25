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

import scala.collection.{mutable, Map}

import com.nvidia.spark.rapids.tool.analysis.StatisticsMetrics

import org.apache.spark.scheduler.AccumulableInfo
import org.apache.spark.sql.rapids.tool.util.EventUtils

/**
 * A class that manages task/stage accumulables -
 * maintains a map of accumulable id to AccumInfo
 */
class AccumManager {
  val accumInfoMap: mutable.HashMap[Long, AccumInfo] = {
    new mutable.HashMap[Long, AccumInfo]()
  }

  // Stage attempts that reported positive GPU spill. See [[hasGpuSpillEvidence]].
  private val gpuSpillStageAttempts: mutable.HashSet[(Int, Int)] = mutable.HashSet.empty

  private def getOrCreateAccumInfo(id: Long, name: Option[String]): AccumInfo = {
    accumInfoMap.getOrElseUpdate(id, AccumInfo(AccumMetaRef(id, name)))
  }

  def addAccToStage(stageId: Int, accumulableInfo: AccumulableInfo): Unit = {
    val accumInfoRef = getOrCreateAccumInfo(accumulableInfo.id, accumulableInfo.name)
    accumInfoRef.addAccumToStage(stageId, accumulableInfo)
  }

  /**
   * Records a task-level accumulable.
   *
   * The per-accumulator statistics are intentionally keyed by stage only (see [[AccumInfo]]).
   * The stage attempt is additionally retained for the GPU spill metrics, because a downward
   * tuning decision must be able to tell spill in a failed attempt apart from spill in the
   * later successful attempt it selected.
   */
  def addAccToTask(stageId: Int, stageAttemptId: Int, accumulableInfo: AccumulableInfo): Unit = {
    val accumInfoRef = getOrCreateAccumInfo(accumulableInfo.id, accumulableInfo.name)
    accumInfoRef.addAccumToTask(stageId, accumulableInfo)
    recordGpuSpillAttempt(stageId, stageAttemptId, accumulableInfo)
  }

  /**
   * Notes the stage attempt when a GPU spill accumulable reports a positive update.
   *
   * Only the spill metric names are tracked, so this adds a bounded amount of state rather than
   * duplicating every accumulator per attempt.
   */
  private def recordGpuSpillAttempt(
      stageId: Int, stageAttemptId: Int, accumulableInfo: AccumulableInfo): Unit = {
    val isSpillMetric =
      accumulableInfo.name.exists(AccumManager.GPU_SPILL_METRIC_NAMES.contains)
    if (isSpillMetric) {
      val positiveUpdate = accumulableInfo.update
        .flatMap(EventUtils.parseAccumFieldToLong)
        .exists(_ > 0L)
      if (positiveUpdate) {
        gpuSpillStageAttempts += ((stageId, stageAttemptId))
      }
    }
  }

  /**
   * True when the given stage attempt reported any positive GPU spill activity.
   *
   * GPU host and disk spill are not visible in Spark's task metrics, so this is the only
   * attempt-scoped spill evidence available for GPU event logs.
   */
  def hasGpuSpillEvidence(stageId: Int, stageAttemptId: Int): Boolean = {
    gpuSpillStageAttempts.contains((stageId, stageAttemptId))
  }

  def getAccStageIds(id: Long): Set[Int] = {
    accumInfoMap.get(id).map(_.getStageIds).getOrElse(Set.empty)
  }

  def getAccumSingleStage: Map[Long, Int] = {
    accumInfoMap.map { case (id, accInfo) =>
      (id, accInfo.getMinStageId)
    }.toMap
  }

  def removeAccumInfo(id: Long): Option[AccumInfo] = {
    accumInfoMap.remove(id)
  }

  def calculateAccStats(id: Long): Option[StatisticsMetrics] = {
    accumInfoMap.get(id).map(_.calculateAccStats())
  }

  def getMaxStageValue(id: Long): Option[Long] = {
    accumInfoMap.get(id).map(_.getMaxTotalAcrossStages.get)
  }

  /**
   * Applies the function `f` to each AccumInfo in the accumInfoMap.
   */
  def applyToAccumInfoMap(f: AccumInfo => Unit): Unit = {
    accumInfoMap.values.foreach(f)
  }
}

object AccumManager {
  /**
   * RAPIDS accumulator names that indicate the GPU spilled. These are reported as durations, so
   * only their positive/zero state is meaningful here, not their magnitude.
   */
  val GPU_SPILL_METRIC_NAMES: Set[String] = Set(
    "gpuSpillToHostTime",
    "gpuSpillToDiskTime",
    "gpuReadSpillFromHostTime",
    "gpuReadSpillFromDiskTime")
}
