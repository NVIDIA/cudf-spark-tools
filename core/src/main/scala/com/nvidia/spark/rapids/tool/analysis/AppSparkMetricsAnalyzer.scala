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

package com.nvidia.spark.rapids.tool.analysis

import scala.collection.mutable.{HashMap, LinkedHashMap}

import com.nvidia.spark.rapids.tool.analysis.photon.PhotonAppSparkMetricsAnalyzer
import com.nvidia.spark.rapids.tool.analysis.util.AggAccumHelper
import com.nvidia.spark.rapids.tool.analysis.util.StageAccumDiagnosticMetrics._
import com.nvidia.spark.rapids.tool.profiling._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.{AppBase, ToolUtils}
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.store.AccumMetaRef

/**
 * Does analysis on the DataFrames from object of AppBase.
 * This class does the following:
 * - aggregates SparkMetrics by Job, Stage, and SQL.
 * - checks for shuffle skewness.
 * - Find the max inputSizes for SQL.
 *
 * The implementation is tuned to improve the performance by reducing the number of times the
 * analyzer visits the Tasks.
 * 1- The assumption is that it is unlikely that the analysis will be aggregating metrics only for
 *     one of for SQL, jobs, or stages. Instead, any analysis is likely to do SQL/Stage levels.
 * 2- The analyzer caches the stage level metrics to avoid recalculating the same metrics several
 *    times
 * 3- The cached stage-level metrics are then used to calculate the aggregates for SQLs, and Jobs
 * 4- It can be used by both Qual/Prof tools: this why it takes app-index as an argument to the
 *    aggregator methods. The index is a value used by the Profiler tool to list records from
 *    multiple applications.
 *
 * @param app the AppBase object to analyze
 */
class AppSparkMetricsAnalyzer(app: AppBase) extends AppAnalysisBase(app) with Logging {
  // Hashmap to cache the stage level metrics. It is initialized to None just in case the caller
  // does not call methods in order starting with stage level metrics.
  private var stageLevelCache:
    Option[LinkedHashMap[Int, StageAggTaskMetricsProfileResult]] = None

  // Getter method used to protect the cache from out-of-order calls.
  // If the stage-level metrics are not generated yet, generates and add them to the cache
  private def stageLevelSparkMetrics(
      index: Int): LinkedHashMap[Int, StageAggTaskMetricsProfileResult] = {
    if (stageLevelCache.isEmpty) {
      stageLevelCache = Some(LinkedHashMap[Int, StageAggTaskMetricsProfileResult]())
      aggregateSparkMetricsByStageInternal(index)
    }
    stageLevelCache.get
  }

  /**
   *  Aggregate the SparkMetrics by stage
   * @param index the App-index (used by the profiler tool)
   * @return sequence of StageAggTaskMetricsProfileResult that contains only Stage Ids
   */
  def aggregateSparkMetricsByStage(index: Int): Seq[StageAggTaskMetricsProfileResult] = {
    stageLevelSparkMetrics(index).values.toSeq
  }

  /**
   * Aggregate the SparkMetrics by Job
   * @param index the App-index (used by the profiler tool)
   * @return sequence of JobAggTaskMetricsProfileResult that contains only Job Ids
   */
  def aggregateSparkMetricsByJob(index: Int): Seq[JobAggTaskMetricsProfileResult] = {
    app.jobIdToInfo.flatMap { case (id, jc) =>
      if (jc.stageIds.isEmpty) {
        None
      } else {
        val jobAggAccumulator = new AggAccumHelper()
        val perJobRec = jobAggAccumulator.accumPerJob(
          jc.stageIds.collect {
            case stageId if stageLevelSparkMetrics(index).contains(stageId) =>
              stageLevelSparkMetrics(index)(stageId)
          })
        if (perJobRec.isEmptyAggregates) {
          None
        } else {
          Some(JobAggTaskMetricsProfileResult(
            id,
            perJobRec.numTasks,
            jc.duration,
            perJobRec.diskBytesSpilledSum,
            perJobRec.durationSum,
            perJobRec.durationMax,
            perJobRec.durationMin,
            perJobRec.durationAvg,
            perJobRec.executorCPUTimeSum,
            perJobRec.executorDeserializeCpuTimeSum,
            perJobRec.executorDeserializeTimeSum,
            perJobRec.executorRunTimeSum,
            perJobRec.inputBytesReadSum,
            perJobRec.inputBytesReadMax,
            perJobRec.inputRecordsReadSum,
            perJobRec.jvmGCTimeSum,
            perJobRec.memoryBytesSpilledSum,
            perJobRec.outputBytesWrittenSum,
            perJobRec.outputRecordsWrittenSum,
            perJobRec.peakExecutionMemoryMax,
            perJobRec.resultSerializationTimeSum,
            perJobRec.resultSizeMax,
            perJobRec.srFetchWaitTimeSum,
            perJobRec.srLocalBlocksFetchedSum,
            perJobRec.srLocalBytesReadSum,
            perJobRec.srRemoteBlocksFetchSum,
            perJobRec.srRemoteBytesReadSum,
            perJobRec.srRemoteBytesReadToDiskSum,
            perJobRec.srTotalBytesReadSum,
            perJobRec.swBytesWrittenSum,
            perJobRec.swRecordsWrittenSum,
            perJobRec.swWriteTimeSum))
        }
      }
    }.toSeq
  }

  private case class AverageStageInfo(avgDuration: Double, avgShuffleReadBytes: Double)

  /**
   * Scans tasks to identify if any exhibits shuffle skewness. If a task has input size larger than
   * 3X the average shuffle read size and larger than 100MB, it is considered as a skew task.
   * @param index the App-index (used by the profiler tool)
   * @return sequence of ShuffleSkewProfileResult that contains only the skew tasks
   */
  def shuffleSkewCheck(index: Int): Seq[ShuffleSkewProfileResult] = {
    // TODO: we can add averageShuffleRead as a field in JobStageAggTaskMetricsProfileResult instead
    //       of making an extra path on the StageAttempts
    val avgStageInfos = app.taskManager.stageAttemptToTasks.collect {
      // TODO: Should we only consider successful tasks?
      case (stageId, attemptsToTasks) if attemptsToTasks.nonEmpty =>
        attemptsToTasks.map { case (attemptId, tasks) =>
          val sumDuration = tasks.map(_.duration).sum
          val avgDuration = ToolUtils.calculateAverage(sumDuration, tasks.size, 2)
          val sumShuffleReadBytes = tasks.map(_.sr_totalBytesRead).sum
          val avgShuffleReadBytes = ToolUtils.calculateAverage(sumShuffleReadBytes, tasks.size, 2)
          ((stageId, attemptId), AverageStageInfo(avgDuration, avgShuffleReadBytes))
        }
    }.flatten

    avgStageInfos.flatMap { case ((stageId, attemptId), avg) =>
      val definedTasks =
        app.taskManager.getTasks(stageId, attemptId, Some(
          tc => (tc.sr_totalBytesRead > 3 * avg.avgShuffleReadBytes)
            && (tc.sr_totalBytesRead > 100 * 1024 * 1024)))
      definedTasks.map { tc =>
        ShuffleSkewProfileResult(stageId, attemptId,
          tc.taskId, tc.attempt, tc.duration, avg.avgDuration, tc.sr_totalBytesRead,
          avg.avgShuffleReadBytes, tc.peakExecutionMemory, tc.successful, tc.endReason)
      }
    }.toSeq
  }

  /**
   * Aggregate the SparkMetrics by SQL
   * @param index the App-index (used by the profiler tool)
   * @return sequence of SQLTaskAggMetricsProfileResult
   */
  def aggregateSparkMetricsBySql(index: Int): Seq[SQLTaskAggMetricsProfileResult] = {
    app.sqlIdToInfo.flatMap { case (sqlId, sqlCase) =>
      if (app.sqlIdToStages.contains(sqlId)) {
        val stagesInSQL = app.sqlIdToStages(sqlId)
        // TODO: Should we only consider successful tasks?
        val sqlAggAccumulator = new AggAccumHelper()
        val preSqlRec = sqlAggAccumulator.accumPerSQL(
          stagesInSQL.collect {
            case stageId if stageLevelSparkMetrics(index).contains(stageId) =>
              stageLevelSparkMetrics(index)(stageId)
          })
        if (preSqlRec.isEmptyAggregates) {
          None
        } else {
          // set this here, so make sure we don't get it again until later
          sqlCase.sqlCpuTimePercent = preSqlRec.executorCpuRatio
          Some(SQLTaskAggMetricsProfileResult(
            app.appId,
            sqlId,
            sqlCase.description,
            preSqlRec.numTasks,
            sqlCase.duration,
            preSqlRec.executorCpuRatio,
            preSqlRec.diskBytesSpilledSum,
            preSqlRec.durationSum,
            preSqlRec.durationMax,
            preSqlRec.durationMin,
            preSqlRec.durationAvg,
            preSqlRec.executorCPUTimeSum,
            preSqlRec.executorDeserializeCpuTimeSum,
            preSqlRec.executorDeserializeTimeSum,
            preSqlRec.executorRunTimeSum,
            preSqlRec.inputBytesReadSum,
            preSqlRec.inputBytesReadMax,
            preSqlRec.inputBytesReadAvg,
            preSqlRec.inputRecordsReadSum,
            preSqlRec.jvmGCTimeSum,
            preSqlRec.memoryBytesSpilledSum,
            preSqlRec.outputBytesWrittenSum,
            preSqlRec.outputRecordsWrittenSum,
            preSqlRec.peakExecutionMemoryMax,
            preSqlRec.resultSerializationTimeSum,
            preSqlRec.resultSizeMax,
            preSqlRec.srFetchWaitTimeSum,
            preSqlRec.srLocalBlocksFetchedSum,
            preSqlRec.srLocalBytesReadSum,
            preSqlRec.srRemoteBlocksFetchSum,
            preSqlRec.srRemoteBytesReadSum,
            preSqlRec.srRemoteBytesReadToDiskSum,
            preSqlRec.srTotalBytesReadSum,
            preSqlRec.swBytesWrittenSum,
            preSqlRec.swRecordsWrittenSum,
            preSqlRec.swWriteTimeSum))
        }
      } else {
        None
      }
    }.toSeq
  }

  /**
   * Aggregates the IO metrics by SQL
   * @param sqlMetricsAggs Spark metrics the aggregated by SQL. This is an optimization tuning to
   *                       avoid recalculating those metrics twice.
   * @return IOAnalysisProfileResult that contains the IO metrics aggregated by SQL
   */
  def aggregateIOMetricsBySql(
      sqlMetricsAggs: Seq[SQLTaskAggMetricsProfileResult]): Seq[IOAnalysisProfileResult] = {
    sqlMetricsAggs.map { sqlAgg =>
      IOAnalysisProfileResult(
        app.appId,
        sqlAgg.sqlId,
        sqlAgg.inputBytesReadSum,
        sqlAgg.inputRecordsReadSum,
        sqlAgg.outputBytesWrittenSum,
        sqlAgg.outputRecordsWrittenSum,
        sqlAgg.diskBytesSpilledSum,
        sqlAgg.memoryBytesSpilledSum,
        sqlAgg.srTotalBytesReadSum,
        sqlAgg.swBytesWrittenSum)
    }.toSeq
  }

  /**
   * Aggregates the duration and CPU time (milliseconds) by SQL
   * @param index App index  (used by the profiler tool)
   * @return a sequence of SQLDurationExecutorTimeProfileResult or Empty if None.
   */
  def aggregateDurationAndCPUTimeBySql(index: Int): Seq[SQLDurationExecutorTimeProfileResult] = {
    app.sqlIdToInfo.map { case (sqlId, sqlCase) =>
      // First, build the SQLIssues string by retrieving the potential issues from the
      // app.sqlIDtoProblematic map.
      val sqlIssues = if (app.sqlIDtoProblematic.contains(sqlId)) {
        ToolUtils.formatPotentialProblems(app.sqlIDtoProblematic(sqlId).toSeq)
      } else {
        ""
      }
      // Then, build the SQLDurationExecutorTimeProfileResult
      SQLDurationExecutorTimeProfileResult(app.appId, sqlCase.rootExecutionID,
        sqlId, sqlCase.duration, sqlCase.hasDatasetOrRDD,
        app.getAppDuration.orElse(Option(0L)), sqlIssues, sqlCase.sqlCpuTimePercent)
    }.toSeq
  }

  /**
   * Aggregates the diagnostic SparkMetrics by stage.
   * @param index    the App-index (used by the profiler tool)
   * @param analyzer optional AppSQLPlanAnalyzer which is used to pull stage level
   *                 information like node names and diagnostic metrics results, only
   *                 Qualification needs to provide this argument.
   * @return sequence of StageDiagnosticAggTaskMetricsProfileResult
   */
  def aggregateDiagnosticMetricsByStage(index: Int, analyzer: Option[AppSQLPlanAnalyzer] = None):
      Seq[StageDiagnosticResult] = {
    if (!isDiagnosticViewsEnabled) {
      return Seq.empty
    }

    // Then get the appropriate analyzer
    val sqlAnalyzer = analyzer match {
      case Some(analyzer) => analyzer
      case None => app.asInstanceOf[ApplicationInfo].planMetricProcessor
    }

    val zeroAccumProfileResults =
      AccumProfileResults(0, AccumMetaRef.EMPTY_ACCUM_META_REF, 0L, 0L, 0L, 0L)
    val emptyNodeNames = Seq.empty[String]
    val emptyDiagnosticMetrics = HashMap.empty[String, AccumProfileResults]
    app.stageManager.getAllStages.map { sm =>
      val tasksInStage = app.taskManager.getTasks(sm.stageInfo.stageId,
        sm.stageInfo.attemptNumber())
      // count duplicate task attempts
      val numTasks = tasksInStage.size
      val nodeNames = sqlAnalyzer.stageToNodeNames.getOrElse(sm.stageInfo.stageId, emptyNodeNames)
      val diagnosticMetricsMap =
        sqlAnalyzer.stageToDiagnosticMetrics
          .getOrElse(sm.stageInfo.stageId, emptyDiagnosticMetrics)
          .withDefaultValue(zeroAccumProfileResults)
      val srTotalBytesMetrics =
        StatisticsMetrics.createFromArr(tasksInStage.iterator.map(_.sr_totalBytesRead).toArray)

      StageDiagnosticResult(
        app.getAppName,
        app.appId,
        sm.stageInfo.stageId,
        sm.duration,
        numTasks,
        srTotalBytesMetrics.min,
        srTotalBytesMetrics.med,
        srTotalBytesMetrics.max,
        srTotalBytesMetrics.total,
        diagnosticMetricsMap(MEMORY_SPILLED_METRIC),
        diagnosticMetricsMap(DISK_SPILLED_METRIC),
        diagnosticMetricsMap(INPUT_BYTES_READ_METRIC),
        diagnosticMetricsMap(OUTPUT_BYTES_WRITTEN_METRIC),
        diagnosticMetricsMap(SW_TOTAL_BYTES_METRIC),
        diagnosticMetricsMap(SR_FETCH_WAIT_TIME_METRIC),
        diagnosticMetricsMap(SW_WRITE_TIME_METRIC),
        diagnosticMetricsMap(GPU_SEMAPHORE_WAIT_METRIC),
        nodeNames)
    }.toSeq
  }

  /**
   * Creates the appropriate accumulator helper for aggregating metrics.
   * This method can be overridden by subclasses to provide platform-specific implementations.
   *
   * @param stageId The stage ID for which to create the helper
   * @return AggAccumHelper instance for standard Spark metrics aggregation
   */
  protected def createAccumHelper(stageId: Int): AggAccumHelper = {
    new AggAccumHelper()
  }

  /**
   * Aggregates the SparkMetrics by completed stage information.
   * This is an internal method to populate the cached metrics
   * to be used by other aggregators.
   * @param index AppIndex (used by the profiler tool)
   */
  protected def aggregateSparkMetricsByStageInternal(index: Int): Unit = {
    app.stageManager.getAllStages.foreach { sm =>
      // TODO: Should we only consider successful tasks?
      val tasksInStage = app.taskManager.getTasks(sm.stageInfo.stageId,
        sm.stageInfo.attemptNumber())

      val accumHelperObj = createAccumHelper(sm.stageInfo.stageId)
      val perStageRec = accumHelperObj.accumPerStage(tasksInStage)
      val stageRow = StageAggTaskMetricsProfileResult(
        sm.stageInfo.stageId,
        // numTasks includes duplicate task attempts
        perStageRec.numTasks,
        sm.duration,
        perStageRec.diskBytesSpilledSum,
        perStageRec.durationSum,
        perStageRec.durationMax,
        perStageRec.durationMin,
        perStageRec.durationAvg,
        perStageRec.executorCPUTimeSum,  // converted to milliseconds by the aggregator
        perStageRec.executorDeserializeCpuTimeSum,  // converted to milliseconds by the aggregator
        perStageRec.executorDeserializeTimeSum,
        perStageRec.executorRunTimeSum,
        perStageRec.inputBytesReadSum,
        perStageRec.inputBytesReadMax,
        perStageRec.inputRecordsReadSum,
        perStageRec.jvmGCTimeSum,
        perStageRec.memoryBytesSpilledSum,
        perStageRec.outputBytesWrittenSum,
        perStageRec.outputRecordsWrittenSum,
        perStageRec.peakExecutionMemoryMax,
        perStageRec.resultSerializationTimeSum,
        perStageRec.resultSizeMax,
        perStageRec.srFetchWaitTimeSum,
        perStageRec.srLocalBlocksFetchedSum,
        perStageRec.srLocalBytesReadSum,
        perStageRec.srRemoteBlocksFetchSum,
        perStageRec.srRemoteBytesReadSum,
        perStageRec.srRemoteBytesReadToDiskSum,
        perStageRec.srTotalBytesReadSum,
        perStageRec.swBytesWrittenSum,
        perStageRec.swRecordsWrittenSum,
        perStageRec.swWriteTimeSum)  // converted to milliseconds by the aggregator
      // This logic is to handle the case where there are multiple attempts for a stage.
      // We check if the StageLevelCache already has a row for the stage.
      // If yes, we aggregate the metrics of the new row with the existing row.
      // If no, we just store the new row.
      val rowToStore = stageLevelSparkMetrics(index)
        .get(sm.stageInfo.stageId)
        .map(_.aggregateStageProfileMetric(stageRow))
        .getOrElse(stageRow)
      stageLevelSparkMetrics(index).put(sm.stageInfo.stageId, rowToStore)
    }
  }

  // ---------------------------------------------------------------------------
  // GPU task metric aggregations (Stage / SQL / App)
  // ---------------------------------------------------------------------------
  //
  // Discovery lives in MetricCatalog.isGpuReportedMetric and is cached per accumulator id on
  // AccumMetaRef, so the rule is stated once rather than reconstructed at each call site.
  //
  // Unit comes from the catalog, which declares it per metric. An undeclared metric falls back
  // to the legacy name heuristic, which is label-only: nothing is scaled by it.
  //
  // There is deliberately no value conversion here. EventUtils.parseAccumFieldToLong already
  // normalizes every serialized form to a canonical unit -- a plain integer stays raw,
  // "00:00:01.773" becomes 1773 milliseconds, "3.28GB (3526702303 bytes)" becomes bytes -- so
  // converting again is what silently deleted every timing metric from this report.

  private def unitForMetric(name: String): String = MetricCatalog.DEFAULT.unitFor(name)

  /**
   * Aggregate GPU task accumulators by stage. Emits one row per (stageId,
   * metricName). Returns Seq.empty when the app has no GPU metrics, which
   * upstream uses to suppress CSV generation.
   */
  def aggregateGpuMetricsByStage(index: Int): Seq[StageAggGpuMetricsProfileResult] = {
    val gpuAccums = app.accumManager.accumInfoMap.values.filter { ai =>
      ai.infoRef.isGpuReportedMetric
    }.toSeq
    if (gpuAccums.isEmpty) {
      return Seq.empty
    }
    val stageCache = stageLevelSparkMetrics(index)
    val rows = scala.collection.mutable.ArrayBuffer[StageAggGpuMetricsProfileResult]()
    gpuAccums.foreach { ai =>
      val name = ai.infoRef.getName()
      val unit = unitForMetric(name)
      ai.getStageIds.foreach { stageId =>
        ai.calculateAccStatsForStage(stageId).foreach { stats =>
          // Invariant: stages with GPU accumulators are tracked by stageManager
          // and therefore cached. The fallback to 0 is defensive for edge cases
          // (e.g. driver-side accumulators) where the stage is absent from the
          // task-metrics cache. It affects only the emitted numTasks column: the
          // SQL/app rollups pool total/count and never read numTasks.
          val numTasks = stageCache.get(stageId).map(_.numTasks).getOrElse {
            logWarning(s"GPU accumulator '$name' references stage $stageId which " +
              s"is not in the stage-task metrics cache; using numTasks = 0.")
            0
          }
          // The row carries what the store holds -- the accumulated total and the number of
          // tasks that reported the metric -- and derives the published sum and avg from them.
          // `stats` has been through readjustTotalStats, which replaces total with max for a
          // max-aggregated metric, so the unadjusted record is the only place the real sum
          // survives, and that sum is the numerator every mean needs.
          val rawStats = ai.getRawStatsForStage(stageId)
          val max = Some(stats.max)
          val row = StageAggGpuMetricsProfileResult(
            stageId = stageId,
            numTasks = numTasks,
            metricName = name,
            unit = unit,
            total = rawStats.map(_.total),
            max = max,
            count = rawStats.map(_.count).getOrElse(0L))
          // Skip rows carrying no signal (both the published sum and max zero/absent).
          if (!(row.sum.forall(_ == 0L) && max.forall(_ == 0L))) {
            rows += row
          }
        }
      }
    }
    rows.toSeq.sortBy(r => (r.stageId, r.metricName))
  }

  /**
   * Rollup helper: groups stage-level GPU rows by metric name and reduces to
   * (unit, sum, max, avg). sum adds the stage sums (None for max metrics); max is
   * the largest stage max; avg divides the pooled stage totals by the pooled stage
   * counts over the stages that recorded the metric. numTasks is intentionally not
   * propagated: see SQLAggGpuMetricsProfileResult / AppAggGpuMetricsProfileResult.
   */
  private def rollupGpuRows(
      rows: Seq[StageAggGpuMetricsProfileResult]
  ): Seq[(String, String, Option[Long], Option[Long], Option[Long])] = {
    rows.groupBy(_.metricName).map { case (metricName, group) =>
      val unit = group.head.unit
      val sumOpt: Option[Long] = {
        val xs = group.flatMap(_.sum)
        if (xs.isEmpty) None else Some(xs.sum)
      }
      val maxOpt: Option[Long] = {
        val xs = group.flatMap(_.max)
        if (xs.isEmpty) None else Some(xs.max)
      }
      // Pooled as sum-of-totals over sum-of-counts, not a weighted average of the stage
      // averages. Two reasons: stage `avg` is already integer-truncated, so re-averaging
      // truncates twice; and the correct weight is the number of tasks that reported the
      // metric, not the stage's task count. GPU accumulables are frequently sparse -- spill
      // and retry metrics land on a handful of tasks in a large stage -- so weighting by
      // numTasks skews the result toward the stages that reported it least, without bound.
      val avgOpt: Option[Long] = {
        val reporting = group.filter(_.count > 0L)
        val totalCount = reporting.map(_.count).sum
        if (totalCount <= 0L) None else Some(reporting.flatMap(_.total).sum / totalCount)
      }
      (metricName, unit, sumOpt, maxOpt, avgOpt)
    }.toSeq
  }

  /**
   * Aggregate GPU task metrics by SQL. Rolls up stage-level rows using
   * app.sqlIdToStages. One row per (sqlId, metricName).
   */
  def aggregateGpuMetricsBySql(
      index: Int,
      stageRows: Seq[StageAggGpuMetricsProfileResult]
  ): Seq[SQLAggGpuMetricsProfileResult] = {
    if (stageRows.isEmpty) {
      return Seq.empty
    }
    val stageMap: Map[Int, Seq[StageAggGpuMetricsProfileResult]] =
      stageRows.groupBy(_.stageId)
    app.sqlIdToStages.toSeq.flatMap { case (sqlId, stageIds) =>
      val rowsForSql: Seq[StageAggGpuMetricsProfileResult] =
        stageIds.toSeq.flatMap(s => stageMap.getOrElse(s, Seq.empty))
      rollupGpuRows(rowsForSql).map { case (metric, unit, sum, max, avg) =>
        SQLAggGpuMetricsProfileResult(
          sqlId = sqlId,
          metricName = metric,
          unit = unit,
          sum = sum,
          max = max,
          avg = avg)
      }
    }.sortBy(r => (r.sqlId, r.metricName))
  }

  /**
   * Aggregate GPU task metrics across the whole application. One row per
   * metricName. For max-aggregated metrics this gives the peak reading any
   * task produced during the run.
   */
  def aggregateGpuMetricsByApp(
      index: Int,
      stageRows: Seq[StageAggGpuMetricsProfileResult]
  ): Seq[AppAggGpuMetricsProfileResult] = {
    if (stageRows.isEmpty) {
      return Seq.empty
    }
    rollupGpuRows(stageRows).map { case (metric, unit, sum, max, avg) =>
      AppAggGpuMetricsProfileResult(
        appId = app.appId,
        metricName = metric,
        unit = unit,
        sum = sum,
        max = max,
        avg = avg)
    }.sortBy(_.metricName)
  }
}

/**
 * Factory for creating the appropriate AppSparkMetricsAnalyzer based on the application type.
 */
object AppSparkMetricsAnalyzer {
  /**
   * Creates an AppSparkMetricsAnalyzer instance appropriate for the given application.
   *
   * For Photon applications, creates a PhotonAppSparkMetricsAnalyzer that handles
   * Photon-specific metric aggregation from accumulators. For other applications,
   * creates a standard AppSparkMetricsAnalyzer.
   *
   * @param app The application to analyze
   * @return AppSparkMetricsAnalyzer or PhotonAppSparkMetricsAnalyzer instance
   */
  def apply(app: AppBase): AppSparkMetricsAnalyzer = {
    if (app.dbPlugin.isPhotonEnabled) {
      new PhotonAppSparkMetricsAnalyzer(app)
    } else {
      new AppSparkMetricsAnalyzer(app)
    }
  }
}
