/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.profiling

import java.io.File

import com.nvidia.spark.rapids.tool.{PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.analysis.MetricCatalog
import com.nvidia.spark.rapids.tool.views.{ProfDataSourceView, RawMetricProfilerView}
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.scheduler.AccumulableInfo
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.rapids.tool.store.{AccumInfo, AccumInfoWithMaxAgg, AccumMetaRef}
import org.apache.spark.sql.types._

case class TestStageDiagnosticResult(
    appName: String,
    appId: String,
    stageId: Long,
    duration: Option[Long],
    numTasks: Int,
    memoryBytesSpilledMBMin: Long,
    memoryBytesSpilledMBMed: Long,
    memoryBytesSpilledMBMax: Long,
    memoryBytesSpilledMBSum: Long,
    diskBytesSpilledMBMin: Long,
    diskBytesSpilledMBMed: Long,
    diskBytesSpilledMBMax: Long,
    diskBytesSpilledMBSum: Long,
    inputBytesReadMin: Long,
    inputBytesReadMed: Long,
    inputBytesReadMax: Long,
    inputBytesReadSum: Long,
    outputBytesWrittenMin: Long,
    outputBytesWrittenMed: Long,
    outputBytesWrittenMax: Long,
    outputBytesWrittenSum: Long,
    srTotalBytesReadMin: Long,
    srTotalBytesReadMed: Long,
    srTotalBytesReadMax: Long,
    srTotalBytesReadSum: Long,
    swBytesWrittenMin: Long,
    swBytesWrittenMed: Long,
    swBytesWrittenMax: Long,
    swBytesWrittenSum: Long,
    srFetchWaitTimeMin: Long,
    srFetchWaitTimeMed: Long,
    srFetchWaitTimeMax: Long,
    srFetchWaitTimeSum: Long,
    swWriteTimeMin: Long,
    swWriteTimeMed: Long,
    swWriteTimeMax: Long,
    swWriteTimeSum: Long,
    gpuSemaphoreWaitSum: Long,
    nodeNames: Seq[String])

case class TestIODiagnosticResult(
    appName: String,
    appId: String,
    sqlId: Long,
    stageId: Long,
    duration: Long,
    nodeId: Long,
    nodeName: String,
    outputRowsMin: Long,
    outputRowsMed: Long,
    outputRowsMax: Long,
    outputRowsSum: Long,
    scanTimeMin: Long,
    scanTimeMed: Long,
    scanTimeMax: Long,
    scanTimeSum: Long,
    outputBatchesMin: Long,
    outputBatchesMed: Long,
    outputBatchesMax: Long,
    outputBatchesSum: Long,
    bufferTimeMin: Long,
    bufferTimeMed: Long,
    bufferTimeMax: Long,
    bufferTimeSum: Long,
    shuffleWriteTimeMin: Long,
    shuffleWriteTimeMed: Long,
    shuffleWriteTimeMax: Long,
    shuffleWriteTimeSum: Long,
    fetchWaitTimeMin: Long,
    fetchWaitTimeMed: Long,
    fetchWaitTimeMax: Long,
    fetchWaitTimeSum: Long,
    gpuDecodeTimeMin: Long,
    gpuDecodeTimeMed: Long,
    gpuDecodeTimeMax: Long,
    gpuDecodeTimeSum: Long)

class AnalysisSuite extends AnyFunSuite {

  private def createTestStageDiagnosticResult(diagnosticsResults: Seq[StageDiagnosticResult]):
      Seq[TestStageDiagnosticResult] = {
    def bytesToMB(numBytes: Long): Long = numBytes / (1024 * 1024)
    def nanoToMilliSec(numNano: Long): Long = numNano / 1000000
    diagnosticsResults.map { result =>
      TestStageDiagnosticResult(
        result.appName,
        result.appId,
        result.stageId,
        result.duration,
        result.numTasks,
        bytesToMB(result.memoryBytesSpilled.min),
        bytesToMB(result.memoryBytesSpilled.median),
        bytesToMB(result.memoryBytesSpilled.max),
        bytesToMB(result.memoryBytesSpilled.total),
        bytesToMB(result.diskBytesSpilled.min),
        bytesToMB(result.diskBytesSpilled.median),
        bytesToMB(result.diskBytesSpilled.max),
        bytesToMB(result.diskBytesSpilled.total),
        result.inputBytesRead.min,
        result.inputBytesRead.median,
        result.inputBytesRead.max,
        result.inputBytesRead.total,
        result.outputBytesWritten.min,
        result.outputBytesWritten.median,
        result.outputBytesWritten.max,
        result.outputBytesWritten.total,
        result.srTotalBytesReadMin,
        result.srTotalBytesReadMed,
        result.srTotalBytesReadMax,
        result.srTotalBytesReadSum,
        result.swBytesWritten.min,
        result.swBytesWritten.median,
        result.swBytesWritten.max,
        result.swBytesWritten.total,
        nanoToMilliSec(result.srFetchWaitTime.min),
        nanoToMilliSec(result.srFetchWaitTime.median),
        nanoToMilliSec(result.srFetchWaitTime.max),
        nanoToMilliSec(result.srFetchWaitTime.total),
        nanoToMilliSec(result.swWriteTime.min),
        nanoToMilliSec(result.swWriteTime.median),
        nanoToMilliSec(result.swWriteTime.max),
        nanoToMilliSec(result.swWriteTime.total),
        result.gpuSemaphoreWait.total,
        result.nodeNames)
    }
  }

  private def createTestIODiagnosticResult(diagnosticsResults: Seq[IODiagnosticResult]):
      Seq[TestIODiagnosticResult] = {
    diagnosticsResults.map {result =>
      TestIODiagnosticResult(
        result.appName,
        result.appId,
        result.sqlId,
        result.stageId,
        result.duration,
        result.nodeId,
        result.nodeName,
        result.outputRows.min,
        result.outputRows.med,
        result.outputRows.max,
        result.outputRows.total,
        result.scanTime.min,
        result.scanTime.med,
        result.scanTime.max,
        result.scanTime.total,
        result.outputBatches.min,
        result.outputBatches.med,
        result.outputBatches.max,
        result.outputBatches.total,
        result.bufferTime.min,
        result.bufferTime.med,
        result.bufferTime.max,
        result.bufferTime.total,
        result.shuffleWriteTime.min,
        result.shuffleWriteTime.med,
        result.shuffleWriteTime.max,
        result.shuffleWriteTime.total,
        result.fetchWaitTime.min,
        result.fetchWaitTime.med,
        result.fetchWaitTime.max,
        result.fetchWaitTime.total,
        result.gpuDecodeTime.min,
        result.gpuDecodeTime.med,
        result.gpuDecodeTime.max,
        result.gpuDecodeTime.total)
    }
  }

  lazy val sparkSession = {
    SparkSession
        .builder()
        .master("local[*]")
        .appName("Rapids Spark Profiling Tool Unit Tests")
        .getOrCreate()
  }

  private val expRoot = ToolTestUtils.getTestResourceFile("ProfilingExpectations")
  private val logDir = ToolTestUtils.getTestResourcePath("spark-events-profiling")
  private val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
  // AutoTuner added a field in SQLTaskAggMetricsProfileResult but it is not among the output
  private val skippedColumnsInSqlAggProfile = Seq("inputBytesReadAvg")

  test("test sqlMetricsAggregation simple") {
    val expectFile = (metric: String) => {
      s"rapids_join_eventlog_${metric}metricsagg_expectation.csv"
    }
    testSqlMetricsAggregation(Array(s"$logDir/rapids_join_eventlog.zstd"),
      expectFile("sql"), expectFile("job"), expectFile("stage"))
  }

  test("test sqlMetricsAggregation second single app") {
    val expectFile = (metric: String) => {
      s"rapids_join_eventlog_${metric}metricsagg2_expectation.csv"
    }
    testSqlMetricsAggregation(Array(s"$logDir/rapids_join_eventlog2.zstd"),
      expectFile("sql"), expectFile("job"), expectFile("stage"))
  }

  test("test photon sql metrics aggregation") {
    val fileName = "nds_q88_photon_db_13_3"
    val expectFile = (metric: String) => {
      s"${fileName}_${metric}_metrics_agg_expectation.csv"
    }
    testSqlMetricsAggregation(Array(s"${qualLogDir}/${fileName}.zstd"),
      expectFile("sql"), expectFile("job"), expectFile("stage"),
      platformName = PlatformNames.DATABRICKS_AWS)
  }

  test("test stage-level diagnostic metrics with diagnostic views enabled") {
    val expectFile = "rapids_join_eventlog_stagediagnosticmetrics_expectation.csv"
    val logs = Array(s"$logDir/rapids_join_eventlog.zstd")
    val logsWithArgs = Array("--enable-diagnostic-views") ++ logs
    val apps = ToolTestUtils.processProfileApps(logsWithArgs, sparkSession)
    assert(apps.size == logs.size)

    // This step is to compute stage to node names and diagnostic metrics mappings,
    // which is used in collecting diagnostic metrics.
    val collect = new CollectInformation(apps.toSeq)
    collect.getSQLToStage
    collect.getStageLevelMetrics

    val diagnosticResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    import org.apache.spark.sql.functions._
    import sparkSession.implicits._
    val actualDf = createTestStageDiagnosticResult(diagnosticResults.stageDiagnostics).toDF.
      withColumn("nodeNames", concat_ws(",", col("nodeNames")))
    compareMetrics(actualDf, expectFile)
  }

  test("test stage-level diagnostic metrics with diagnostic views disabled") {
    val logs = Array(s"$logDir/rapids_join_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    assert(apps.size == logs.size)

    val collect = new CollectInformation(apps.toSeq)
    collect.getSQLToStage
    collect.getStageLevelMetrics

    val diagnosticResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    assert(diagnosticResults.stageDiagnostics.isEmpty)
  }

  test("test IO diagnostic metrics with diagnostic views enabled") {
    val expectFile = "rapids_join_eventlog_iodiagnosticmetrics_expectation.csv"
    val logs = Array(s"$logDir/rapids_join_eventlog.zstd")
    val logsWithArgs = Array("--enable-diagnostic-views") ++ logs
    val apps = ToolTestUtils.processProfileApps(logsWithArgs, sparkSession)
    assert(apps.size == logs.size)

    val collect = new CollectInformation(apps.toSeq)
    // Computes IO diagnostic metrics mapping which is later used in getIODiagnosticMetrics
    collect.getSQLPlanMetrics
    val diagnosticResults = collect.getIODiagnosticMetrics

    import sparkSession.implicits._
    val actualDf = createTestIODiagnosticResult(diagnosticResults).toDF
    compareMetrics(actualDf, expectFile)
  }

  test("test IO diagnostic metrics with diagnostic views disabled") {
    val logs = Array(s"$logDir/rapids_join_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    assert(apps.size == logs.size)

    val collect = new CollectInformation(apps.toSeq)
    collect.getSQLPlanMetrics
    val diagnosticResults = collect.getIODiagnosticMetrics
    assert(diagnosticResults.isEmpty)
  }

  private def testSqlMetricsAggregation(logs: Array[String], expectFileSQL: String,
      expectFileJob: String, expectFileStage: String,
      platformName: String = PlatformNames.DEFAULT): Unit = {
    val args = Array("--platform", platformName) ++ logs
    val apps = ToolTestUtils.processProfileApps(args, sparkSession)
    assert(apps.size == logs.size)
    val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    import sparkSession.implicits._
    // Check the SQL metrics
    val sqlAggsFiltered = aggResults.sqlAggs.toDF.drop(skippedColumnsInSqlAggProfile: _*)
    compareMetrics(sqlAggsFiltered, expectFileSQL)
    // Check the job metrics
    compareMetrics(aggResults.jobAggs.toDF, expectFileJob)
    // Check the stage metrics
    compareMetrics(aggResults.stageAggs.toDF, expectFileStage)
  }

  private def compareMetrics(actualDf: DataFrame, expectFileName: String): Unit = {
    val expectationFile = new File(expRoot, expectFileName)
    val dfExpect = ToolTestUtils.readExpectationCSV(sparkSession, expectationFile.getPath())
    ToolTestUtils.compareDataFrames(actualDf, dfExpect)
  }

  test("test sqlMetrics duration, execute cpu time and potential_problems") {
    val logs = Array(s"$qualLogDir/complex_dec_eventlog.zstd")
    val expectFile = "rapids_duration_and_cpu_expectation.csv"

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    import sparkSession.implicits._
    val sqlAggDurCpu = aggResults.sqlDurAggs
    val resultExpectation = new File(expRoot, expectFile)
    val schema = new StructType()
      .add("appID", StringType, true)
      .add("rootsqlID", LongType, true)
      .add("sqlID", LongType, true)
      .add("sqlDuration", LongType, true)
      .add("containsDataset", BooleanType, true)
      .add("appDuration", LongType, true)
      .add("potentialProbs", StringType, true)
      .add("executorCpuTime", DoubleType, true)
    val actualDf = sqlAggDurCpu.toDF

    val dfExpect = sparkSession.read.option("header", "true").option("nullValue", "-")
      .schema(schema).csv(resultExpectation.getPath())

    ToolTestUtils.compareDataFrames(actualDf, dfExpect)
  }

  test("test shuffleSkewCheck empty") {
    val apps =
      ToolTestUtils.processProfileApps(Array(s"$logDir/rapids_join_eventlog.zstd"), sparkSession)
    assert(apps.size == 1)

    val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    val shuffleSkewInfo = aggResults.taskShuffleSkew
    assert(shuffleSkewInfo.isEmpty)
  }

  test("test contains dataset false") {
    val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
    val logs = Array(s"$qualLogDir/nds_q86_test")

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    val sqlDurAndCpu = aggResults.sqlDurAggs
    val containsDs = sqlDurAndCpu.filter(_.containsDataset === true)
    assert(containsDs.isEmpty)
  }

  test("test contains dataset true") {
    val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
    val logs = Array(s"$qualLogDir/dataset_eventlog")

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    val sqlDurAndCpu = aggResults.sqlDurAggs
    val containsDs = sqlDurAndCpu.filter(_.containsDataset === true)
    assert(containsDs.size == 1)
  }

  test("test duration for null appInfo") {
    val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
    val logs = Array(s"$qualLogDir/dataset_eventlog")

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    apps.foreach { app =>
      app.appMetaData = None
    }
    val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    val metrics = aggResults.sqlDurAggs
    metrics.foreach(m => assert(m.appDuration.get == 0L))
  }

  test("test photon scan metrics") {
    val args = Array(
      "--platform",
      PlatformNames.DATABRICKS_AWS,
      s"$qualLogDir/nds_q88_photon_db_13_3.zstd"
    )
    val apps = ToolTestUtils.processProfileApps(args, sparkSession)
    val dataSourceResults = ProfDataSourceView.getRawView(apps.toSeq)
    assert(dataSourceResults.exists(_.scan_time > 0))
  }

  // ---------------------------------------------------------------------------
  // GPU task metric aggregations (stage / SQL / app)
  // ---------------------------------------------------------------------------

  test("GPU metric aggregation: stage / sql / app rows produced for GPU log") {
    val logs = Array(s"$logDir/gpu_oom_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val agg = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    assert(agg.gpuStageAggs.nonEmpty, "expected stage-level GPU rows")
    assert(agg.gpuSqlAggs.nonEmpty, "expected SQL-level GPU rows")
    assert(agg.gpuAppAggs.nonEmpty, "expected app-level GPU rows")

    // Discovery: the catalog owns the rule; a declared metric qualifies on its family and an
    // undeclared one on the legacy prefixes.
    agg.gpuStageAggs.foreach { row =>
      assert(MetricCatalog.DEFAULT.isGpuReportedMetric(row.metricName),
        s"unexpected metric name: ${row.metricName}")
    }

    // Unit comes from the catalog. Asserted against the catalog rather than re-derived from the
    // name, because re-deriving it here is what let the name-based rules ship broken.
    agg.gpuStageAggs.foreach { row =>
      assert(row.unit == MetricCatalog.DEFAULT.unitFor(row.metricName),
        s"unit mismatch for ${row.metricName}: got ${row.unit}")
    }

    // Every GPU accumulable that carries a non-zero value produces a row. The absence of this
    // assertion is why eight of seventeen metrics could vanish unnoticed: they were reduced to
    // zero by a spurious unit conversion and then dropped by the zero-signal filter. A metric
    // that is genuinely all-zero in the log is still legitimately suppressed.
    val expectedNames = apps.head.accumManager.accumInfoMap.values
      .filter { ai =>
        ai.infoRef.isGpuReportedMetric && ai.getStageIds.exists { sId =>
          ai.calculateAccStatsForStage(sId).exists(st => st.max != 0L || st.total != 0L)
        }
      }
      .map(_.infoRef.getName())
      .toSet
    val emittedNames = agg.gpuStageAggs.map(_.metricName).toSet
    assert(expectedNames.diff(emittedNames).isEmpty,
      s"GPU accumulables with non-zero values but missing from the report: " +
        s"${expectedNames.diff(emittedNames)}")

    // Rollup math: SQL.sum == Σ stage.sum across the SQL's stages, per metric.
    val stageRowsByMetric = agg.gpuStageAggs.groupBy(_.metricName)
    val sqlRowsByMetric = agg.gpuSqlAggs.groupBy(_.metricName)
    sqlRowsByMetric.foreach { case (metric, sqlRows) =>
      val sqlSum = sqlRows.flatMap(_.sum).sum
      val stageSum = stageRowsByMetric.getOrElse(metric, Seq.empty).flatMap(_.sum).sum
      assert(sqlSum == stageSum, s"SQL sum mismatch for $metric: $sqlSum vs $stageSum")
      val sqlMax = sqlRows.flatMap(_.max)
      val stageMax = stageRowsByMetric.getOrElse(metric, Seq.empty).flatMap(_.max)
      if (stageMax.nonEmpty) {
        assert(sqlMax.nonEmpty && sqlMax.max == stageMax.max,
          s"SQL max mismatch for $metric")
      }
    }

    // App row exactly matches the rollup of all stage rows per metric.
    val appByMetric = agg.gpuAppAggs.groupBy(_.metricName).mapValues(_.head)
    stageRowsByMetric.foreach { case (metric, stageRows) =>
      val appRow = appByMetric(metric)
      val expectedSum = stageRows.flatMap(_.sum)
      assert(appRow.sum.map(s => s == expectedSum.sum).getOrElse(expectedSum.isEmpty),
        s"App sum mismatch for $metric")
      val expectedMax = stageRows.flatMap(_.max)
      assert(appRow.max.map(m => m == expectedMax.max).getOrElse(expectedMax.isEmpty),
        s"App max mismatch for $metric")
    }
  }

  test("GPU metric aggregation: max-aggregated metrics carry max and avg but no sum") {
    val logs = Array(s"$logDir/gpu_oom_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val agg = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    // Read from the catalog rather than duplicating the set here: the local copy this replaces
    // drifted silently against the real one.
    val maxOnlyNames = MetricCatalog.DEFAULT.maxAggregatedNames
    val maxRows = agg.gpuStageAggs.filter(r => maxOnlyNames.contains(r.metricName)) ++
      agg.gpuSqlAggs.filter(r => maxOnlyNames.contains(r.metricName)) ++
      agg.gpuAppAggs.filter(r => maxOnlyNames.contains(r.metricName))
    assert(maxRows.nonEmpty, "expected at least one max-aggregated metric row")
    maxRows.foreach { r =>
      val sum = r match {
        case s: StageAggGpuMetricsProfileResult => s.sum
        case s: SQLAggGpuMetricsProfileResult => s.sum
        case s: AppAggGpuMetricsProfileResult => s.sum
      }
      val avg = r match {
        case s: StageAggGpuMetricsProfileResult => s.avg
        case s: SQLAggGpuMetricsProfileResult => s.avg
        case s: AppAggGpuMetricsProfileResult => s.avg
      }
      val max = r match {
        case s: StageAggGpuMetricsProfileResult => s.max
        case s: SQLAggGpuMetricsProfileResult => s.max
        case s: AppAggGpuMetricsProfileResult => s.max
      }
      // sum stays empty: adding per-task peaks is meaningless because they never coexist.
      assert(sum.isEmpty, s"max-aggregated metric should have empty sum: $r")
      // avg is populated: it is the mean, over the tasks that reported the metric, of each
      // task's reported value -- a different quantity from the sum, and previously discarded.
      assert(avg.isDefined, s"max-aggregated metric should carry an avg: $r")
      assert(max.isDefined, s"max-aggregated metric should have a max value: $r")
    }
  }

  test("GPU metric aggregation: empty for CPU-only event log") {
    val logs = Array(s"$qualLogDir/nds_q86_test")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val agg = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    assert(agg.gpuStageAggs.isEmpty, "CPU-only log should produce no GPU stage rows")
    assert(agg.gpuSqlAggs.isEmpty, "CPU-only log should produce no GPU SQL rows")
    assert(agg.gpuAppAggs.isEmpty, "CPU-only log should produce no GPU app rows")
  }

  test("GPU metric avg is the arithmetic mean, not the ratcheting rolling mean") {
    // `med` is a rolling mean recomputed with integer division on every update, so it drifts to
    // the floor: on a real log it reports 1 for a metric whose true mean is 5.89. avg must be
    // the raw sum over the raw count instead. Oracle computed here from the store's own totals,
    // independently of how the analyzer builds the row.
    val logs = Array(s"$logDir/gpu_oom_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val agg = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    // Keyed by (name, stageId), NOT by name: accumInfoMap is keyed by accumulator id and each
    // metric gets a distinct id per stage, so a name-keyed map keeps one entry and silently
    // skips every other stage's rows.
    val rawByNameAndStage = apps.head.accumManager.accumInfoMap.values
      .filter(_.infoRef.isGpuReportedMetric)
      .flatMap(ai => ai.getStageIds.flatMap(sId =>
        ai.getRawStatsForStage(sId).map(raw => (ai.infoRef.getName(), sId) -> raw)))
      .toMap
    var checked = 0
    agg.gpuStageAggs.foreach { row =>
      val raw = rawByNameAndStage.get((row.metricName, row.stageId))
      assert(raw.isDefined, s"no raw stats for ${row.metricName} stage ${row.stageId}")
      val expected = if (raw.get.count > 0L) Some(raw.get.total / raw.get.count) else None
      assert(row.avg == expected,
        s"${row.metricName} stage ${row.stageId}: avg ${row.avg} != sum/count $expected")
      checked += 1
    }
    // every emitted row is checked, not an arbitrary subset
    assert(checked == agg.gpuStageAggs.size, s"checked $checked of ${agg.gpuStageAggs.size}")
    assert(checked > 0, "expected at least one GPU row to check")
  }

  test("a max-aggregated metric publishes max and avg but never sum") {
    val logs = Array(s"$logDir/gpu_oom_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val agg = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    val maxRows = agg.gpuStageAggs.filter(r =>
      MetricCatalog.DEFAULT.isAggregatedByMax(r.metricName))
    assert(maxRows.nonEmpty)
    maxRows.foreach { r =>
      assert(r.sum.isEmpty, s"per-task peaks must not be summed: $r")
      assert(r.max.isDefined && r.avg.isDefined, s"max and avg both expected: $r")
      // and avg must not have collapsed onto max or onto the floor
      assert(r.avg.get <= r.max.get, s"avg above max: $r")
    }
  }

  test("SQL and app avg pool the reporting tasks rather than re-averaging stage means") {
    // The rollup used to weight each stage mean by the stage's task count. That denominator
    // counts tasks which never reported the metric, and GPU accumulables are often sparse, so
    // the published mean drifted toward the stages that reported it least. It also truncated
    // twice, once at stage level and again in the rollup.
    val logs = Array(s"$logDir/gpu_oom_eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val agg = RawMetricProfilerView.getAggMetrics(apps.toSeq)
    // Oracle read straight from the store, keyed by (name, stageId) because accumInfoMap is
    // keyed by accumulator id and each metric gets a distinct id per stage.
    val rawByNameAndStage = apps.head.accumManager.accumInfoMap.values
      .filter(_.infoRef.isGpuReportedMetric)
      .flatMap(ai => ai.getStageIds.flatMap(sId =>
        ai.getRawStatsForStage(sId).map(raw => (ai.infoRef.getName(), sId) -> raw)))
      .toMap
    // Restricted to the stages that actually produced a row, so the zero-signal filter in
    // aggregateGpuMetricsByStage does not make the oracle disagree for the wrong reason.
    val pooled = agg.gpuStageAggs.groupBy(_.metricName).map { case (name, group) =>
      val stats = group.flatMap(r => rawByNameAndStage.get((name, r.stageId)))
        .filter(_.count > 0L)
      val pooledTotal = stats.map(_.total).sum
      val pooledCount = stats.map(_.count).sum
      name -> ((pooledTotal, pooledCount))
    }
    assert(agg.gpuAppAggs.nonEmpty, "expected app-level GPU rows")
    agg.gpuAppAggs.foreach { row =>
      val (total, count) = pooled(row.metricName)
      assert(count > 0L, s"no reporting tasks for ${row.metricName}")
      assert(row.avg.contains(total / count),
        s"${row.metricName}: avg ${row.avg} != pooled $total/$count")
    }
    // SQL rows pool over that SQL's own stage set, derived here rather than reusing the
    // app-level pooling above.
    val sqlToStages = apps.head.sqlIdToStages
    assert(agg.gpuSqlAggs.nonEmpty, "expected SQL-level GPU rows")
    agg.gpuSqlAggs.foreach { row =>
      val stageIds = sqlToStages.getOrElse(row.sqlId, Seq.empty).toSet
      val stats = agg.gpuStageAggs
        .filter(r => r.metricName == row.metricName && stageIds.contains(r.stageId))
        .flatMap(r => rawByNameAndStage.get((r.metricName, r.stageId)))
        .filter(_.count > 0L)
      val sqlCount = stats.map(_.count).sum
      assert(sqlCount > 0L, s"no reporting tasks for ${row.metricName} in SQL ${row.sqlId}")
      assert(row.avg.contains(stats.map(_.total).sum / sqlCount),
        s"SQL ${row.sqlId} ${row.metricName}: avg ${row.avg} is not the pooled mean")
    }

    // Regression guard: on this fixture the two formulas genuinely disagree for at least one
    // metric, so reverting to the task-count weighting fails here instead of passing silently.
    val byNumTasks = agg.gpuStageAggs.groupBy(_.metricName).map { case (name, group) =>
      val weighted = group.flatMap(r => r.avg.map(a => (a, r.numTasks.toLong)))
      val tasks = weighted.map(_._2).sum
      name -> (if (tasks == 0L) None else Some(weighted.map(p => p._1 * p._2).sum / tasks))
    }
    assert(agg.gpuAppAggs.exists(row => byNumTasks(row.metricName) != row.avg),
      "fixture no longer distinguishes the two rollup formulas; the guard is now vacuous")
  }

  test("a scaled metric renders divided in the emitted row") {
    // Pins the published strings rather than the stored Longs: the store holds thousandths.
    // total and count are the stored inputs; sum and avg are derived on the way out.
    val row = StageAggGpuMetricsProfileResult(
      stageId = 1, numTasks = 72, metricName = "gpuOnGpuTasksWaitingGPUAvgCount",
      unit = "count", total = Some(3570L), max = Some(2500L), count = 5L)
    // the stored total is present; it is the publishing of it that is suppressed
    assert(row.total.isDefined && row.sum.isEmpty, "max-aggregated total must stay unpublished")
    assert(row.convertToCSVSeq().toSeq == Seq("1", "72", "gpuOnGpuTasksWaitingGPUAvgCount",
      "count", "", "2.5", "0.714"))
    // an unscaled metric is byte-identical to before
    val plain = StageAggGpuMetricsProfileResult(
      stageId = 1, numTasks = 72, metricName = "gpuMaxTaskFootprint",
      unit = "bytes", total = Some(4234820190L), max = Some(7123115846L), count = 3L)
    assert(plain.convertToCSVSeq().toSeq == Seq("1", "72", "gpuMaxTaskFootprint",
      "bytes", "", "7123115846", "1411606730"))
  }

  test("every render site divides out the scale, not just the stage-level one") {
    // Four independent copies of render() exist. The SQL and app rows look up the scale by
    // metric name; AccumProfileResults reads it from its AccumMetaRef instead, a different
    // source, and had no test at all.
    val name = "gpuOnGpuTasksWaitingGPUAvgCount"
    val sqlRow = SQLAggGpuMetricsProfileResult(
      sqlId = 0, metricName = name, unit = "count",
      sum = None, max = Some(2500L), avg = Some(714L))
    assert(sqlRow.convertToCSVSeq().toSeq.takeRight(3) == Seq("", "2.5", "0.714"))
    val appRow = AppAggGpuMetricsProfileResult(
      appId = "app-1", metricName = name, unit = "count",
      sum = None, max = Some(2500L), avg = Some(714L))
    assert(appRow.convertToCSVSeq().toSeq.takeRight(3) == Seq("", "2.5", "0.714"))
    // AccumProfileResults, which feeds stage_level_all_metrics.csv
    val ref = AccumMetaRef(116L, Some(name))
    assert(ref.storageScale == 1000L, "precondition: the metric is scaled")
    val accRow = AccumProfileResults(1, ref, min = 0L, median = 714L, max = 2500L, total = 2500L)
    assert(accRow.convertToCSVSeq().toSeq.takeRight(4) == Seq("0", "0.714", "2.5", "2.5"))
    // and an unscaled accumulable is unchanged
    val plainRef = AccumMetaRef(104L, Some("gpuMaxTaskFootprint"))
    assert(plainRef.storageScale == 1L)
    val plainAcc = AccumProfileResults(1, plainRef, 1L, 2L, 3L, 6L)
    assert(plainAcc.convertToCSVSeq().toSeq.takeRight(4) == Seq("1", "2", "3", "6"))
  }

  test("a decimal metric round-trips from raw event value to rendered cell") {
    // End to end through the store: parse -> fixed-point storage -> render. The only GPU
    // fixture emits "0" for this metric on every task, so without this the whole path is
    // exercised nowhere above the isolated unit tests.
    val ref = AccumMetaRef(116L, Some("gpuOnGpuTasksWaitingGPUAvgCount"))
    val info = AccumInfo(ref)
    assert(info.isInstanceOf[AccumInfoWithMaxAgg], "declared aggregation: max")
    Seq("0.5", "1.0", "2.5", "0").foreach { v =>
      info.addAccumToTask(7, AccumulableInfo(116L, Some("gpuOnGpuTasksWaitingGPUAvgCount"),
        Some(v), None, internal = false, countFailedValues = false, None))
    }
    val raw = info.getRawStatsForStage(7)
    assert(raw.isDefined)
    // stored in thousandths: 500 + 1000 + 2500 + 0
    assert(raw.get.total == 4000L, s"stored total ${raw.get.total}")
    assert(raw.get.count == 4L)
    assert(raw.get.max == 2500L)
    val row = AccumProfileResults(7, ref, raw.get.min, raw.get.total / raw.get.count,
      raw.get.max, raw.get.max)
    assert(row.convertToCSVSeq().toSeq.takeRight(4) == Seq("0", "1", "2.5", "2.5"))
  }
}
