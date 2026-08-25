/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.tuning

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.{AppSummaryInfoBaseProvider, EventLogPathProcessor,
  PlatformFactory, PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.analysis.AggRawMetricsResult
import com.nvidia.spark.rapids.tool.profiling.{ApplicationSummaryInfo, CollectInformation,
  DataSourceProfileResult, ProfileArgs, PySparkMemoryEvidence, SingleAppSummaryInfoProvider,
  StageAggTaskMetricsProfileResult}
import com.nvidia.spark.rapids.tool.views.RawMetricProfilerView

import org.apache.spark.sql.rapids.tool.AccumToStageRetriever
import org.apache.spark.sql.rapids.tool.plangraph.{SparkPlanGraph, SparkPlanGraphNode,
  SQLPlanMetric, ToolsPlanGraph}
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.util.RapidsToolsConfUtil

class FileScanInputMetricsSuite extends ProfilingAutoTunerSuiteBase {
  private val oneMiB = 1024L * 1024L

  private case class GraphNodeSpec(
      id: Long,
      name: String,
      desc: String,
      stageId: Int)

  private def buildGraph(specs: Seq[GraphNodeSpec]): ToolsPlanGraph = {
    val accumIdToStage = specs.map(spec => (1000L + spec.id) -> spec.stageId).toMap
    val nodes = specs.map { spec =>
      new SparkPlanGraphNode(
        spec.id,
        spec.name,
        spec.desc,
        Seq(SQLPlanMetric("number of output rows", 1000L + spec.id, "sum")))
    }
    val retriever = new AccumToStageRetriever {
      override def getStageIDsFromAccumIds(accumIds: Seq[Long]): Set[Int] = {
        accumIds.flatMap(accumIdToStage.get).toSet
      }
    }
    new ToolsPlanGraph(SparkPlanGraph(nodes, Seq.empty), retriever)
  }

  private def dataSource(
      sqlId: Long,
      version: Int,
      nodeId: Long,
      fromFinalPlan: Boolean = true): DataSourceProfileResult = {
    DataSourceProfileResult(
      sqlId, version, nodeId, "Parquet", 0L, 0L, 0L, 0L,
      "file:/input", "", "struct<id:long>", "", "", fromFinalPlan)
  }

  private def stageMetric(
      stageId: Long,
      maxInputBytes: Long,
      numTasks: Int = 1): StageAggTaskMetricsProfileResult = {
    StageAggTaskMetricsProfileResult(
      id = stageId,
      numTasks = numTasks,
      duration = None,
      diskBytesSpilledSum = 0L,
      durationSum = 0L,
      durationMax = 0L,
      durationMin = 0L,
      durationAvg = 0.0,
      executorCPUTimeSum = 0L,
      executorDeserializeCpuTimeSum = 0L,
      executorDeserializeTimeSum = 0L,
      executorRunTimeSum = 0L,
      inputBytesReadSum = maxInputBytes,
      inputBytesReadMax = maxInputBytes,
      inputRecordsReadSum = 0L,
      jvmGCTimeSum = 0L,
      memoryBytesSpilledSum = 0L,
      outputBytesWrittenSum = 0L,
      outputRecordsWrittenSum = 0L,
      peakExecutionMemoryMax = 0L,
      resultSerializationTimeSum = 0L,
      resultSizeMax = 0L,
      srFetchWaitTimeSum = 0L,
      srLocalBlocksFetchedSum = 0L,
      srcLocalBytesReadSum = 0L,
      srRemoteBlocksFetchSum = 0L,
      srRemoteBytesReadSum = 0L,
      srRemoteBytesReadToDiskSum = 0L,
      srTotalBytesReadSum = 0L,
      swBytesWrittenSum = 0L,
      swRecordsWrittenSum = 0L,
      swWriteTimeSum = 0L)
  }

  private def rawMetrics(stages: Seq[StageAggTaskMetricsProfileResult]): AggRawMetricsResult = {
    AggRawMetricsResult(
      jobAggs = Seq.empty,
      stageAggs = stages,
      taskShuffleSkew = Seq.empty,
      sqlAggs = Seq.empty,
      ioAggs = Seq.empty,
      sqlDurAggs = Seq.empty,
      stageDiagnostics = Seq.empty)
  }

  private def appSummary(
      dataSources: Seq[DataSourceProfileResult],
      stages: Seq[StageAggTaskMetricsProfileResult]): ApplicationSummaryInfo = {
    ApplicationSummaryInfo(
      appInfo = Seq.empty,
      dsInfo = dataSources,
      execInfo = Seq.empty,
      jobInfo = Seq.empty,
      rapidsProps = Seq.empty,
      rapidsJar = Seq.empty,
      sqlMetrics = Seq.empty,
      stageMetrics = Seq.empty,
      jobAggMetrics = Seq.empty,
      stageAggMetrics = stages,
      sqlTaskAggMetrics = Seq.empty,
      durAndCpuMet = Seq.empty,
      skewInfo = Seq.empty,
      failedTasks = Seq.empty,
      failedStages = Seq.empty,
      failedJobs = Seq.empty,
      removedBMs = Seq.empty,
      removedExecutors = Seq.empty,
      unsupportedOps = Seq.empty,
      sparkProps = Seq.empty,
      sqlStageInfo = Seq.empty,
      wholeStage = Seq.empty,
      appLogPath = Seq.empty,
      ioMetrics = Seq.empty,
      sysProps = Seq.empty,
      sqlCleanedAlignedIds = Seq.empty,
      sparkRapidsBuildInfo = Seq.empty,
      writeOpsInfo = Seq.empty,
      sqlPlanInfo = Seq.empty)
  }

  private class TestQualProvider(
      dataSources: Seq[DataSourceProfileResult],
      stages: Seq[StageAggTaskMetricsProfileResult],
      graph: Option[ToolsPlanGraph],
      properties: Map[String, String] = Map.empty)
    extends QualAppSummaryInfoProvider(null, None, rawMetrics(stages), dataSources) {
    override protected[tool] def planGraphForSqlVersion(
        sqlId: Long, version: Int): Option[ToolsPlanGraph] = graph
    override def getAllProperties: Map[String, String] = properties
    override def getSparkProperty(propKey: String): Option[String] = properties.get(propKey)
    override def getRapidsProperty(propKey: String): Option[String] = properties.get(propKey)
    override def getSystemProperty(propKey: String): Option[String] = None
    override def getSparkVersion: Option[String] = Some(testSparkVersion)
    override def hasSqlCacheEvidence: Boolean = false
    override def getPySparkMemoryEvidence: Seq[PySparkMemoryEvidence] = Seq.empty
    override def getClassPathEntries: Map[String, String] = Map.empty
  }

  private class TestProfilingProvider(
      dataSources: Seq[DataSourceProfileResult],
      stages: Seq[StageAggTaskMetricsProfileResult],
      graph: Option[ToolsPlanGraph],
      properties: Map[String, String] = Map.empty)
    extends SingleAppSummaryInfoProvider(null, appSummary(dataSources, stages)) {
    override protected[tool] def planGraphForSqlVersion(
        sqlId: Long, version: Int): Option[ToolsPlanGraph] = graph
    override def getAllProperties: Map[String, String] = properties
    override def getSparkProperty(propKey: String): Option[String] = properties.get(propKey)
    override def getRapidsProperty(propKey: String): Option[String] = properties.get(propKey)
    override def getSystemProperty(propKey: String): Option[String] = None
    override def getSparkVersion: Option[String] = Some(testSparkVersion)
    override def hasSqlCacheEvidence: Boolean = false
    override def getPySparkMemoryEvidence: Seq[PySparkMemoryEvidence] = Seq.empty
    override def scanStagesWithGpuOom: Set[Long] = Set.empty
    override def getClassPathEntries: Map[String, String] = Map.empty
  }

  private def maxPartitionRecommendation(
      provider: AppSummaryInfoBaseProvider,
      helper: AutoTunerHelper,
      properties: Map[String, String]): Option[String] = {
    val platform = PlatformFactory.createInstance(PlatformNames.EMR)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 1,
      numExecs = 2,
      numExecutorNodes = 2,
      sparkProperties = properties,
      systemProperties = Map.empty)
    helper.buildAutoTunerFromProps(provider, platform).getRecommendedProperties()._1
      .find(_.name == "spark.sql.files.maxPartitionBytes")
      .map(_.getTuneValue())
  }

  test("base provider reports no reliable file scan input") {
    assert(new AppSummaryInfoBaseProvider().getMaxFileScanInput.isEmpty)
  }

  test("AutoTuner test provider can represent an absent reliable input") {
    val provider = getMockInfoProvider(
      maxInput = 3292825256.0,
      spilledMetrics = Seq.empty,
      jvmGCFractions = Seq.empty,
      propsFromLog = mutable.Map.empty[String, String],
      sparkVersion = Some(testSparkVersion),
      maxFileScanInputOverride = Some(None))
    assert(provider.getMaxFileScanInput.isEmpty)
  }

  test("qualification provider rejects the CI34 257 MiB cache-only input") {
    val graph = buildGraph(Seq(
      GraphNodeSpec(10L, "InMemoryTableScan", "InMemoryRelation cached", 20)))
    val properties = Map(
      "spark.master" -> "yarn",
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "16g",
      "spark.sql.files.maxPartitionBytes" -> "256m")
    val provider = new TestQualProvider(
      Seq(dataSource(1L, 1, 10L)), Seq(stageMetric(20L, 269751712L, 500)), Some(graph),
      properties)

    assert(provider.getMaxFileScanInput.isEmpty)
    assert(maxPartitionRecommendation(
      provider, QualificationAutoTunerHelper, properties).isEmpty)
  }

  test("profiling provider rejects the CI34 3140 MiB cache-only input") {
    val graph = buildGraph(Seq(
      GraphNodeSpec(10L, "InMemoryTableScan", "InMemoryRelation cached", 20)))
    val properties = Map(
      "spark.master" -> "yarn",
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "16g",
      "spark.sql.files.maxPartitionBytes" -> "254m")
    val provider = new TestProfilingProvider(
      Seq(dataSource(1L, 1, 10L)), Seq(stageMetric(20L, 3292825256L, 96)), Some(graph),
      properties)

    assert(provider.getMaxFileScanInput.isEmpty)
    assert(maxPartitionRecommendation(provider, ProfilingAutoTunerHelper, properties).isEmpty)
  }

  test("both concrete providers expose genuine file scan input") {
    val graph = buildGraph(Seq(GraphNodeSpec(10L, "Scan parquet", "file scan", 20)))
    val dataSources = Seq(dataSource(1L, 1, 10L))
    val stages = Seq(stageMetric(20L, 512L * oneMiB))

    assert(new TestQualProvider(dataSources, stages, Some(graph))
      .getMaxFileScanInput.contains(512.0 * oneMiB))
    assert(new TestProfilingProvider(dataSources, stages, Some(graph))
      .getMaxFileScanInput.contains(512.0 * oneMiB))
  }

  test("uses the largest genuine final-plan file scan stage input") {
    val graph = buildGraph(Seq(
      GraphNodeSpec(10L, "Scan parquet", "file scan", 10),
      GraphNodeSpec(11L, "BatchScan parquet", "data source scan", 11)))
    val dataSources = Seq(dataSource(1L, 1, 10L), dataSource(1L, 1, 11L))
    val stageMetrics = Seq(stageMetric(10L, 300L * oneMiB), stageMetric(11L, 192L * oneMiB))

    val result = FileScanInputMetrics.maxInputBytes(
      dataSources, stageMetrics, (_, version) => if (version == 1) Some(graph) else None)

    assert(result.contains(300.0 * oneMiB))
  }

  test("excludes stages containing InMemoryTableScan or InMemoryRelation") {
    val graph = buildGraph(Seq(
      GraphNodeSpec(10L, "Scan parquet", "file scan", 20),
      GraphNodeSpec(12L, "InMemoryTableScan", "InMemoryRelation cached", 20),
      GraphNodeSpec(11L, "Scan parquet", "genuine file scan", 10)))
    val dataSources = Seq(dataSource(1L, 1, 10L), dataSource(1L, 1, 11L))
    val stageMetrics = Seq(stageMetric(20L, 3140L * oneMiB), stageMetric(10L, 300L * oneMiB))

    val result = FileScanInputMetrics.maxInputBytes(
      dataSources, stageMetrics, (_, _) => Some(graph))

    assert(result.contains(300.0 * oneMiB))
  }

  test("returns None without a trustworthy file scan") {
    val stageMetrics = Seq(stageMetric(20L, 3140L * oneMiB))

    assert(FileScanInputMetrics.maxInputBytes(
      Seq.empty, stageMetrics, (_, _) => None).isEmpty)
  }

  test("AQE version mismatch cannot reuse a latest-plan node ID") {
    val graph = buildGraph(Seq(GraphNodeSpec(10L, "Scan parquet", "file scan", 20)))
    val staleDataSource = dataSource(1L, 0, 10L)
    val stageMetrics = Seq(stageMetric(20L, 300L * oneMiB))

    assert(FileScanInputMetrics.maxInputBytes(
      Seq(staleDataSource), stageMetrics,
      (_, version) => if (version == 1) Some(graph) else None).isEmpty)
  }

  test("nullable and empty plan descriptions are safe") {
    val nullDescGraph = buildGraph(Seq(GraphNodeSpec(10L, "Scan parquet", null, 20)))
    val emptyDescGraph = buildGraph(Seq(GraphNodeSpec(10L, "Scan parquet", "", 20)))
    val cacheGraph = buildGraph(Seq(GraphNodeSpec(10L, "InMemoryTableScan", null, 20)))
    val dataSources = Seq(dataSource(1L, 1, 10L))
    val stageMetrics = Seq(stageMetric(20L, 300L * oneMiB))

    assert(FileScanInputMetrics.maxInputBytes(
      dataSources, stageMetrics, (_, _) => Some(nullDescGraph)).nonEmpty)
    assert(FileScanInputMetrics.maxInputBytes(
      dataSources, stageMetrics, (_, _) => Some(emptyDescGraph)).nonEmpty)
    assert(FileScanInputMetrics.maxInputBytes(
      dataSources, stageMetrics, (_, _) => Some(cacheGraph)).isEmpty)
  }

  test("rejects non-final, missing, unmapped, and zero-only scan inputs") {
    val mappedGraph = buildGraph(Seq(GraphNodeSpec(10L, "Scan parquet", "file scan", 20)))
    val unmappedGraph = buildGraph(Seq(GraphNodeSpec(11L, "Scan parquet", "file scan", 20)))
    val positiveStage = Seq(stageMetric(20L, 300L * oneMiB))

    assert(FileScanInputMetrics.maxInputBytes(
      Seq(dataSource(1L, 1, 10L, fromFinalPlan = false)), positiveStage,
      (_, _) => Some(mappedGraph)).isEmpty)
    assert(FileScanInputMetrics.maxInputBytes(
      Seq(dataSource(1L, 1, 10L)), positiveStage, (_, _) => None).isEmpty)
    assert(FileScanInputMetrics.maxInputBytes(
      Seq(dataSource(1L, 1, 10L)), positiveStage,
      (_, _) => Some(unmappedGraph)).isEmpty)
    assert(FileScanInputMetrics.maxInputBytes(
      Seq(dataSource(1L, 1, 10L)), Seq(stageMetric(20L, 0L)),
      (_, _) => Some(mappedGraph)).isEmpty)
  }

  test("real AQE final Parquet data source binds to the latest plan version") {
    val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
    val eventLogPath = ToolTestUtils.getTestResourcePath(
      "spark-events-qualification/nds_q72_dataproc_2_2.zstd")
    val appArgs = new ProfileArgs(Array(eventLogPath))
    val eventLogInfo = EventLogPathProcessor
      .getEventLogInfo(appArgs.eventlog().head, hadoopConf).head._1
    val app = new ApplicationInfo(hadoopConf, eventLogInfo)
    val collect = new CollectInformation(Seq(app))
    val dataSources = collect.getDataSourceInfo.filter { ds =>
      ds.fromFinalPlan && ds.format.toLowerCase.contains("parquet")
    }
    val stageMetrics = RawMetricProfilerView.getAggMetrics(Seq(app)).stageAggs

    assert(dataSources.nonEmpty)
    dataSources.foreach { ds =>
      val latestVersion = app.sqlManager.getPlanById(ds.sqlID).map(_.plan.version)
      assert(latestVersion.contains(ds.version))
      assert(FileScanInputMetrics.latestPlanGraph(app, ds.sqlID, ds.version - 1).isEmpty)
    }
    val result = FileScanInputMetrics.maxInputBytes(
      dataSources, stageMetrics,
      (sqlId, version) => FileScanInputMetrics.latestPlanGraph(app, sqlId, version))
    val mappings = dataSources.map { ds =>
      val stages = FileScanInputMetrics.latestPlanGraph(app, ds.sqlID, ds.version)
        .map { graph =>
          (graph.getNodeStageRawAssignment(ds.nodeId),
            graph.getNodeStageLogicalAssignment(ds.nodeId))
        }.getOrElse((Set.empty, Set.empty))
      (ds.sqlID, ds.version, ds.nodeId, ds.format, stages)
    }
    val graphScanNodes = FileScanInputMetrics.latestPlanGraph(
      app, dataSources.head.sqlID, dataSources.head.version).toSeq.flatMap { graph =>
      graph.allNodes.filter(node => Option(node.name).exists(_.toLowerCase.contains("scan")))
        .map { node =>
          (node.id, node.name, graph.getNodeStageRawAssignment(node.id),
            graph.getNodeStageLogicalAssignment(node.id), node.metrics.map(_.accumulatorId))
        }
    }
    assert(result.nonEmpty, s"dataSources=$mappings stageInputs=" +
      stageMetrics.map(metric => metric.id -> metric.inputBytesReadMax) +
      s" graphScanNodes=$graphScanNodes")
  }
}
