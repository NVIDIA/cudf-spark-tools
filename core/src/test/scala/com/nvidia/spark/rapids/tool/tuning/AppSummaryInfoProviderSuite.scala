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

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.nvidia.spark.rapids.BaseNoSparkSuite
import com.nvidia.spark.rapids.tool.{AppSummaryInfoBaseProvider, EventLogPathProcessor,
  PlatformFactory, PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.analysis.AggRawMetricsResult
import com.nvidia.spark.rapids.tool.profiling.{AppInfoProfileResults, ApplicationSummaryInfo,
  RapidsPropertyProfileResult, SingleAppSummaryInfoProvider, SQLPlanInfoProfileResult}
import com.nvidia.spark.rapids.tool.tuning.config.{ProfTuningConfigProvider, TuningConfigProvider}
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.execution.{SparkPlanInfo => UpstreamSparkPlanInfo}
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.store.SparkPlanInfoTruncated
import org.apache.spark.sql.rapids.tool.util.{RapidsToolsConfUtil, SparkRuntime}

class AppSummaryInfoProviderSuite extends BaseNoSparkSuite {

  private val cacheSerializerProperty = AutoTuner.CACHE_SERIALIZER_PROPERTY

  private val sparkProperties = Map(
    "spark.master" -> "yarn",
    "spark.executor.cores" -> "8",
    "spark.executor.instances" -> "2",
    "spark.executor.memory" -> "16g",
    "spark.dynamicAllocation.enabled" -> "false")

  private val emptyAggMetrics = AggRawMetricsResult(
    Seq.empty, Seq.empty, Seq.empty, Seq.empty, Seq.empty, Seq.empty, Seq.empty)

  private def plan(name: String, children: UpstreamSparkPlanInfo*): UpstreamSparkPlanInfo = {
    new UpstreamSparkPlanInfo(name, name, children, Map.empty, Seq.empty)
  }

  private def addPlanVersions(app: org.apache.spark.sql.rapids.tool.AppBase): Unit = {
    app.sqlManager.addNewExecution(1L, plan("Project"), "primary")
    app.sqlManager.addAQE(1L,
      plan("AdaptiveSparkPlan", plan("InMemoryTableScan", plan("InMemoryRelation"))),
      "aqe-cache")
    app.sqlManager.addAQE(1L, plan("Project"), "aqe-final")
  }

  private def emptySummary(
      sqlPlanInfo: Seq[SQLPlanInfoProfileResult] = Seq.empty): ApplicationSummaryInfo = {
    val summarySparkProperties = sparkProperties.toSeq.map { case (key, value) =>
      RapidsPropertyProfileResult(key, Array(key, value))
    }
    ApplicationSummaryInfo(
      appInfo = Seq(AppInfoProfileResults(
        appName = "ProviderTest",
        appId = Some("local-provider-test"),
        attemptId = None,
        sparkUser = "test",
        startTime = 100000L,
        endTime = Some(200000L),
        duration = Some(100000L),
        durationStr = "1m 40s",
        sparkRuntime = SparkRuntime.SPARK_RAPIDS,
        sparkVersion = "3.5.0",
        pluginEnabled = true,
        totalCoreSeconds = 0L)),
      dsInfo = Seq.empty,
      execInfo = Seq.empty,
      jobInfo = Seq.empty,
      rapidsProps = Seq.empty,
      rapidsJar = Seq.empty,
      sqlMetrics = Seq.empty,
      stageMetrics = Seq.empty,
      jobAggMetrics = Seq.empty,
      stageAggMetrics = Seq.empty,
      sqlTaskAggMetrics = Seq.empty,
      durAndCpuMet = Seq.empty,
      skewInfo = Seq.empty,
      failedTasks = Seq.empty,
      failedStages = Seq.empty,
      failedJobs = Seq.empty,
      removedBMs = Seq.empty,
      removedExecutors = Seq.empty,
      unsupportedOps = Seq.empty,
      sparkProps = summarySparkProperties,
      sqlStageInfo = Seq.empty,
      wholeStage = Seq.empty,
      appLogPath = Seq.empty,
      ioMetrics = Seq.empty,
      sysProps = Seq.empty,
      sqlCleanedAlignedIds = Seq.empty,
      sparkRapidsBuildInfo = Seq.empty,
      writeOpsInfo = Seq.empty,
      sqlPlanInfo = sqlPlanInfo)
  }

  private def cacheSerializerRecommendation(
      provider: AppSummaryInfoBaseProvider,
      helper: AutoTunerHelper,
      excludedProperties: List[String] = List.empty): Option[String] = {
    val targetCluster = if (excludedProperties.nonEmpty) {
      Some(ToolTestUtils.buildTargetClusterInfo(excludeSparkProperties = excludedProperties))
    } else {
      None
    }
    val platform = PlatformFactory.createInstance(PlatformNames.EMR, targetCluster)
    platform.configureClusterInfoFromEventLog(
      coresPerExecutor = 8,
      execsPerNode = 1,
      numExecs = 2,
      numExecutorNodes = 2,
      sparkProperties,
      systemProperties = Map.empty)
    val (recommendations, _) = helper.buildAutoTunerFromProps(provider, platform)
      .getRecommendedProperties(showOnlyUpdatedProps = false)
    recommendations.find(_.name == cacheSerializerProperty).map(_.getTuneValue())
  }

  private def configuredCacheSerializer: String = {
    TuningConfigProvider.builder.build[ProfTuningConfigProvider]
      .getEntry(AutoTuner.CACHE_SERIALIZER_CONFIG).getDefault
  }

  private def withMinimalEventLog(f: String => Unit): Unit = {
    withMinimalEventLog(Seq.empty)(f)
  }

  private def withMinimalEventLog(
      extraEvents: Seq[String])(f: String => Unit): Unit = {
    val events = Seq(
      """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}""",
      """{"Event":"SparkListenerApplicationStart","App Name":"ProviderTest",""" +
        """"App ID":"local-provider-test","Timestamp":100000,"User":"test"}""") ++
      extraEvents ++ Seq(
      """{"Event":"SparkListenerApplicationEnd","Timestamp":200000}""")
    TrampolineUtil.withTempDir { tempDir =>
      val eventLog = Paths.get(tempDir.getAbsolutePath, "eventlog")
      Files.write(eventLog, events.mkString("\n").getBytes(StandardCharsets.UTF_8))
      f(eventLog.toString)
    }
  }

  private def stageSubmittedWithScope(scopeName: String): String = {
    s"""{"Event":"SparkListenerStageSubmitted","Stage Info":{"Stage ID":0,""" +
      s""""Stage Attempt ID":0,"Stage Name":"cache stage","Number of Tasks":1,""" +
      s""""RDD Info":[{"RDD ID":0,"Name":"MapPartitionsRDD",""" +
      s""""Scope":"{\\"id\\":\\"1\\",\\"name\\":\\"$scopeName\\"}",""" +
      s""""Callsite":"count","Parent IDs":[],"Storage Level":{"Use Disk":false,""" +
      s""""Use Memory":false,"Deserialized":false,"Replication":1},"Barrier":false,""" +
      s""""Number of Partitions":1,"Number of Cached Partitions":0,"Memory Size":0,""" +
      s""""Disk Size":0}],"Parent IDs":[],"Details":"","Submission Time":110000,""" +
      s""""Accumulables":[],"Resource Profile Id":0},"Properties":{}}"""
  }

  test("qualification provider exposes cache nodes observed only in an AQE plan") {
    withMinimalEventLog { eventLog =>
      val app = createAppFromEventlog(eventLog)
      app.sparkProperties ++= sparkProperties
      addPlanVersions(app)
      val provider = new QualAppSummaryInfoProvider(
        app, None, emptyAggMetrics, Seq.empty)

      provider.hasSqlCacheEvidence shouldBe true
      cacheSerializerRecommendation(provider, QualificationAutoTunerHelper) shouldBe
        Some(configuredCacheSerializer)
    }
  }

  test("profiling provider uses observed plans without changing the PRE-AQE summary") {
    withMinimalEventLog { eventLog =>
      val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
      val eventLogInfo = EventLogPathProcessor.getEventLogInfo(eventLog, hadoopConf).head._1
      val app = new ApplicationInfo(hadoopConf, eventLogInfo)
      addPlanVersions(app)
      val primaryPlan = SparkPlanInfoTruncated("Project", "Project", Seq.empty)
      val summary = emptySummary(Seq(SQLPlanInfoProfileResult(1L, primaryPlan)))
      val provider = new SingleAppSummaryInfoProvider(app, summary)

      summary.sqlPlanInfo.map(_.sparkPlanInfo.nodeName) shouldBe Seq("Project")
      provider.hasSqlCacheEvidence shouldBe true
      cacheSerializerRecommendation(provider, ProfilingAutoTunerHelper) shouldBe
        Some(configuredCacheSerializer)
    }
  }

  test("providers expose cache nodes observed only in stage RDD scopes") {
    withMinimalEventLog(Seq(stageSubmittedWithScope("TableCacheQueryStage"))) { eventLog =>
      val qualApp = createAppFromEventlog(eventLog)
      qualApp.sparkProperties ++= sparkProperties
      val qualProvider = new QualAppSummaryInfoProvider(
        qualApp, None, emptyAggMetrics, Seq.empty)

      val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
      val eventLogInfo = EventLogPathProcessor.getEventLogInfo(eventLog, hadoopConf).head._1
      val profApp = new ApplicationInfo(hadoopConf, eventLogInfo)
      val profProvider = new SingleAppSummaryInfoProvider(profApp, emptySummary())

      qualProvider.hasSqlCacheEvidence shouldBe true
      profProvider.hasSqlCacheEvidence shouldBe true
      cacheSerializerRecommendation(provider = qualProvider, QualificationAutoTunerHelper) shouldBe
        Some(configuredCacheSerializer)
      cacheSerializerRecommendation(provider = profProvider, ProfilingAutoTunerHelper) shouldBe
        Some(configuredCacheSerializer)
    }
  }

  test("providers expose no cache evidence when no cache node was observed") {
    withMinimalEventLog(Seq(stageSubmittedWithScope("mapPartitions"))) { eventLog =>
      val qualApp = createAppFromEventlog(eventLog)
      qualApp.sparkProperties ++= sparkProperties
      qualApp.sqlManager.addNewExecution(1L, plan("Project", plan("Filter")), "primary")
      val qualProvider = new QualAppSummaryInfoProvider(
        qualApp, None, emptyAggMetrics, Seq.empty)

      val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
      val eventLogInfo = EventLogPathProcessor.getEventLogInfo(eventLog, hadoopConf).head._1
      val profApp = new ApplicationInfo(hadoopConf, eventLogInfo)
      profApp.sqlManager.addNewExecution(1L, plan("Project", plan("Filter")), "primary")
      val profProvider = new SingleAppSummaryInfoProvider(profApp, emptySummary())

      qualProvider.hasSqlCacheEvidence shouldBe false
      profProvider.hasSqlCacheEvidence shouldBe false
      cacheSerializerRecommendation(qualProvider, QualificationAutoTunerHelper) shouldBe None
      cacheSerializerRecommendation(profProvider, ProfilingAutoTunerHelper) shouldBe None
    }
  }

  test("cache serializer exclusion is honored when cache nodes were observed") {
    withMinimalEventLog { eventLog =>
      val app = createAppFromEventlog(eventLog)
      app.sparkProperties ++= sparkProperties
      addPlanVersions(app)
      val provider = new QualAppSummaryInfoProvider(
        app, None, emptyAggMetrics, Seq.empty)

      cacheSerializerRecommendation(
        provider,
        QualificationAutoTunerHelper,
        excludedProperties = List(cacheSerializerProperty)) shouldBe None
    }
  }

  test("providers recommend cache serializer from a Spark-generated named cache plan") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "named-cache") { spark =>
        val source = spark.range(100).selectExpr("id AS customer_id", "id * 10 AS total")
        source.createOrReplaceTempView("hot_orders")
        spark.catalog.cacheTable("hot_orders")
        spark.table("hot_orders").count()
        val cachedQuery = spark.sql("SELECT customer_id FROM hot_orders WHERE total > 100")
        cachedQuery.count()
        cachedQuery
      }
      val qualApp = createAppFromEventlog(eventLog)
      qualApp.sparkProperties ++= sparkProperties
      def collectNodeNames(
          plan: org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanInfo): Seq[String] = {
        plan.nodeName +: plan.children.flatMap(collectNodeNames)
      }
      val nodeNames = qualApp.sqlManager.getPlanInfos.values.toSeq.flatMap(collectNodeNames)
      withClue(s"Observed Spark plan nodes: ${nodeNames.mkString(", ")}") {
        nodeNames should contain("Scan In-memory table hot_orders")
      }
      val qualProvider = new QualAppSummaryInfoProvider(
        qualApp, None, emptyAggMetrics, Seq.empty)

      val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
      val eventLogInfo = EventLogPathProcessor.getEventLogInfo(eventLog, hadoopConf).head._1
      val profApp = new ApplicationInfo(hadoopConf, eventLogInfo)
      val profProvider = new SingleAppSummaryInfoProvider(profApp, emptySummary())

      qualProvider.hasSqlCacheEvidence shouldBe true
      profProvider.hasSqlCacheEvidence shouldBe true
      cacheSerializerRecommendation(qualProvider, QualificationAutoTunerHelper) shouldBe
        Some(configuredCacheSerializer)
      cacheSerializerRecommendation(profProvider, ProfilingAutoTunerHelper) shouldBe
        Some(configuredCacheSerializer)
    }
  }
}
