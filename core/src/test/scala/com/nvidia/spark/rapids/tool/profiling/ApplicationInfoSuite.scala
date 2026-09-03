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
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths, StandardOpenOption}

import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.tool.{EventLogPathProcessor, PlatformNames, StatusReportCounts, ToolTestUtils}
import com.nvidia.spark.rapids.tool.views.RawMetricProfilerView
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.IOUtils
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.prop.TableDrivenPropertyChecks._

import org.apache.spark.internal.Logging
import org.apache.spark.resource.ResourceProfile
import org.apache.spark.sql.{SparkSession, TrampolineUtil}
import org.apache.spark.sql.rapids.tool.UnsupportedSparkRuntimeException
import org.apache.spark.sql.rapids.tool.profiling._
import org.apache.spark.sql.rapids.tool.util.{FSUtils, SparkRuntime}

class ApplicationInfoSuite extends AnyFunSuite with Logging {

  lazy val sparkSession: SparkSession = {
    SparkSession
        .builder()
        .master("local[*]")
        .appName("Rapids Spark Profiling Tool Unit Tests")
        .getOrCreate()
  }

  lazy val hadoopConf: Configuration = sparkSession.sparkContext.hadoopConfiguration

  private val expRoot = ToolTestUtils.getTestResourceFile("ProfilingExpectations")
  private val logDir = ToolTestUtils.getTestResourcePath("spark-events-profiling")
  private val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")

  test("test single event") {
    testSqlCompression()
  }

  test("zstd: test sqlMetrics duration and execute cpu time") {
    testSqlCompression(Option("zstd"))
  }

  test("snappy: test sqlMetrics duration and execute cpu time") {
    testSqlCompression(Option("snappy"))
  }

  test("lzf: test sqlMetrics duration and execute cpu time") {
    testSqlCompression(Option("lz4"))
  }

  test("lz4: test sqlMetrics duration and execute cpu time") {
    testSqlCompression(Option("lzf"))
  }

  private def testSqlCompression(compressionNameOpt: Option[String] = None): Unit = {
    val rawLog = s"$logDir/eventlog_minimal_events"
    compressionNameOpt.foreach { compressionName =>
      val codec = TrampolineUtil.createCodec(sparkSession.sparkContext.getConf,
        compressionName)
      TrampolineUtil.withTempDir { tempDir =>
        val compressionFileName = new File(tempDir,
          "eventlog_minimal_events." + compressionName)
        val inputStream = Files.newInputStream(Paths.get(rawLog))
        val outputStream = codec.compressedOutputStream(
          Files.newOutputStream(compressionFileName.toPath, StandardOpenOption.CREATE))
        // copy and close streams
        IOUtils.copyBytes(inputStream, outputStream, 4096, true)
        testSingleEventFile(Array(tempDir.toString))
        compressionFileName.delete()
      }
    }
    if (compressionNameOpt.isEmpty) {
      testSingleEventFile(Array(rawLog))
    }
  }

  private def testSingleEventFile(logs: Array[String]): Unit = {
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    assert(apps.size == 1)
    val firstApp = apps.head
    assert(firstApp.sparkVersion.equals("3.1.1"))
    assert(firstApp.gpuMode)
    assert(firstApp.jobIdToInfo.keys.toSeq.contains(1))
    val stageInfo = firstApp.stageManager.getStage(0, 0)
    assert(stageInfo.isDefined && stageInfo.get.stageInfo.numTasks == 1)
    assert(firstApp.stageManager.getStage(2, 0).isDefined)
    assert(firstApp.taskManager.getTasks(stageID = 1, stageAttemptID = 0)
      .head.successful)
    assert(firstApp.taskManager.getTasks(stageID = 1, stageAttemptID = 0)
      .head.endReason.equals("Success"))
    val execInfo = firstApp.executorIdToInfo.get(firstApp.executorIdToInfo.keys.head)
    assert(execInfo.isDefined && execInfo.get.totalCores.equals(8))
    val rp = firstApp.resourceProfIdToInfo.get(firstApp.resourceProfIdToInfo.keys.head)
    assert(rp.isDefined)
    val memory = rp.get.executorResources(ResourceProfile.MEMORY)
    assert(memory.amount.equals(1024L))
    // assert attemptID
    assert(firstApp.attemptId == 1)
  }

  test("test rapids jar") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/rapids_join_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)
    assert(apps.head.sparkVersion.equals("3.0.1"))
    assert(apps.head.gpuMode)
    val rapidsJar =
      apps.head.classpathEntries.filterKeys(_ matches ".*rapids-4-spark_2.12-0.5.0.jar.*")
    val cuDAJar = apps.head.classpathEntries.filterKeys(_ matches ".*cudf-0.19.2-cuda11.jar.*")
    assert(rapidsJar.size == 1, "Rapids jar check")
    assert(cuDAJar.size == 1, "CUDA jar check")

    val collect = new CollectInformation(apps.toSeq)
    val rapidsJarResults = collect.getRapidsJARInfo
    assert(rapidsJarResults.size === 2)
    assert(rapidsJarResults.count(_.jar.contains("rapids-4-spark_2.12-0.5.0.jar")) === 1)
    assert(rapidsJarResults.count(_.jar.contains("cudf-0.19.2-cuda11.jar")) === 1)

    assert(apps.head.getEventLogPath == s"file:$logDir/rapids_join_eventlog.zstd")

    // assert attemptID
    assert(apps.head.attemptId == 1)
  }

  test("test sql and resourceprofile eventlog") {
    val eventLog = s"$logDir/rp_sql_eventlog.zstd"
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--output-directory",
        tempDir.getAbsolutePath,
        eventLog))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
    }
  }

  test("test spark2 eventlog") {
    val eventLog = Array(s"$logDir/spark2-eventlog.zstd")
    val apps = ToolTestUtils.processProfileApps(eventLog, sparkSession)
    assert(apps.size == 1)
    assert(apps.head.sparkVersion.equals("2.2.3"))
    assert(!apps.head.gpuMode)
    assert(apps.head.jobIdToInfo.keys.toSeq.size == 6)
    assert(apps.head.jobIdToInfo.keys.toSeq.contains(0))
    val stage0 = apps.head.stageManager.getStage(0, 0)
    assert(stage0.isDefined)
    assert(stage0.get.stageInfo.numTasks.equals(1))
  }

  test("test no sql eventlog") {
    val eventLog = s"$logDir/rp_nosql_eventlog"
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--output-directory",
        tempDir.getAbsolutePath,
        eventLog))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
    }
  }

  test("test printSQLPlanMetrics") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/rapids_join_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)

    val collect = new CollectInformation(apps.toSeq)
    val sqlMetrics = collect.getSQLPlanMetrics
    val resultExpectation =
      new File(expRoot, "rapids_join_eventlog_sqlmetrics_expectation.csv")
    assert(sqlMetrics.size == 83)

    import org.apache.spark.sql.functions._
    import sparkSession.implicits._
    val df = sqlMetrics.toDF.withColumn("stageIds", concat_ws(",", col("stageIds")))
    val dfExpect = ToolTestUtils.readExpectationCSV(sparkSession, resultExpectation.getPath)
    ToolTestUtils.compareDataFrames(df, dfExpect)
  }

  test("test GpuMetrics in eventlog") {
    TrampolineUtil.withTempDir { outputDir =>
      TrampolineUtil.withTempDir { tmpEventLogDir =>
        val eventLogFilePath = Paths.get(tmpEventLogDir.getAbsolutePath, "gpu_metrics_eventlog")
        // scalastyle:off line.size.limit
        val eventLogContent =
          """{"Event":"SparkListenerLogStart","Spark Version":"3.2.1"}
            |{"Event":"SparkListenerApplicationStart","App Name":"GPUMetrics", "App ID":"local-16261043003", "Timestamp":123456, "User":"User1"}
            |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":5073,"Index":5054,"Attempt":0,"Partition ID":5054,"Launch Time":1712248533994,"Executor ID":"100","Host":"10.154.65.143","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253284920,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.492","Value":"03:13:31.359","Internal":false,"Count Failed Values":true},{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.845","Value":"00:29:39.521","Internal":false,"Count Failed Values":true},{"ID":1016,"Name":"gpuSplitAndRetryCount","Update":"1","Value":"2","Internal":false,"Count Failed Values":true}]}}
            |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":2913,"Index":2894,"Attempt":0,"Partition ID":2894,"Launch Time":1712248532696,"Executor ID":"24","Host":"10.154.65.135","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253285639,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.758","Value":"03:13:32.117","Internal":false,"Count Failed Values":true},{"ID":1015,"Name":"gpuReadSpillFromDiskTime","Update":"00:00:02.599","Value":"00:33:37.153","Internal":false,"Count Failed Values":true},{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.845","Value":"00:29:39.521","Internal":false,"Count Failed Values":true}]}}
            |{"Event":"SparkListenerTaskEnd","Stage ID":11,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":2045,"Index":2026,"Attempt":0,"Partition ID":2026,"Launch Time":1712248530708,"Executor ID":"84","Host":"10.154.65.233","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253285667,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.003","Value":"03:13:32.120","Internal":false,"Count Failed Values":true},{"ID":1015,"Name":"gpuReadSpillFromDiskTime","Update":"00:00:00.955","Value":"00:33:38.108","Internal":false,"Count Failed Values":true}]}}""".stripMargin
        // scalastyle:on line.size.limit
        Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

        val profileArgs = Array(
          "--output-directory", outputDir.getAbsolutePath,
          eventLogFilePath.toString
        )

        val appArgs = new ProfileArgs(profileArgs)
        val (exit, _) = ProfileMain.mainInternal(appArgs)
        assert(exit == 0)

        val apps = ArrayBuffer[ApplicationInfo]()
        var index = 1

        val eventLogPaths = appArgs.eventlog()
        eventLogPaths.foreach { path =>
          val eventLogInfo = EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1
          apps += new ApplicationInfo(hadoopConf, eventLogInfo)
          index += 1
        }
        assert(apps.size == 1)

        val collect = new CollectInformation(apps.toSeq)
        val stageLevelResults = collect.getStageLevelMetrics

        // Sample eventlog has 4 gpu metrics - gpuSemaphoreWait, gpuReadSpillFromDiskTime
        // gpuSpillToHostTime, gpuSplitAndRetryCount. But gpuSemaphoreWait and
        // gpuReadSpillFromDiskTime metrics are updated in 2 stages(stage 10 and 11).
        // So the result will have 6 rows in total since we are reporting stage level metrics.
        assert(stageLevelResults.size == 6)
        // gpu metrics
        val gpuSemaphoreWait = stageLevelResults.find(_.accMetaRef.getName() == "gpuSemaphoreWait")
        assert(gpuSemaphoreWait.isDefined)
      }
    }
  }

  test("test printSQLPlans") {
    TrampolineUtil.withTempDir { tempOutputDir =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$logDir/rapids_join_eventlog.zstd"))
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path,
            sparkSession.sparkContext.hadoopConfiguration).head._1)
      }
      assert(apps.size == 1)
      CollectInformation.printSQLPlans(apps.toSeq, tempOutputDir.getAbsolutePath)
      val outputDir = new File(tempOutputDir, apps.head.appId)
      val dotDirs = ToolTestUtils.listFilesMatching(outputDir, _.endsWith("planDescriptions.log"))
      assert(dotDirs.length === 1)
    }
  }

  test("test sql_plan_info_pre_aqe file generation Profiler") {
    TrampolineUtil.withTempDir { tempOutputDir =>
      val appArgs = new ProfileArgs(Array(
        "--csv",
        "--output-directory",
        tempOutputDir.getAbsolutePath,
        s"$logDir/rapids_join_eventlog.zstd"))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
      val outputDir = new File(tempOutputDir, s"${Profiler.SUBDIR}/local-1622814619968")
      val sqlPlanInfoFiles =
        ToolTestUtils.listFilesMatching(outputDir, { f =>
          f.endsWith("sql_plan_info_pre_aqe.json")
        })
      assert(sqlPlanInfoFiles.length === 1)
    }
  }

  test("test sql_plan_info_pre_aqe.json with SparkListenerSQLExecutionStart event") {
    TrampolineUtil.withTempDir { tempDir =>
      val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
      // scalastyle:off line.size.limit
      val eventLogContent =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.2.1"}
          |{"Event":"SparkListenerApplicationStart","App Name":"SQL_Plan_Info_Test", "App ID":"local-16261043003", "Timestamp":123456, "User":"User1"}
          |{"Event":"org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart","executionId":0,"description":"createOrReplaceTempView at NativeMethodAccessorImpl.java:0","details":"SQLID","physicalPlanDescription":"== Parsed Logical Plan ==\nCreateViewCommand `temp`, false, true, LocalTempView, true\n   +- Relation [temp1#0,temp2] ","sparkPlanInfo":{"nodeName":"Execute CreateViewCommand","simpleString":"Execute CreateViewCommand","children":[],"metadata":{},"metrics":[]},"time":1716319724551,"modifiedConfigs":{}}
          |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":5073,"Index":5054,"Attempt":0,"Partition ID":5054,"Launch Time":1712248533994,"Executor ID":"100","Host":"10.154.65.143","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253284920,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.492","Value":"03:13:31.359","Internal":false,"Count Failed Values":true},{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.845","Value":"00:29:39.521","Internal":false,"Count Failed Values":true},{"ID":1016,"Name":"gpuSplitAndRetryCount","Update":"1","Value":"2","Internal":false,"Count Failed Values":true}]}}}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

      val appArgs = new ProfileArgs(Array(
        "--csv",
        "--output-directory",
        tempDir.getAbsolutePath,
        eventLogFilePath.toString))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)

      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/local-16261043003")
      // assert that a json file was generated
      val sqlPlanInfoFiles = ToolTestUtils.listFilesMatching(tempSubDir, { f =>
        f.endsWith("sql_plan_info_pre_aqe.json")
      })
      assert(sqlPlanInfoFiles.length === 1)

      val actualFilePath = s"${tempSubDir.getAbsolutePath}/sql_plan_info_pre_aqe.json"
      val actualResult = FSUtils.readFileContentAsUTF8(actualFilePath)
      // scalastyle:off line.size.limit
      val expectedResult =
        s"""{"sqlID":0,"sparkPlanInfo":{"nodeName":"Execute CreateViewCommand","simpleString":"Execute CreateViewCommand","children":[]}}""".stripMargin
      // scalastyle:on line.size.limit
      // assert that the spark rapids build info json file is same as expected
      assert(actualResult == expectedResult)
    }
  }

  test("test read GPU datasourcev1") {
    TrampolineUtil.withTempDir { _ =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$logDir/eventlog-gpu-dsv1.zstd"))
      var index: Int = 1
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path,
            sparkSession.sparkContext.hadoopConfiguration).head._1)
        index += 1
      }
      assert(apps.size == 1)
      val collect = new CollectInformation(apps.toSeq)
      val dsRes = collect.getDataSourceInfo
      assert(dsRes.size == 5)
      val allFormats = dsRes.map { r =>
        r.format
      }.toSet
      // The cuDF plugin reports supported GPU scans as "gpu*"
      val expectedFormats = Set("Text", "gpuCSV", "gpuParquet", "gpuORC", "gpuJSON")
      assert(allFormats.equals(expectedFormats))
      val allSchema = dsRes.map { r =>
        r.schema
      }.toSet
      assert(allSchema.forall(_.nonEmpty))
      val schemaParquet = dsRes.filter { r =>
        r.sqlID == 4
      }
      assert(schemaParquet.size == 1)
      val schemaText = dsRes.filter { r =>
        r.sqlID == 0
      }
      assert(schemaText.size == 1)
      val parquetRow = schemaParquet.head
      assert(parquetRow.schema.contains("loan_id"))
    }
  }

  test("test read GPU datasourcev2") {
    TrampolineUtil.withTempDir { _ =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$logDir/eventlog-gpu-dsv2.zstd"))
      var index: Int = 1
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
        index += 1
      }
      assert(apps.size == 1)
      val collect = new CollectInformation(apps.toSeq)
      val dsRes = collect.getDataSourceInfo
      assert(dsRes.size == 5)
      val allFormats = dsRes.map { r =>
        r.format
      }.toSet
      // The cuDF plugin reports supported GPU scans as "gpu*"
      val expectedFormats =
        Set("Text", "gpucsv", "gpujson", "gpuparquet", "gpuorc")
      assert(allFormats.equals(expectedFormats))
      val allSchema = dsRes.map { r =>
        r.schema
      }.toSet
      assert(allSchema.forall(_.nonEmpty))
      val schemaParquet = dsRes.filter { r =>
        r.sqlID == 3
      }
      assert(schemaParquet.size == 1)
      val parquetRow = schemaParquet.head
      assert(parquetRow.schema.contains("loan_id"))
    }
  }

  test("test read datasourcev1") {
    TrampolineUtil.withTempDir { _ =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$logDir/eventlog_dsv1.zstd"))
      var index: Int = 1
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
        index += 1
      }
      assert(apps.size == 1)
      val collect = new CollectInformation(apps.toSeq)
      val dsRes = collect.getDataSourceInfo
      assert(dsRes.size == 7)
      val allFormats = dsRes.map { r =>
        r.format
      }.toSet
      val expectedFormats = Set("Text", "CSV", "Parquet", "ORC", "JSON")
      assert(allFormats.equals(expectedFormats))
      val allSchema = dsRes.map { r =>
        r.schema
      }.toSet
      assert(allSchema.forall(_.nonEmpty))
      val schemaParquet = dsRes.filter { r =>
        r.sqlID == 2
      }
      assert(schemaParquet.size == 1)
      val parquetRow = schemaParquet.head
      assert(parquetRow.schema.contains("loan400"))
      assert(parquetRow.location.contains("lotscolumnsout"))
    }
  }

  test("test read datasourcev2") {
    TrampolineUtil.withTempDir { _ =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$logDir/eventlog_dsv2.zstd"))
      var index: Int = 1
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
        index += 1
      }

      assert(apps.size == 1)
      val collect = new CollectInformation(apps.toSeq)
      val dsRes = collect.getDataSourceInfo
      assert(dsRes.size == 9)
      val allFormats = dsRes.map { r =>
        r.format
      }.toSet
      val expectedFormats = Set("Text", "csv", "parquet", "orc", "json")
      assert(allFormats.equals(expectedFormats))
      val allSchema = dsRes.map { r =>
        r.schema
      }.toSet
      assert(allSchema.forall(_.nonEmpty))
      val schemaParquet = dsRes.filter { r =>
        r.sqlID == 2
      }
      assert(schemaParquet.size == 1)
      val parquetRow = schemaParquet.head
      // schema is truncated in v2
      assert(!parquetRow.schema.contains("loan400"))
      assert(parquetRow.schema.contains("..."))
      assert(parquetRow.location.contains("lotscolumnsout"))
    }
  }

  test("test IOMetrics") {
    TrampolineUtil.withTempDir { _ =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$logDir/eventlog-gpu-dsv1.zstd"))
      var index: Int = 1
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
        index += 1
      }
      assert(apps.size == 1)
      val aggResults = RawMetricProfilerView.getAggMetrics(apps.toSeq)
      val ioMetrics = aggResults.ioAggs
      assert(ioMetrics.size == 5)
      val metricsSqlId1 = ioMetrics.filter(metrics => metrics.sqlId == 1)
      assert(metricsSqlId1.size == 1)
      assert(metricsSqlId1.head.inputBytesReadSum == 1348)
      assert(metricsSqlId1.head.inputRecordsReadSum == 66)
      assert(metricsSqlId1.head.outputBytesWrittenSum == 0)
      assert(metricsSqlId1.head.outputRecordsWrittenSum == 0)
      assert(metricsSqlId1.head.diskBytesSpilledSum == 0)
      assert(metricsSqlId1.head.memoryBytesSpilledSum == 0)
      assert(metricsSqlId1.head.srTotalBytesReadSum == 0)
      assert(metricsSqlId1.head.swTotalBytesWriteSum == 0)
    }
  }

  test("test jdbc read") {
    TrampolineUtil.withTempDir { _ =>
      val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
      val appArgs = new ProfileArgs(Array(s"$qualLogDir/jdbc_eventlog.zstd"))
      var index: Int = 1
      val eventlogPaths = appArgs.eventlog()
      for (path <- eventlogPaths) {
        apps += new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
        index += 1
      }
      assert(apps.size == 1)
      val collect = new CollectInformation(apps.toSeq)
      val dsRes = collect.getDataSourceInfo
      val format = dsRes.map(r => r.format).toSet.mkString
      val expectedFormat = "JDBC"
      val location = dsRes.map(r => r.location).toSet.mkString
      val expectedLocation = "TBLS"
      assert(format.equals(expectedFormat))
      assert(location.equals(expectedLocation))
      dsRes.foreach { r =>
        assert(r.schema.contains("bigint"))
      }
    }
  }

  test("test printJobInfo") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/rp_sql_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)
    val collect = new CollectInformation(apps.toSeq)
    val jobInfo = collect.getJobInfo

    assert(jobInfo.size == 2)
    val firstRow = jobInfo.head
    assert(firstRow.jobID === 0)
    assert(firstRow.stageIds.size === 1)
    assert(firstRow.sqlID === None)
    assert(firstRow.startTime === 1622846402778L)
    assert(firstRow.endTime === Some(1622846410240L))

    val secondRow = jobInfo(1)
    assert(secondRow.jobID === 1)
    assert(secondRow.stageIds.size === 4)
    assert(secondRow.sqlID.isDefined && secondRow.sqlID.get === 0)
    assert(secondRow.startTime === 1622846431114L)
    assert(secondRow.endTime === Some(1622846441591L))
  }

  test("test sqlToStages") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/rp_sql_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)
    val collect = new CollectInformation(apps.toSeq)
    val sqlToStageInfo = collect.getSQLToStage

    assert(sqlToStageInfo.size == 4)
    val firstRow = sqlToStageInfo.head
    assert(firstRow.stageId === 1)
    assert(firstRow.sqlID === 0)
    assert(firstRow.jobID === 1)
    assert(firstRow.duration === Some(8174))
    assert(firstRow.nodeNames.mkString(",") === "Exchange(9),WholeStageCodegen (1)(10),Scan(13)")
  }

  test("test wholeStage mapping") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/rp_sql_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)
    val collect = new CollectInformation(apps.toSeq)
    val wholeStageMapping = collect.getWholeStageCodeGenMapping

    assert(wholeStageMapping.size == 10)
    val firstRow = wholeStageMapping.head
    assert(firstRow.nodeID === 0)
    assert(firstRow.sqlID === 0)
    assert(firstRow.parent === "WholeStageCodegen (6)")
    assert(firstRow.child === "HashAggregate")
    assert(firstRow.childNodeID === 1)
    val lastRow = wholeStageMapping(9)
    assert(lastRow.nodeID === 17)
    assert(lastRow.sqlID === 0)
    assert(lastRow.parent === "WholeStageCodegen (3)")
    assert(lastRow.child === "SerializeFromObject")
    assert(lastRow.childNodeID === 19)
  }


  test("test multiple resource profile in single app") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs = new ProfileArgs(Array(s"$logDir/rp_nosql_eventlog"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)
    assert(apps.head.resourceProfIdToInfo.size == 2)

    val row0 = apps.head.resourceProfIdToInfo(0)
    assert(row0.resourceProfileId.equals(0))
    val execMem0 = row0.executorResources.get(ResourceProfile.MEMORY)
    val execCores0 = row0.executorResources.get(ResourceProfile.CORES)
    assert(execMem0.isDefined && execMem0.get.amount === 20480L)
    assert(execCores0.isDefined && execCores0.get.amount === 4)

    val row1 = apps.head.resourceProfIdToInfo(1)
    assert(row1.resourceProfileId.equals(1))
    val execMem1 = row1.executorResources.get(ResourceProfile.MEMORY)
    val execCores1 = row1.executorResources.get(ResourceProfile.CORES)
    assert(execMem1.isDefined && execMem1.get.amount === 6144L)
    assert(execCores1.isDefined && execCores1.get.amount === 2)
  }

  test("test spark2 and spark3 event logs") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs = new ProfileArgs(Array(s"$logDir/tasks_executors_fail_compressed_eventlog.zstd",
      s"$logDir/spark2-eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 2)
    val collect = new CollectInformation(apps.toSeq)
    val execInfos = collect.getExecutorInfo
    // just the fact it worked makes sure we can run with both files
    // since we give them indexes above they should be in the right order
    // and spark2 event info should be second
    val firstRow = execInfos.head
    assert(firstRow.resourceProfileId === 0)

    val secondRow = execInfos(1)
    assert(secondRow.resourceProfileId === 0)
  }

  test("test filename match") {
    val matchFileName = "udf"
    val appArgs = new ProfileArgs(Array(
      "--match-event-logs",
      matchFileName,
      s"$qualLogDir/udf_func_eventlog",
      s"$qualLogDir/udf_dataset_eventlog",
      s"$qualLogDir/dataset_eventlog"
    ))

    val (result, _) = EventLogPathProcessor.processAllPaths(appArgs.filterCriteria.toOption,
      appArgs.matchEventLogs.toOption, appArgs.eventlog(), hadoopConf)
    assert(result.length == 2)
  }

  test("test filter file newest") {
    val tempFile1 = File.createTempFile("tempOutputFile1", "")
    val tempFile2 = File.createTempFile("tempOutputFile2", "")
    val tempFile3 = File.createTempFile("tempOutputFile3", "")
    val tempFile4 = File.createTempFile("tempOutputFile4", "")
    try {
      tempFile1.deleteOnExit()
      tempFile2.deleteOnExit()
      tempFile3.deleteOnExit()
      tempFile4.deleteOnExit()

      tempFile1.setLastModified(98765432) // newest file
      tempFile2.setLastModified(12324567) // oldest file
      tempFile3.setLastModified(34567891) // second newest file
      tempFile4.setLastModified(23456789)
      val filterNew = "2-newest-filesystem"
      val appArgs = new ProfileArgs(Array(
        "--filter-criteria",
        filterNew,
        tempFile1.toString,
        tempFile2.toString,
        tempFile3.toString
      ))

      val (result, _) = EventLogPathProcessor.processAllPaths(appArgs.filterCriteria.toOption,
        appArgs.matchEventLogs.toOption, appArgs.eventlog(), hadoopConf)
      assert(result.length == 2)
      // Validate 2 newest files
      assert(result.head.eventLog.getName.equals(tempFile1.getName))
      assert(result(1).eventLog.getName.equals(tempFile3.getName))
    } finally {
      tempFile1.delete()
      tempFile2.delete()
      tempFile3.delete()
      tempFile4.delete()
    }
  }

  test("test filter file oldest and file name match") {

    val tempFile1 = File.createTempFile("tempOutputFile1", "")
    val tempFile2 = File.createTempFile("tempOutputFile2", "")
    val tempFile3 = File.createTempFile("tempOutputFile3", "")
    val tempFile4 = File.createTempFile("tempOutputFile4", "")
    try {
      tempFile1.deleteOnExit()
      tempFile2.deleteOnExit()
      tempFile3.deleteOnExit()
      tempFile4.deleteOnExit()

      tempFile1.setLastModified(98765432) // newest file
      tempFile2.setLastModified(12324567) // oldest file
      tempFile3.setLastModified(34567891) // second newest file
      tempFile4.setLastModified(23456789)

      val filterOld = "3-oldest-filesystem"
      val matchFileName = "temp"
      val appArgs = new ProfileArgs(Array(
        "--filter-criteria",
        filterOld,
        "--match-event-logs",
        matchFileName,
        tempFile1.toString,
        tempFile2.toString,
        tempFile3.toString,
        tempFile4.toString
      ))

      val (result, _) = EventLogPathProcessor.processAllPaths(appArgs.filterCriteria.toOption,
        appArgs.matchEventLogs.toOption, appArgs.eventlog(), hadoopConf)
      assert(result.length == 3)
      // Validate 3 oldest files
      assert(result.head.eventLog.getName.equals(tempFile2.getName))
      assert(result(1).eventLog.getName.equals(tempFile4.getName))
      assert(result(2).eventLog.getName.equals(tempFile3.getName))
    } finally {
      tempFile1.delete()
      tempFile2.delete()
      tempFile3.delete()
      tempFile4.delete()
    }
  }

  test("test reading driver log") {
    val driverlog = s"$logDir/driverlog"
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--driverlog", driverlog,
        "--output-directory",
        tempDir.getAbsolutePath))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)
      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/driver")
      val dotDirs = ToolTestUtils.listFilesMatching(tempSubDir, { f =>
        f.endsWith(".csv")
      })
      assert(dotDirs.length === 1)
      for (file <- dotDirs) {
        assert(file.getAbsolutePath.endsWith(".csv"))
        val df = sparkSession.read.option("header", "true").csv(file.getAbsolutePath)
        val res = df.collect()
        assert(res.nonEmpty)
        val unsupportedHex = df.filter(df("operatorName") === "Hex").count()
        assert(unsupportedHex == 1)
        assert(res.length == 3)
      }
    }
  }

  test("test gds-ucx-parameters") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/gds_ucx_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)
    val collect = new CollectInformation(apps.toSeq)
    for (_ <- apps) {
      val rapidsProps = collect.getRapidsProperties
      val rows = rapidsProps.map(_.rows.head)
      assert(rows.length == 5) // 5 properties captured.
      // verify  ucx parameters are captured.
      assert(rows.contains("spark.executorEnv.UCX_RNDV_SCHEME"))

      // verify gds parameters are captured.
      assert(rows.contains("spark.rapids.memory.gpu.direct.storage.spill.alignedIO"))

      val sparkProps = collect.getSparkProperties
      val sparkPropsRows = sparkProps.map(_.rows.head)
      assert(sparkPropsRows.contains("spark.eventLog.dir"))
      assert(sparkPropsRows.contains("spark.plugins"))
    }
  }

  test("test executor info local mode") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/spark2-eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)

    val collect = new CollectInformation(apps.toSeq)
    val execInfo = collect.getExecutorInfo
    assert(execInfo.size == 1)
    assert(execInfo.head.numExecutors === 1)
    assert(execInfo.head.maxMem === 384093388L)
  }

  test("test executor info cluster mode") {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs =
      new ProfileArgs(Array(s"$logDir/tasks_executors_fail_compressed_eventlog.zstd"))
    var index: Int = 1
    val eventlogPaths = appArgs.eventlog()
    for (path <- eventlogPaths) {
      apps += new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
      index += 1
    }
    assert(apps.size == 1)

    val collect = new CollectInformation(apps.toSeq)
    val execInfo = collect.getExecutorInfo
    assert(execInfo.size == 1)
    assert(execInfo.head.numExecutors === 8)
    assert(execInfo.head.maxMem === 5538054144L)
  }

  test("test status reports for multiple eventlogs") {
    // Test with normal and malformed eventlogs and verify their status reports
    val bad_eventLog = s"$logDir/malformed_json_eventlog.zstd"
    val eventLog = s"$logDir/rapids_join_eventlog.zstd"
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--csv",
        "--output-directory",
        tempDir.getAbsolutePath,
        bad_eventLog,
        eventLog))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)

      // Status counts: 1 SUCCESS, 0 FAILURE, 0 SKIPPED, 1 UNKNOWN
      val expectedStatusCount = StatusReportCounts(1, 0, 0, 1)
      // Compare the expected status counts with the actual status counts from the application
      ToolTestUtils.compareStatusReport(sparkSession, expectedStatusCount,
        s"${tempDir.getAbsolutePath}/rapids_4_spark_profile/profiling_status.csv")
    }
  }

  test("test csv file output with and without diagnostic views") {
    // The test cases are defined in the table below.
    // The test cases are:
    // 1. Profiler run with failures without diagnostic views
    // 2. Profiler run with failures with diagnostic views
    // 3. Profiler run without failures without diagnostic views
    // 4. Profiler run without failures with diagnostic views
    val testCases = Table(
      ("testName", "eventLog", "appId", "expectedFileCount", "enableDiagnosticViews"),
      ("failures without diagnostic views",
        s"$logDir/tasks_executors_fail_compressed_eventlog.zstd",
        "application_1603128018386_7846",
        21,
        false),
      ("failures with diagnostic views",
        s"$logDir/tasks_executors_fail_compressed_eventlog.zstd",
        "application_1603128018386_7846",
        23,
        true),
      ("gpu without diagnostic views",
        s"$qualLogDir/udf_dataset_eventlog",
        "local-1651188809790",
        17,
        false),
      ("gpu with diagnostic views",
        s"$qualLogDir/udf_dataset_eventlog",
        "local-1651188809790",
        19,
        true)
    )

    forAll(testCases) { (testName, eventLog, appId, expectedFileCount, enableDiagnosticViews) =>
      TrampolineUtil.withTempDir { tempDir =>
        val baseArgs = Array(
          "--csv",
          "--output-directory",
          tempDir.getAbsolutePath
        )
        val args = if (enableDiagnosticViews) {
          baseArgs :+ "--enable-diagnostic-views" :+ eventLog
        } else {
          baseArgs :+ eventLog
        }
        val appArgs = new ProfileArgs(args)
        val (exit, _) = ProfileMain.mainInternal(appArgs)
        assert(exit == 0)
        val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/$appId")

        // assert that files were generated
        val dotDirs = ToolTestUtils.listFilesMatching(tempSubDir, { f =>
          f.endsWith(".csv")
        })
        // compare the number of files generated
        assert(dotDirs.length === expectedFileCount, s"Test case '$testName' failed: " +
            s"expected $expectedFileCount files but got ${dotDirs.length}")
        for (file <- dotDirs) {
          assert(file.getAbsolutePath.endsWith(".csv"))
          // just load each one to make sure formatted properly
          val df = sparkSession.read.option("header", "true").csv(file.getAbsolutePath)
          val res = df.collect()
          assert(res.nonEmpty)
        }

        // Status counts: 1 SUCCESS, 0 FAILURE, 0 SKIPPED, 0 UNKNOWN
        val expectedStatusCount = StatusReportCounts(1, 0, 0, 0)
        // Compare the expected status counts with the actual status counts from the application
        ToolTestUtils.compareStatusReport(sparkSession, expectedStatusCount,
          s"${tempDir.getAbsolutePath}/rapids_4_spark_profile/profiling_status.csv")
      }
    }
  }

  test("test collectionAccumulator") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "collectaccum") { spark =>
        val a = spark.sparkContext.collectionAccumulator[Long]("testCollect")
        val rdd = spark.sparkContext.parallelize(Array(1, 2, 3, 4))
        // run something to add it to collectionAccumulator
        rdd.foreach(x => a.add(x))
        import spark.implicits._
        rdd.toDF
      }

      TrampolineUtil.withTempDir { collectFileDir =>
        val appArgs = new ProfileArgs(Array(
          "--output-directory",
          collectFileDir.getAbsolutePath,
          eventLog))

        val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
        var index: Int = 1
        val eventlogPaths = appArgs.eventlog()
        for (path <- eventlogPaths) {
          apps += new ApplicationInfo(hadoopConf,
            EventLogPathProcessor.getEventLogInfo(path, hadoopConf).head._1)
          index += 1
        }
        // the fact we generated app properly shows we can handle collectionAccumulators
        // right now we just ignore them so nothing else to check
        assert(apps.size == 1)
      }
    }
  }

  test("test gpu reused subquery") {
    val apps = ToolTestUtils.processProfileApps(Array(s"$logDir/nds_q66_gpu.zstd"), sparkSession)
    val collect = new CollectInformation(apps.toSeq)
    val sqlToStageInfo = collect.getSQLToStage
    val countScanParquet = sqlToStageInfo.flatMap(_.nodeNames).count(_.contains("GpuScan parquet"))
    // There are 12 `GpuScan parquet` raw nodes, but 4 are inside `ReusedExchange`, 1 is inside
    // `ReusedSubquery`, so we expect 7.
    assert(countScanParquet == 7)
  }

  test("test gpu event log with SparkRapidsBuildInfoEvent") {
    TrampolineUtil.withTempDir { tempDir =>
      val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
      // scalastyle:off line.size.limit
      val eventLogContent =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.2.1"}
          |{"Event":"com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent","sparkRapidsBuildInfo":{"url":"https://github.com/NVIDIA/spark-rapids.git","branch":"HEAD","revision":"f932a7802bbf31b6205358d1abd7c7b49c8bea3c","version":"24.06.0","date":"2024-06-13T19:48:28Z","cudf_version":"24.06.0","user":"root"},"sparkRapidsJniBuildInfo":{"url":"https://github.com/NVIDIA/spark-rapids-jni.git","branch":"HEAD","gpu_architectures":"70;75;80;86;90","revision":"e9c92a5339437ce0cd72bc384084bd7ee45b37f9","version":"24.06.0","date":"2024-06-08T01:21:57Z","user":"root"},"cudfBuildInfo":{"url":"https://github.com/rapidsai/cudf.git","branch":"HEAD","gpu_architectures":"70;75;80;86;90","revision":"7c706cc4004d5feaae92544b3b29a00c64f7ed86","version":"24.06.0","date":"2024-06-08T01:21:55Z","user":"root"},"sparkRapidsPrivateBuildInfo":{"url":"https://gitlab-master.nvidia.com/nvspark/spark-rapids-private.git","branch":"HEAD","revision":"755b4dd03c753cacb7d141f3b3c8ff9f83888b69","version":"24.06.0","date":"2024-06-08T11:44:03Z","user":"root"}}
          |{"Event":"SparkListenerApplicationStart","App Name":"GPUMetrics", "App ID":"local-16261043003", "Timestamp":123456, "User":"User1"}
          |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":5073,"Index":5054,"Attempt":0,"Partition ID":5054,"Launch Time":1712248533994,"Executor ID":"100","Host":"10.154.65.143","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253284920,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.492","Value":"03:13:31.359","Internal":false,"Count Failed Values":true},{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.845","Value":"00:29:39.521","Internal":false,"Count Failed Values":true},{"ID":1016,"Name":"gpuSplitAndRetryCount","Update":"1","Value":"2","Internal":false,"Count Failed Values":true}]}}
          |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":2913,"Index":2894,"Attempt":0,"Partition ID":2894,"Launch Time":1712248532696,"Executor ID":"24","Host":"10.154.65.135","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253285639,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.758","Value":"03:13:32.117","Internal":false,"Count Failed Values":true},{"ID":1015,"Name":"gpuReadSpillFromDiskTime","Update":"00:00:02.599","Value":"00:33:37.153","Internal":false,"Count Failed Values":true},{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.845","Value":"00:29:39.521","Internal":false,"Count Failed Values":true}]}}
          |{"Event":"SparkListenerTaskEnd","Stage ID":11,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":2045,"Index":2026,"Attempt":0,"Partition ID":2026,"Launch Time":1712248530708,"Executor ID":"84","Host":"10.154.65.233","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712253285667,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.003","Value":"03:13:32.120","Internal":false,"Count Failed Values":true},{"ID":1015,"Name":"gpuReadSpillFromDiskTime","Update":"00:00:00.955","Value":"00:33:38.108","Internal":false,"Count Failed Values":true}]}}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

      val appArgs = new ProfileArgs(Array(
        "--csv",
        "--output-directory",
        tempDir.getAbsolutePath,
        eventLogFilePath.toString))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)

      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/local-16261043003")
      // assert that a json file was generated
      val dotDirs = ToolTestUtils.listFilesMatching(tempSubDir, { f =>
        f.endsWith("build_info.json")
      })
      assert(dotDirs.length === 1)

      val actualFilePath = s"${tempSubDir.getAbsolutePath}/spark_rapids_build_info.json"
      val actualResult = FSUtils.readFileContentAsUTF8(actualFilePath)
      val expectedResult =
        s"""|[ {
            |  "sparkRapidsBuildInfo" : {
            |    "url" : "https://github.com/NVIDIA/spark-rapids.git",
            |    "branch" : "HEAD",
            |    "revision" : "f932a7802bbf31b6205358d1abd7c7b49c8bea3c",
            |    "version" : "24.06.0",
            |    "date" : "2024-06-13T19:48:28Z",
            |    "cudf_version" : "24.06.0",
            |    "user" : "root"
            |  },
            |  "sparkRapidsJniBuildInfo" : {
            |    "url" : "https://github.com/NVIDIA/spark-rapids-jni.git",
            |    "branch" : "HEAD",
            |    "gpu_architectures" : "70;75;80;86;90",
            |    "revision" : "e9c92a5339437ce0cd72bc384084bd7ee45b37f9",
            |    "version" : "24.06.0",
            |    "date" : "2024-06-08T01:21:57Z",
            |    "user" : "root"
            |  },
            |  "cudfBuildInfo" : {
            |    "url" : "https://github.com/rapidsai/cudf.git",
            |    "branch" : "HEAD",
            |    "gpu_architectures" : "70;75;80;86;90",
            |    "revision" : "7c706cc4004d5feaae92544b3b29a00c64f7ed86",
            |    "version" : "24.06.0",
            |    "date" : "2024-06-08T01:21:55Z",
            |    "user" : "root"
            |  },
            |  "sparkRapidsPrivateBuildInfo" : {
            |    "url" : "https://gitlab-master.nvidia.com/nvspark/spark-rapids-private.git",
            |    "branch" : "HEAD",
            |    "revision" : "755b4dd03c753cacb7d141f3b3c8ff9f83888b69",
            |    "version" : "24.06.0",
            |    "date" : "2024-06-08T11:44:03Z",
            |    "user" : "root"
            |  }
            |} ]""".stripMargin
      // Assert that the spark rapids build info json file is same as expected
      // Use proper JSON comparison that ignores field ordering differences
      assert(ToolTestUtils.compareJsonIgnoringFieldOrder(actualResult, expectedResult),
        s"JSON content differs:\nActual: $actualResult\nExpected: $expectedResult")
    }
  }

  // TODO: GPU event log run with spark rapids 24.06+ includes SparkRapidsBuildInfoEvent.
  // Once the test event logs are updated, we should remove this test.
  test("test gpu event log with no SparkRapidsBuildInfoEvent") {
    val eventLog = s"$logDir/nds_q66_gpu.zstd"
    TrampolineUtil.withTempDir { tempDir =>
      val appArgs = new ProfileArgs(Array(
        "--csv",
        "--output-directory",
        tempDir.getAbsolutePath,
        eventLog))
      val (exit, _) = ProfileMain.mainInternal(appArgs)
      assert(exit == 0)

      val tempSubDir = new File(tempDir, s"${Profiler.SUBDIR}/application_1701368813061_0008")
      // assert that a json file was generated
      val dotDirs = ToolTestUtils.listFilesMatching(tempSubDir, { f =>
        f.endsWith("build_info.json")
      })
      assert(dotDirs.length === 1)

      val actualFilePath = s"${tempSubDir.getAbsolutePath}/spark_rapids_build_info.json"
      val actualResult = FSUtils.readFileContentAsUTF8(actualFilePath)
      val expectedResult =
        s"""|[ {
            |  "sparkRapidsBuildInfo" : { },
            |  "sparkRapidsJniBuildInfo" : { },
            |  "cudfBuildInfo" : { },
            |  "sparkRapidsPrivateBuildInfo" : { }
            |} ]""".stripMargin
      // assert that the spark rapids build info json file is same as expected
      assert(actualResult == expectedResult)
    }
  }

  // scalastyle:off line.size.limit
  val supportedSparkRuntimeTestCases: Map[String, Seq[(String, SparkRuntime.SparkRuntime)]] = Map(
    // tests for standard Spark runtime
    s"$qualLogDir/nds_q86_test" -> Seq(
      (PlatformNames.DATABRICKS_AWS, SparkRuntime.SPARK),                    // Expected: SPARK on Databricks AWS
      (PlatformNames.ONPREM, SparkRuntime.SPARK)                             // Expected: SPARK on Onprem
    ),
    // tests for Spark Rapids runtime
    s"$logDir/nds_q66_gpu.zstd" -> Seq(
      (PlatformNames.DATABRICKS_AWS, SparkRuntime.SPARK_RAPIDS),             // Expected: SPARK_RAPIDS on Databricks AWS
      (PlatformNames.ONPREM, SparkRuntime.SPARK_RAPIDS)                      // Expected: SPARK_RAPIDS on Onprem
    ),
    // tests for Photon runtime with fallback to SPARK for unsupported platforms
    s"$qualLogDir/nds_q88_photon_db_13_3.zstd" -> Seq(
      (PlatformNames.DATABRICKS_AWS, SparkRuntime.PHOTON),                   // Expected: PHOTON on Databricks AWS
      (PlatformNames.DATABRICKS_AZURE, SparkRuntime.PHOTON)                  // Expected: PHOTON on Databricks Azure
    )
  )
  // scalastyle:on line.size.limit

  supportedSparkRuntimeTestCases.foreach { case (logPath, platformRuntimeCases) =>
    val baseFileName = logPath.split("/").last
    platformRuntimeCases.foreach { case (platform, expectedRuntime) =>
      test(s"test eventlog $baseFileName on $platform has supported runtime: $expectedRuntime") {
        val args = Array("--platform", platform, logPath)
        val apps = ToolTestUtils.processProfileApps(args, sparkSession)
        assert(apps.size == 1)
        assert(apps.head.getSparkRuntime == expectedRuntime)
      }
    }
  }

  // scalastyle:off line.size.limit
  val unsupportedSparkRuntimeTestCases: Map[String, Seq[String]] = Map(
    s"$qualLogDir/nds_q88_photon_db_13_3.zstd" -> Seq(
      PlatformNames.ONPREM,                                                // Expected: PHOTON runtime on Onprem is not supported
      PlatformNames.DATAPROC                                               // Expected: PHOTON runtime on Dataproc is not supported
    )
  )
  // scalastyle:on line.size.limit

  unsupportedSparkRuntimeTestCases.foreach { case (logPath, platformNames) =>
    val baseFileName = logPath.split("/").last
    platformNames.foreach { platform =>
      test(s"test eventlog $baseFileName on $platform has unsupported runtime") {
        val args = Array("--platform", platform, logPath)
        intercept[UnsupportedSparkRuntimeException] {
          ToolTestUtils.processProfileApps(args, sparkSession)
        }
      }
    }
  }

  test("test text file expansion with event log paths") {
    TrampolineUtil.withTempDir { tempDir =>
      // Create a text file with multiple event log paths
      val txtFile = new File(tempDir, "eventlogs.txt")
      val txtContent = s"""${qualLogDir}/udf_func_eventlog
                          |${qualLogDir}/udf_dataset_eventlog
                          |${qualLogDir}/dataset_eventlog""".stripMargin

      // Write the text file
      val writer = new java.io.FileWriter(txtFile)
      try {
        writer.write(txtContent)
      } finally {
        writer.close()
      }

      val (result, _) = EventLogPathProcessor.processAllPaths(
        None, None, List(txtFile.getAbsolutePath), hadoopConf)

      // Should find 3 event logs from the text file
      assert(result.length == 3)
      val eventLogNames = result.map(_.eventLog.getName).sorted
      assert(eventLogNames.contains("udf_func_eventlog"))
      assert(eventLogNames.contains("udf_dataset_eventlog"))
      assert(eventLogNames.contains("dataset_eventlog"))
    }
  }

  test("test mixed input: text file and direct event log paths") {
    TrampolineUtil.withTempDir { tempDir =>
      // Create a text file with some event log paths
      val txtFile = new File(tempDir, "partial_logs.txt")
      val txtContent = s"""${qualLogDir}/udf_func_eventlog
                          |${qualLogDir}/udf_dataset_eventlog""".stripMargin

      val writer = new java.io.FileWriter(txtFile)
      try {
        writer.write(txtContent)
      } finally {
        writer.close()
      }

      // Mix text file with direct event log path
      val inputPaths = List(
        txtFile.getAbsolutePath,
        s"${qualLogDir}/dataset_eventlog"
      )

      val (result, _) = EventLogPathProcessor.processAllPaths(
        None, None, inputPaths, hadoopConf)

      // Should find 3 event logs total (2 from txt + 1 direct)
      assert(result.length == 3)
      val eventLogNames = result.map(_.eventLog.getName).sorted
      assert(eventLogNames.contains("udf_func_eventlog"))
      assert(eventLogNames.contains("udf_dataset_eventlog"))
      assert(eventLogNames.contains("dataset_eventlog"))
    }
  }

  test("test text file with comma-separated paths") {
    TrampolineUtil.withTempDir { tempDir =>
      val txtFile = new File(tempDir, "comma_separated.txt")
      val txtContent = s"${qualLogDir}/udf_func_eventlog,${qualLogDir}/dataset_eventlog"

      val writer = new java.io.FileWriter(txtFile)
      try {
        writer.write(txtContent)
      } finally {
        writer.close()
      }

      val (result, _) = EventLogPathProcessor.processAllPaths(
        None, None, List(txtFile.getAbsolutePath), hadoopConf)

      // Should find 2 event logs from comma-separated line
      assert(result.length == 2)
      val eventLogNames = result.map(_.eventLog.getName).sorted
      assert(eventLogNames.contains("udf_func_eventlog"))
      assert(eventLogNames.contains("dataset_eventlog"))
    }
  }

  test("test empty text file") {
    TrampolineUtil.withTempDir { tempDir =>
      val txtFile = new File(tempDir, "empty.txt")
      txtFile.createNewFile() // Create empty file

      val (result, _) = EventLogPathProcessor.processAllPaths(
        None, None, List(txtFile.getAbsolutePath), hadoopConf)

      // Should return empty result for empty text file
      assert(result.isEmpty)
    }
  }

  test("test modifiedConfigs parsed from SQLExecutionStart and merged into sparkProperties") {
    TrampolineUtil.withTempDir { tempDir =>
      val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
      // scalastyle:off line.size.limit
      val eventLogContent =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}
          |{"Event":"SparkListenerApplicationStart","App Name":"ModifiedConfigs_Test","App ID":"local-modconftest","Timestamp":123456,"User":"testUser"}
          |{"Event":"SparkListenerEnvironmentUpdate","JVM Information":{},"Spark Properties":{"spark.sql.autoBroadcastJoinThreshold":"10485760","spark.app.name":"OriginalAppName","spark.master":"local[*]"},"Hadoop Properties":{},"System Properties":{"file.encoding":"UTF-8"},"Classpath Entries":{}}
          |{"Event":"org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart","executionId":0,"rootExecutionId":0,"description":"test query","details":"SELECT * FROM t1","physicalPlanDescription":"== Physical Plan ==\nRange (1)\n\n(1) Range\nOutput [1]: [id#0L]\nArguments: Range (0, 10, step=1, splits=Some(4))\n","sparkPlanInfo":{"nodeName":"Range","simpleString":"Range (0, 10, step=1, splits=Some(4))","children":[],"metadata":{},"metrics":[]},"time":123457,"modifiedConfigs":{"spark.sql.autoBroadcastJoinThreshold":"-1","spark.app.name":"saralihalli-test"}}
          |{"Event":"SparkListenerApplicationEnd","Timestamp":123460}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

      val app = new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(
          eventLogFilePath.toString, hadoopConf).head._1)

      val sqlInfo = app.sqlIdToInfo.get(0L)
      assert(sqlInfo.isDefined)
      // modifiedConfigs field was added in Spark 3.3 (SPARK-34735).
      // On Spark 3.2.x the field doesn't exist, so reflection returns empty and
      // the baseline sparkProperties remain unchanged.
      if (sqlInfo.get.modifiedConfigs.nonEmpty) {
        // Spark 3.3+: modifiedConfigs is parsed and merged
        assert(sqlInfo.get.modifiedConfigs("spark.sql.autoBroadcastJoinThreshold") == "-1")
        assert(sqlInfo.get.modifiedConfigs("spark.app.name") == "saralihalli-test")
        assert(app.sparkProperties.getOrElse(
          "spark.sql.autoBroadcastJoinThreshold", "") == "-1")
        assert(app.sparkProperties.getOrElse(
          "spark.app.name", "") == "saralihalli-test")
      } else {
        // Spark 3.2.x: modifiedConfigs not available, baseline unchanged
        assert(app.sparkProperties.getOrElse(
          "spark.sql.autoBroadcastJoinThreshold", "") == "10485760")
        assert(app.sparkProperties.getOrElse(
          "spark.app.name", "") == "OriginalAppName")
      }
      // Baseline properties that were NOT overridden are always present
      assert(app.sparkProperties.getOrElse(
        "spark.master", "") == "local[*]")
    }
  }

  test("test modifiedConfigs redacts sensitive properties") {
    TrampolineUtil.withTempDir { tempDir =>
      val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
      // scalastyle:off line.size.limit
      // modifiedConfigs includes a sensitive S3 secret key that should be redacted
      val eventLogContent =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}
          |{"Event":"SparkListenerApplicationStart","App Name":"Redaction_Test","App ID":"local-redacttest","Timestamp":123456,"User":"testUser"}
          |{"Event":"SparkListenerEnvironmentUpdate","JVM Information":{},"Spark Properties":{"spark.master":"local[*]"},"Hadoop Properties":{},"System Properties":{"file.encoding":"UTF-8"},"Classpath Entries":{}}
          |{"Event":"org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart","executionId":0,"rootExecutionId":0,"description":"test","details":"SELECT 1","physicalPlanDescription":"== Physical Plan ==\nRange (1)\n","sparkPlanInfo":{"nodeName":"Range","simpleString":"Range","children":[],"metadata":{},"metrics":[]},"time":123457,"modifiedConfigs":{"spark.hadoop.fs.s3a.secret.key":"my-secret-key","spark.app.name":"safe-value"}}
          |{"Event":"SparkListenerApplicationEnd","Timestamp":123460}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

      val app = new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(
          eventLogFilePath.toString, hadoopConf).head._1)

      val sqlInfo = app.sqlIdToInfo.get(0L)
      assert(sqlInfo.isDefined)
      // On Spark 3.3+, verify sensitive key is redacted in sparkProperties
      if (sqlInfo.get.modifiedConfigs.nonEmpty) {
        // The secret key should be redacted after merge
        assert(app.sparkProperties.get("spark.hadoop.fs.s3a.secret.key")
          .forall(_.contains("redacted")))
        // Non-sensitive key should pass through unchanged
        assert(app.sparkProperties.getOrElse("spark.app.name", "") == "safe-value")
      }
    }
  }

  test("test empty modifiedConfigs does not affect sparkProperties") {
    TrampolineUtil.withTempDir { tempDir =>
      val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
      // scalastyle:off line.size.limit
      val eventLogContent =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}
          |{"Event":"SparkListenerApplicationStart","App Name":"EmptyModConf_Test","App ID":"local-emptymodconf","Timestamp":123456,"User":"testUser"}
          |{"Event":"SparkListenerEnvironmentUpdate","JVM Information":{},"Spark Properties":{"spark.sql.autoBroadcastJoinThreshold":"10485760","spark.master":"local[*]"},"Hadoop Properties":{},"System Properties":{"file.encoding":"UTF-8"},"Classpath Entries":{}}
          |{"Event":"org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart","executionId":0,"rootExecutionId":0,"description":"test query","details":"SELECT 1","physicalPlanDescription":"== Physical Plan ==\nRange (1)\n","sparkPlanInfo":{"nodeName":"Range","simpleString":"Range","children":[],"metadata":{},"metrics":[]},"time":123457,"modifiedConfigs":{}}
          |{"Event":"SparkListenerApplicationEnd","Timestamp":123460}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

      val app = new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(
          eventLogFilePath.toString, hadoopConf).head._1)

      val sqlInfo = app.sqlIdToInfo.get(0L)
      assert(sqlInfo.isDefined)
      assert(sqlInfo.get.modifiedConfigs.isEmpty)

      // Verify baseline sparkProperties are unchanged
      assert(app.sparkProperties.getOrElse(
        "spark.sql.autoBroadcastJoinThreshold", "") == "10485760")
    }
  }

  test("application exit code is parsed when present") {
    Seq(0, 13).foreach { exitCode =>
      TrampolineUtil.withTempDir { tempDir =>
        val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
        // scalastyle:off line.size.limit
        val eventLogContent =
          s"""{"Event":"SparkListenerLogStart","Spark Version":"4.0.2"}
            |{"Event":"SparkListenerApplicationStart","App Name":"ExitCode_Test","App ID":"local-exit-code-test","Timestamp":123456,"User":"testUser"}
            |{"Event":"SparkListenerApplicationEnd","Timestamp":123460,"ExitCode":$exitCode}"""
            .stripMargin
        // scalastyle:on line.size.limit
        Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

        val app = new ApplicationInfo(hadoopConf,
          EventLogPathProcessor.getEventLogInfo(
            eventLogFilePath.toString, hadoopConf).head._1)

        assert(app.getAppExitCode.contains(exitCode))
      }
    }
  }

  test("application exit code is unknown when absent") {
    TrampolineUtil.withTempDir { tempDir =>
      val eventLogFilePath = Paths.get(tempDir.getAbsolutePath, "test_eventlog")
      // scalastyle:off line.size.limit
      val eventLogContent =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.5.7"}
          |{"Event":"SparkListenerApplicationStart","App Name":"Legacy_Test","App ID":"local-legacy-test","Timestamp":123456,"User":"testUser"}
          |{"Event":"SparkListenerApplicationEnd","Timestamp":123460}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(eventLogFilePath, eventLogContent.getBytes(StandardCharsets.UTF_8))

      val app = new ApplicationInfo(hadoopConf,
        EventLogPathProcessor.getEventLogInfo(
          eventLogFilePath.toString, hadoopConf).head._1)

      assert(app.getAppExitCode.isEmpty)
    }
  }
}
