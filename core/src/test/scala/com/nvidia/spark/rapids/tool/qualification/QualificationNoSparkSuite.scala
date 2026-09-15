/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.qualification

import java.nio.file.{Files, Paths}
import java.time.LocalDateTime

import com.nvidia.spark.rapids.BaseNoSparkSuite
import com.nvidia.spark.rapids.tool.{EventLogPathProcessor, PlatformNames, StatusReportCounts, ToolTestUtils}
import com.nvidia.spark.rapids.tool.qualification.checkers.{QToolOutFileCheckerImpl, QToolOutJsonFileCheckerImpl, QToolResultCoreChecker, QToolStatusChecker, QToolTestCtxtBuilder}
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.rapids.tool.{SourceClusterInfo, ToolUtils}
import org.apache.spark.sql.rapids.tool.util.UTF8Source


/**
 * Unit tests for the Qualification tool that do not require creating a Spark session to generate
 * an eventlog.
 */
class QualificationNoSparkSuite extends BaseNoSparkSuite {
  private val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
  private val profLogDir = ToolTestUtils.getTestResourcePath("spark-events-profiling")
  private val expRoot = ToolTestUtils.getTestResourcePath("QualificationExpectations")

  def qualEventLog(fileName: String): String = s"$qualLogDir/$fileName"
  def expectedQualLoc(dirName: String): String = s"$expRoot/$dirName"
  def profEventLog(fileName: String): String = s"$profLogDir/$fileName"

  /** Parse udf_report.json and return (udfs list, metrics option). */
  private def readUdfReport(jsonFile: java.io.File)
      : (Seq[Map[String, Any]], Option[Map[String, Any]]) = {
    implicit val formats: DefaultFormats.type = DefaultFormats
    val source = UTF8Source.fromFile(jsonFile)
    val content = try source.mkString finally source.close()
    val json = JsonMethods.parse(content)
    val udfs = (json \ "udfs").extract[Seq[Map[String, Any]]]
    val metrics = (json \ "metrics").extractOpt[Map[String, Any]]
    (udfs, metrics)
  }

  test("test order desc") {
    // Apps are sorted in descending order based on GPU opportunity and EndTime
    val logFiles = Array(
      qualEventLog("dataset_eventlog"),
      qualEventLog("dsAndDf_eventlog.zstd"),
      qualEventLog("udf_dataset_eventlog"),
      qualEventLog("udf_func_eventlog")
    )

    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withMultipleThreads(4) // use multiple threads to evaluate race conditions
      .withChecker(
        QToolStatusChecker("Check that the app statuses are valid")
          .withExpectedCounts(StatusReportCounts(4, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(4)
          .withSuccessCode()
          .withCheckBlock(
            "apps appear in expected order",
            qRes => {
              qRes.appSummaries.map(_.appId) should be(
                Array("local-1622043423018",
                  "local-1651187225439",
                  "local-1651188809790",
                  "local-1623281204390"))
            }))
      .build()
  }

  test("test datasource read format included") {
    // this is also a test for dsV1
    val logFiles = Array(profEventLog("eventlog_dsv1.zstd"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("There is 1 successful app")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("Unsupported read and write formats")
          .withExpectedSize(1)
          .withSuccessCode()
          .withCheckBlock(
            "verify the fields in the app summary",
            qRes => {
              val unsupReadFormats = qRes.appSummaries.flatMap(_.readFileFormatAndTypesNotSupported)
              unsupReadFormats should be (Array("Text[*]"))
              val unsupWriteFormats = qRes.appSummaries.flatMap(_.writeDataFormat)
              unsupWriteFormats should be (Array("json"))
            }
          ))
      .withChecker(
        QToolOutFileCheckerImpl("check the content of summary file")
          .withExpectedRows("only 1 row in the summary", 1)
          .withContentVisitor(
            "read and write unsupported data formats are correct",
            csvContainer => {
              val rowsHead = csvContainer.csvRows.head
              val unsupReadFormats =
                rowsHead("Unsupported Read File Formats and Types")
              unsupReadFormats should be ("Text[*]")
              val unsupWriteFormats = rowsHead("Unsupported Write Data Format")
              unsupWriteFormats should be ("JSON")
            }))
      .build()
  }

  test("skip gpu event logs") {
    val logFiles = Array(qualEventLog("gpu_eventlog"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("There is 1 skipped app")
          .withExpectedCounts(StatusReportCounts(0, 0, 1, 0)))
      .withChecker(
        QToolOutFileCheckerImpl("Content of status file")
          .withTableLabel("coreCSVStatus")
          .withExpectedRows("expect only 1 row", 1)
          .withContentVisitor("Columns are correct while AppID might not be set",
            csvContainer => {
              val rowsHead = csvContainer.csvRows.head
              val status = rowsHead("Status")
              status should be ("SKIPPED")
              val appId = rowsHead("App ID")
              appId should be ("N/A")
              val description = rowsHead("Description")
              description should include (
                "GpuEventLogException: Cannot parse event logs from GPU run: skipping this file")
            }))
      .build()
  }

  test("malformed json eventlog fails without failing at runtime") {
    val logFiles = Array(
      qualEventLog("nds_q86_test"),
      profEventLog("malformed_json_eventlog.zstd"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("There is 1 successful app and 1 unknown state")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 1)))
      .withChecker(
        QToolOutFileCheckerImpl("check the content of summary file")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc("nds_q86_test")))
      .build()
  }

  test("spark2 eventlog") {
    val logFiles = Array(profEventLog("spark2-eventlog.zstd"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolOutFileCheckerImpl("check the content of summary file")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc("spark_2")))
      .build()
  }

  test("test udf event logs") {
    val logFiles = Array(
      qualEventLog("dataset_eventlog"),
      qualEventLog("dsAndDf_eventlog.zstd"),
      qualEventLog("udf_dataset_eventlog"),
      qualEventLog("udf_func_eventlog")
    )
    val expectedLabel = "simple_udf"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withMultipleThreads(4) // use multiple threads to evaluate race conditions
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app statuses are valid")
          .withExpectedCounts(StatusReportCounts(4, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(4)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the per-SQL summaries")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("missing sql end") {
    val logFiles = Array(qualEventLog("join_missing_sql_end"))
    val expectedLabel = "missing_sql_end"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app statuses are valid")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("check the per-SQL summaries")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("eventlog with no jobs") {
    // test an eventlog that has no jobs (SQL)
    val logFiles = Array(qualEventLog("empty_eventlog"))
    val expectedLabel = "empty_eventlog"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table should have 0 rows")
          .withTableLabel("perSqlCSVReport")
          .withExpectedRows("No rows", 0)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("eventlog with rdd only jobs") {
    val logFiles = Array(qualEventLog("rdd_only_eventlog"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table should have 0 rows")
          .withTableLabel("perSqlCSVReport")
          .withExpectedRows("No rows", 0))
      .build()
  }

  test("truncated log file 1") {
    val logFiles = Array(qualEventLog("truncated_eventlog"))
    val expectedLabel = "truncated_1_end"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedRows("expect 2 sqlIds", 2)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("nds q86 test") {
    val logFiles = Array(qualEventLog("nds_q86_test"))
    val expectedLabel = "nds_q86_test"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("event log rolling") {
    // event log rolling creates files under a directory
    val logFiles = Array(qualEventLog("eventlog_v2_local-1623876083964"))
    val expectedLabel = "eventlog_rolling"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("db event log rolling") {
    // these test files were setup to simulate the directory structure
    // when running on Databricks and are not really from there
    val logFiles = Array(qualEventLog("db_sim_eventlog"))
    val expectedLabel = "db_eventlog_rolling"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("db event log rolling with a rolled file dated after the current file") {
    // The rolled file names carry the rotation time; the current file has no stamp and is dated
    // when the directory is read. A rolled stamp later than that moment (a rotation shortly
    // before the run, read from a machine whose local clock is behind the stamp's zone) must
    // still sort before the current file, or the events replay out of order and every SQL
    // spanning the rotation loses its duration. The copy renames the rolled file to a
    // far-future stamp and expects the same output as the original fixture.
    val srcDir = Paths.get(qualEventLog("db_sim_eventlog"))
    val expectedLabel = "db_eventlog_rolling"
    TrampolineUtil.withTempDir { tempDir =>
      Files.copy(srcDir.resolve("eventlog"), Paths.get(tempDir.getAbsolutePath, "eventlog"))
      Files.copy(srcDir.resolve("eventlog-2021-06-15--15-00.gz"),
        Paths.get(tempDir.getAbsolutePath, "eventlog-2099-12-31--23-59.gz"))
      QToolTestCtxtBuilder(eventlogs = Array(tempDir.getAbsolutePath))
        .withPerSQL()
        .withChecker(
          QToolStatusChecker("Check that the app should succeed")
            .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
        .withChecker(
          QToolResultCoreChecker("check app count is valid and status is success")
            .withExpectedSize(1)
            .withSuccessCode())
        .withChecker(
          QToolOutFileCheckerImpl("check the core app summaries has valid data")
            .withExpectedRows("expect only 1 row", 1)
            .withExpectedLoc(expectedQualLoc(expectedLabel)))
        .withChecker(
          QToolOutFileCheckerImpl("Per-SQL table content")
            .withTableLabel("perSqlCSVReport")
            .withExpectedLoc(expectedQualLoc(expectedLabel)))
        .build()
    }
  }

  test("the undated current db event log is dated after every rolled file") {
    val current = EventLogPathProcessor.getDBEventLogFileDate("eventlog")
    val rolled = EventLogPathProcessor.getDBEventLogFileDate("eventlog-2099-12-31--23-59.gz")
    assert(current == LocalDateTime.MAX)
    assert(rolled.isBefore(current))
  }

  runConditionalTest("nds q86 with failure test",
    shouldSkipFailedLogsForSpark) {
    val logFiles = Array(qualEventLog("nds_q86_fail_test"))
    val expectedLabel = "nds_q86_fail_test"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has valid data")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("event log write format") {
    val logFiles = Array(qualEventLog("writeformat_eventlog"))
    val expectedLabel = "writeformat_eventlog"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has unsupported write format")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("event log nested types in ReadSchema") {
    val logFiles = Array(qualEventLog("nested_type_eventlog"))
    val expectedLabel = "nested_type_eventlog"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("jdbc problematic") {
    val logFiles = Array(qualEventLog("jdbc_eventlog.zstd"))
    val expectedLabel = "jdbc_eventlog"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("read datasource v2") {
    val logFiles = Array(profEventLog("eventlog_dsv2.zstd"))
    val expectedLabel = "read_dsv2"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("dsv1 complex") {
    val logFiles = Array(qualEventLog("complex_dec_eventlog.zstd"))
    val expectedLabel = "dsv1_complex"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("dsv2 nested complex") {
    val logFiles = Array(qualEventLog("eventlog_nested_dsv2"))
    val expectedLabel = "nested_dsv2"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("repeated jobName") {
    val logFiles = Array(qualEventLog("empty_eventlog"), qualEventLog("nested_type_eventlog"))
    val expectedLabel = "repeated_jobname"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("Check that the app should succeed")
          .withExpectedCounts(StatusReportCounts(2, 0, 0, 0)))
      .withChecker(
        QToolResultCoreChecker("check app count is valid and status is success")
          .withExpectedSize(2)
          .withSuccessCode())
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 2 rows", 2)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("status report generation for wildcard event log") {
    // correct wildcard event log with 3 matches
    val logFiles = Array(qualEventLog("cluster_information/eventlog_3node*"))

    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withMultipleThreads(4) // use multiple threads to evaluate race conditions
      .withChecker(
        QToolStatusChecker("3 SUCCESS, 0 FAILURE, 0 SKIPPED, 0 UNKNOWN")
          .withExpectedCounts(StatusReportCounts(3, 0, 0, 0)))
      .build()
  }

  test("status report generation for incorrect wildcard event log") {
    // incorrect wildcard event log with 2 matches
    val logFiles = Array(
      qualEventLog("cluster_information/eventlog_xxxx*"),  // incorrect wildcard event log
      qualEventLog("cluster_information/eventlog_yyyy*"))  // incorrect wildcard event log

    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withMultipleThreads(4) // use multiple threads to evaluate race conditions
      .withChecker(
        QToolStatusChecker("0 SUCCESS, 2 FAILURE, 0 SKIPPED, 0 UNKNOWN")
          .withExpectedCounts(StatusReportCounts(0, 2, 0, 0)))
      .build()
  }

  test("status report generation for mixed event log") {
    val logFiles = Array(
      qualEventLog("nds_q86_test"),                         // correct event log
      qualEventLog("cluster_information/eventlog_xxxx*"),   // incorrect wildcard event log
      qualEventLog("cluster_information/eventlog_2node*"))  // incorrect wildcard event log

    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withMultipleThreads(4) // use multiple threads to evaluate race conditions
      .withChecker(
        QToolStatusChecker("2 SUCCESS, 2 FAILURE, 0 SKIPPED, 0 UNKNOWN")
          .withExpectedCounts(StatusReportCounts(2, 1, 0, 0)))
      .build()
  }

  test("support for photon eventlog") {
    val logFiles = Array(qualEventLog("nds_q88_photon_db_13_3.zstd"))
    val expectedLabel = "photon_db_13_3"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withPlatform(PlatformNames.DATABRICKS_AWS)
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("1 SUCCESS, 0 FAILURE, 0 SKIPPED, 0 UNKNOWN")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutFileCheckerImpl("check the core app summaries has nested types")
          .withExpectedRows("expect only 1 row", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("Per-SQL table content")
          .withTableLabel("perSqlCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel))
          .withRunCondition(
            () => {
              (ToolUtils.isSpark340OrLater(),
                "Skip file comparisons for Spark [-, 3.4[ because root sqlID is not a valid field")
            }))
      .withChecker(
        QToolOutFileCheckerImpl("Execs table content")
          .withTableLabel("execCSVReport")
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("process multiple attempts of the same app ID and skip lower attempts") {
    val logFiles = Array(qualEventLog("multiple_attempts/*"))
    val expectedLabel = "support_multiple_attempts"
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withMultipleThreads(4) // use multiple threads to evaluate race conditions
      .withPerSQL()
      .withChecker(
        QToolStatusChecker("1 SUCCESS, 0 FAILURE, 3 SKIPPED because of attempts, 0 UNKNOWN")
          .withExpectedCounts(StatusReportCounts(1, 0, 3, 0)))
      .withChecker(
        QToolResultCoreChecker("check only app with higher attempt ID appears in the core-summary")
          .withExpectedSize(1)
          .withSuccessCode()
          .withCheckBlock(
            "only attemptID 4 should be present",
            qRes => {
              qRes.appSummaries.map(_.appId) shouldBe Array("application_1724877841851_0016")
              qRes.appSummaries.map(_.estimatedInfo.attemptId) shouldBe Array(4)
            }))
      .withChecker(
        QToolOutFileCheckerImpl("check that the summary file is correct")
          .withExpectedRows("only 1 app is reported", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .withChecker(
        QToolOutFileCheckerImpl("check the content of status file")
          .withExpectedRows("4 rows in the status", 4)
          .withTableLabel("coreCSVStatus")
          .withContentVisitor(
            "AttemptId [1, 3] should have description stating that they are skipped",
            csvContainer => {
              // only row with success should have attempt
              csvContainer.csvRows.foreach { r =>
                r.get("Status") match {
                  case Some("SUCCESS") =>
                    r("App ID") shouldBe "application_1724877841851_0016"
                  case Some("SKIPPED") =>
                    r("Description") should include regex
                      "skipping this attempt (1|2|3) as a newer attemptId is being processed"
                    r("App ID") shouldBe "N/A"
                  case _ =>
                    fail(s"Unexpected status ${r.get("Status")}")
                }
              }
            }))
      .withChecker(
        QToolOutFileCheckerImpl("check that the per-app folders do not include skipped attempts")
          .withTableLabel("perSqlCSVReport")
          .withExpectedRows("only 1 row in the per-SQL summary", 1)
          .withExpectedLoc(expectedQualLoc(expectedLabel)))
      .build()
  }

  test("cluster information generation is disabled") {
    val logFiles = Array(qualEventLog("cluster_information/eventlog_2nodes_8cores"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withToolArgs(Array("--no-cluster-report"))
      .withChecker(
        QToolOutFileCheckerImpl("check that the clusterinfo json file does not exist")
          .withTableLabel("clusterInfoJSONReport")
          .withNoGeneratedFile())
      .build()
  }

  runConditionalTest("test subexecutionId mapping to rootExecutionId",
    subExecutionSupportedSparkGTE340) {
    val eventlog = qualEventLog("db_subExecution_id.zstd")
    val app = createAppFromEventlog(eventlog)
    // Get sum of durations of all the sqlIds. It contains duplicate values
    val totalSqlDuration =
      app.sqlIdToInfo.values.map(x => x.duration.getOrElse(0L)).sum

    // This is to group the sqlIds based on the rootExecutionId. So that we can verify the
    // subExecutionId to rootExecutionId mapping.
    val rootIdToSqlId = app.sqlIdToInfo.groupBy { case (_, info) =>
      info.rootExecutionID
    }
    assert(rootIdToSqlId(Some(5L)).keySet == Set(5, 6, 7, 8, 9, 10))
    QToolTestCtxtBuilder(eventlogs = Array(eventlog))
      .withChecker(
        QToolResultCoreChecker("check the sql durations is counted twice")
          .withExpectedSize(1)
          .withSuccessCode()
          .withCheckBlock(
            "make sure the durations of the sqlId's are not double counted",
            qRes => {
              assert (qRes.appSummaries.head.sparkSqlDFWallClockDuration < totalSqlDuration)
            }))
      .build()
  }

  test("MLFunctions with xgboost") {
    QToolTestCtxtBuilder(eventlogs = Array(qualEventLog("xgboost_eventlog.zstd")))
      .withToolArgs(Array("--ml-functions"))
      .withChecker(
        QToolOutFileCheckerImpl("Check the MLFunctions report")
          .withTableLabel("mlFunctionsCSVReport")
          .withContentVisitor(
            "check the columns to count the stageIDs and the functionNames",
            csvF => {
              csvF.getColumn("Stage ID") should have size 11
              csvF.getColumn("ML Functions").count { mlFn =>
                mlFn.contains("ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier.train")
              } shouldBe 4
            }))
      .withChecker(
        QToolOutFileCheckerImpl("Check the MLFunctionsDurations report")
          .withTableLabel("mlFunctionsDurationsCSVReport")
          .withContentVisitor(
            "ml_functions_durations should contain the correct function name XGBoost",
            csvF => {
              csvF.getColumn("Stage Ids").mkString.split(";") should have size 4
              csvF.getColumn("ML Function Name") shouldBe Array("XGBoost")
            }))
      .build()
  }

  // Expected results as a map of platform -> cluster info.
  val expectedPlatformClusterInfoMap: Seq[(String, SourceClusterInfo)] = Seq(
    PlatformNames.DATABRICKS_AWS ->
      SourceClusterInfo(vendor = PlatformNames.DATABRICKS_AWS,
        coresPerExecutor = 8,
        numExecsPerNode = 1,
        numExecutors = 2,
        numWorkerNodes = 2,
        executorHeapMemory = 0L,
        dynamicAllocationEnabled = false,
        "N/A", "N/A", "N/A",
        driverNodeType = Some("m6gd.2xlarge"),
        workerNodeType = Some("m6gd.2xlarge"),
        driverHost = Some("10.10.10.100"),
        clusterId = Some("1212-214324-test"),
        clusterName = Some("test-db-aws-cluster")),
    PlatformNames.DATABRICKS_AZURE ->
      SourceClusterInfo(vendor = PlatformNames.DATABRICKS_AZURE,
        coresPerExecutor = 8,
        numExecsPerNode = 1,
        numExecutors = 2,
        numWorkerNodes = 2,
        executorHeapMemory = 0L,
        dynamicAllocationEnabled = false,
        "N/A", "N/A", "N/A",
        driverNodeType = Some("Standard_E8ds_v4"),
        workerNodeType = Some("Standard_E8ds_v4"),
        driverHost = Some("10.10.10.100"),
        clusterId = Some("1212-214324-test"),
        clusterName = Some("test-db-azure-cluster")),
    PlatformNames.DATAPROC ->
      SourceClusterInfo(vendor = PlatformNames.DATAPROC,
        coresPerExecutor = 8,
        numExecsPerNode = 1,
        numExecutors = 2,
        numWorkerNodes = 2,
        executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false,
        "N/A", "N/A", "N/A",
        driverHost = Some("dataproc-test-m.c.internal")),
    PlatformNames.EMR ->
      SourceClusterInfo(vendor = PlatformNames.EMR,
        coresPerExecutor = 8,
        numExecsPerNode = 1,
        numExecutors = 2,
        numWorkerNodes = 2,
        executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false,
        "N/A", "N/A", "N/A",
        driverHost = Some("10.10.10.100"),
        clusterId = Some("j-123AB678XY321")),
    PlatformNames.ONPREM ->
      SourceClusterInfo(vendor = PlatformNames.ONPREM,
        coresPerExecutor = 8,
        numExecsPerNode = 1,
        numExecutors = 2,
        numWorkerNodes = 2,
        executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false,
        "N/A", "N/A", "N/A",
        driverHost = Some("10.10.10.100"))
  )

  expectedPlatformClusterInfoMap.foreach { case (platform, expectedClusterInfo) =>
    test(s"test cluster information JSON for platform - $platform ") {
      val logFile = qualEventLog(s"cluster_information/platform/$platform")
      runQualificationAndTestClusterInfo(logFile, platform, Some(expectedClusterInfo))
    }
  }

  // Expected results as a map of event log -> cluster info.
  // scalastyle:off line.size.limit
  val expectedClusterInfoMap: Seq[(String, Option[SourceClusterInfo])] = Seq(
    "eventlog_2nodes_8cores" -> // 2 executor nodes with 8 cores.
      Some(SourceClusterInfo(vendor = PlatformNames.DEFAULT, coresPerExecutor = 8,
        numExecsPerNode = 1, numExecutors = 2, numWorkerNodes = 2, executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false, "N/A", "N/A", "N/A", driverHost = Some("10.10.10.100"))),
    "eventlog_3nodes_12cores_multiple_executors" -> // 3 nodes, each with 2 executors having 12 cores.
      Some(SourceClusterInfo(vendor = PlatformNames.DEFAULT, coresPerExecutor = 12,
        numExecsPerNode = -1, numExecutors = 4, numWorkerNodes = 3, executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false, "N/A", "N/A", "N/A", driverHost = Some("10.59.184.210"))),
    "eventlog_4nodes_8cores_dynamic_alloc.zstd" -> // using dynamic allocation, total of 5 nodes, each with max 7
      // executor running having 4 cores. At the end it had 1 active executor.
      Some(SourceClusterInfo(vendor = PlatformNames.DEFAULT, coresPerExecutor = 4,
        numExecsPerNode = -1, numExecutors = 7, numWorkerNodes = 5, executorHeapMemory = 20480L,
        dynamicAllocationEnabled = true, dynamicAllocationMaxExecutors = "2147483647",
        dynamicAllocationMinExecutors = "0", dynamicAllocationInitialExecutors = "2",
        driverHost = Some("10.10.6.9"))),
    "eventlog_3nodes_12cores_variable_cores" -> // 3 nodes with varying cores: 8, 12, and 8, each with 1 executor.
      Some(SourceClusterInfo(vendor = PlatformNames.DEFAULT, coresPerExecutor = 12,
        numExecsPerNode = 1, numExecutors = 3, numWorkerNodes = 3, executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false, "N/A", "N/A", "N/A", driverHost = Some("10.10.10.100"))),
    "eventlog_3nodes_12cores_exec_removed" -> // 2 nodes, each with 1 executor having 12 cores, 1 executor removed.
      Some(SourceClusterInfo(vendor = PlatformNames.DEFAULT, coresPerExecutor = 12,
        numExecsPerNode = 1, numExecutors = 2, numWorkerNodes = 2, executorHeapMemory = 1024L,
        dynamicAllocationEnabled = false, "N/A", "N/A", "N/A", driverHost = Some("10.10.10.100"))),
    "eventlog_driver_only" -> None // Event log with driver only
  )
  // scalastyle:on line.size.limit

  expectedClusterInfoMap.foreach { case (eventlogPath, expectedClusterInfo) =>
    test(s"test cluster information JSON - $eventlogPath") {
      val logFile = qualEventLog(s"cluster_information/$eventlogPath")
      runQualificationAndTestClusterInfo(logFile, PlatformNames.DEFAULT, expectedClusterInfo)
    }
  }

  /**
   * Runs the qualification tool and verifies cluster information against expected values.
   */
  private def runQualificationAndTestClusterInfo(
      eventlogPath: String,
      platform: String,
      expectedClusterInfo: Option[SourceClusterInfo]): Unit = {
    QToolTestCtxtBuilder(eventlogs = Array(eventlogPath))
      .withPlatform(platform)
      .withChecker(
        QToolResultCoreChecker("check the tool is successful")
          .withExpectedSize(1)
          .withSuccessCode())
      .withChecker(
        QToolOutJsonFileCheckerImpl("check that the clusterinfo json content is valid")
          .withContentVisitor {
            (_, f) =>
              val actualClusterInfo =
                ToolTestUtils.loadClusterSummaryFromJson(f).sourceClusterInfo
              actualClusterInfo shouldBe expectedClusterInfo
          })
      .build()
  }

  test("UDF report - Scala UDF via DataFrame API") {
    val logFiles = Array(qualEventLog("udf_func_eventlog"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutJsonFileCheckerImpl("UDF report detects Scala UDFs")
          .withTableLabel("udfReportJSON")
          .withContentVisitor((_, jsonFile) => {
            val (udfs, metrics) = readUdfReport(jsonFile)
            udfs.size shouldBe 3
            // All UDFs are inside Project execs, across SQL IDs 2, 5, 8
            udfs.foreach(_("exec") shouldBe "Project")
            udfs.map(u => u("sql_id").asInstanceOf[BigInt].toInt).sorted shouldBe Seq(2, 5, 8)
            udfs.map(u => u("stage_id").asInstanceOf[BigInt].toInt).sorted shouldBe Seq(2, 5, 8)
            metrics shouldBe defined
            metrics.get("unsupported_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 982
            metrics.get("app_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 4666
            metrics.get("unsupported_task_duration_pct").asInstanceOf[Double] shouldBe 21.0
          }))
      .build()
  }

  test("UDF report - Scala UDF with Dataset") {
    val logFiles = Array(qualEventLog("udf_dataset_eventlog"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutJsonFileCheckerImpl("UDF report detects Dataset UDF")
          .withTableLabel("udfReportJSON")
          .withContentVisitor((_, jsonFile) => {
            val (udfs, metrics) = readUdfReport(jsonFile)
            udfs.size shouldBe 1
            udfs.head("exec") shouldBe "Project"
            udfs.head("sql_id").asInstanceOf[BigInt].toInt shouldBe 0
            udfs.head("stage_id").asInstanceOf[BigInt].toInt shouldBe 1
            metrics shouldBe defined
            metrics.get("unsupported_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 170
            metrics.get("app_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 283
            metrics.get("unsupported_task_duration_pct").asInstanceOf[Double] shouldBe 60.1
          }))
      .build()
  }

  test("UDF report - Pandas UDF (ArrowEvalPython)") {
    val logFiles = Array(qualEventLog("pandas_execs_eventlog.zstd"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutJsonFileCheckerImpl("UDF report detects Pandas UDF")
          .withTableLabel("udfReportJSON")
          .withContentVisitor((_, jsonFile) => {
            val (udfs, metrics) = readUdfReport(jsonFile)
            udfs.size shouldBe 1
            udfs.head("name") shouldBe "ArrowEvalPython"
            udfs.head("exec") shouldBe "ArrowEvalPython"
            udfs.head("sql_id").asInstanceOf[BigInt].toInt shouldBe 0
            udfs.head("stage_id").asInstanceOf[BigInt].toInt shouldBe 0
            metrics shouldBe defined
            metrics.get("unsupported_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 28449
            metrics.get("app_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 142402
            metrics.get("unsupported_task_duration_pct").asInstanceOf[Double] shouldBe 20.0
          }))
      .build()
  }

  test("UDF report - Java/Hive UDF") {
    val logFiles = Array(qualEventLog("java_udf_eventlog"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutJsonFileCheckerImpl("UDF report detects Java/Hive UDF")
          .withTableLabel("udfReportJSON")
          .withContentVisitor((_, jsonFile) => {
            val (udfs, metrics) = readUdfReport(jsonFile)
            udfs.size shouldBe 1
            udfs.head("name") shouldBe "IntegerMultiplyBy2UDF"
            udfs.head("exec") shouldBe "Project"
            udfs.head("sql_id").asInstanceOf[BigInt].toInt shouldBe 2
            udfs.head("stage_id").asInstanceOf[BigInt].toInt shouldBe 1
            metrics shouldBe defined
            metrics.get("unsupported_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 44
            metrics.get("app_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 176
            metrics.get("unsupported_task_duration_pct").asInstanceOf[Double] shouldBe 25.0
          }))
      .build()
  }

  test("UDF report - Python UDF (BatchEvalPython)") {
    val logFiles = Array(qualEventLog("python_udf_eventlog"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutJsonFileCheckerImpl("UDF report detects Python UDF")
          .withTableLabel("udfReportJSON")
          .withContentVisitor((_, jsonFile) => {
            val (udfs, metrics) = readUdfReport(jsonFile)
            udfs.size shouldBe 1
            udfs.head("name") shouldBe "BatchEvalPython"
            udfs.head("exec") shouldBe "BatchEvalPython"
            udfs.head("sql_id").asInstanceOf[BigInt].toInt shouldBe 0
            udfs.head("stage_id").asInstanceOf[BigInt].toInt shouldBe 1
            metrics shouldBe defined
            metrics.get("unsupported_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 186
            metrics.get("app_task_duration_ms").asInstanceOf[BigInt].toLong shouldBe 467
          }))
      .build()
  }

  test("UDF report - no UDFs") {
    val logFiles = Array(qualEventLog("nds_q86_test"))
    QToolTestCtxtBuilder(eventlogs = logFiles)
      .withChecker(
        QToolStatusChecker("app should succeed")
          .withExpectedCounts(StatusReportCounts(1, 0, 0, 0)))
      .withChecker(
        QToolOutJsonFileCheckerImpl("UDF report shows no UDFs")
          .withTableLabel("udfReportJSON")
          .withContentVisitor((_, jsonFile) => {
            val (udfs, metrics) = readUdfReport(jsonFile)
            udfs shouldBe empty
            // metrics is None in the report (serialized as null in JSON);
            // extractOpt returns Some(Map()) for null, so check emptiness
            metrics.forall(_.isEmpty) shouldBe true
          }))
      .build()
  }
}
