/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.planparser

import java.io.{File, PrintWriter}

import scala.collection.mutable
import scala.util.control.NonFatal

import com.nvidia.spark.rapids.tool.ToolTestUtils
import com.nvidia.spark.rapids.tool.planparser.delta.DeltaLakeOps
import com.nvidia.spark.rapids.tool.planparser.ops.{ExprOpRef, OpActions, OpRef, OpTypes}
import com.nvidia.spark.rapids.tool.qualification._
import org.scalatest.exceptions.TestFailedException
import org.scalatest.matchers.should.Matchers
import org.scalatest.prop.TableDrivenPropertyChecks._
import org.scalatest.prop.TableFor2

import org.apache.spark.sql.{DataFrame, SparkSession, TrampolineUtil}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.rapids.tool.ExecHelper._
import org.apache.spark.sql.rapids.tool.ToolUtils
import org.apache.spark.sql.rapids.tool.plangraph.{SQLPlanMetric, ToolsPlanGraph}
import org.apache.spark.sql.rapids.tool.util.{FSUtils, UTF8Source}

class SQLPlanParserSuite extends BasePlanParserSuite with Matchers {

  test("Error parser does not cause entire app to fail") {
    // The purpose of this test is to make sure that the SQLParser won't trigger an exception that
    // causes the entire app analysis to fail.
    // In order to simulate unexpected scenarios, the test modifies the eventlog on the fly by
    // injecting faulty expressions.
    //
    // For example:
    //    Filter (((value#8 <> 100) AND (value#8 > 50)) OR (value#8 = 0))
    //    One of the predicates is converted from "<" to "<>".
    //    The latter is an invalid predicate causing the parser to throw a scala-match error.
    TrampolineUtil.withTempDir { eventLogDir =>
      // generate the original eventlog
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "WholeStageFilterProject") { spark =>
        import spark.implicits._
        val df = spark.sparkContext.makeRDD(1 to 100, 3).toDF
        val df2 = spark.sparkContext.makeRDD(1 to 100, 3).toDF
        df.select($"value" as "a")
          .join(df2.select($"value" as "b"), $"a" === $"b")
          .filter("(((b < 100) AND (a > 50)) OR (a = 0))")
          .sort($"b")
      }
      // create a temporary file to write the modified events
      val faultyEventlog = new File(s"$eventLogDir/faulty_eventlog")
      val pWriter = new PrintWriter(faultyEventlog)
      val bufferedSource = UTF8Source.fromFile(eventLog)
      try {
        bufferedSource.getLines.map( l =>
          if (l.contains("SparkListenerSQLExecutionStart")) {
            l.replaceAll("value#2 <", "value#2 <>")
          } else {
            l
          }
        ).foreach { modifiedLine =>
          // scalastyle:off println
          pWriter.println(modifiedLine)
          // scalastyle:on println
        }
      } finally {
        bufferedSource.close()
        pWriter.close()
      }
      // start processing the faulty eventlog.
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(faultyEventlog.getAbsolutePath)
      assert(app.sqlPlans.size == 1)
      try {
        app.sqlPlans.foreach { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
      } catch {
        case NonFatal(e) =>
          throw new TestFailedException(
            s"The SQLParser crashed while processing incorrect expression", e, 0)
      }
    }
  }

  test("WholeStage with Filter, Project, Sort and SortMergeJoin") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "WholeStageFilterProject") { spark =>
        import spark.implicits._
        val df = spark.sparkContext.makeRDD(1 to 100000, 6).toDF
        val df2 = spark.sparkContext.makeRDD(1 to 100000, 6).toDF
        df.select( $"value" as "a")
          .join(df2.select($"value" as "b"), $"a" === $"b")
          .filter($"b" < 100)
          .sort($"b")
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      val sparkVersion = app.sparkVersion
      assert(app.sqlPlans.size == 1)
      app.sqlPlans.foreach { case (sqlID, plan) =>
        val planInfo = SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        verifyPlanExecToStageMap(planInfo)
        val wholeStages = planInfo.execInfo.filter(_.exec.contains("WholeStageCodegen"))
        val allChildren = wholeStages.flatMap(_.children).flatten
        val sorts = allChildren.filter(_.exec == "Sort")
        val filters = allChildren.filter(_.exec == "Filter")
        val projects = allChildren.filter(_.exec == "Project")
        val smj = allChildren.filter(_.exec == "SortMergeJoin")
        val bhj = allChildren.filter(_.exec == "BroadcastHashJoin")
        val (execInfoSize, wholeStagesSize, numSupported, expChildren, numSort, numSMJ, numBHJ) =
          if (ToolUtils.isSpark320OrLater(sparkVersion)) {
            // - The plan has BroadcastExchange which will change the expected number of execs.
            //   Keeping in mind that the parser marks duplicate execs as "shouldRemove" instead of
            //   deleting them
            // - All "wholestages" should all supported
            // - The children of wholestages will be down by 2 compared to pre-3.2
            //    Sort
            //    BroadcastHashJoin
            //    Project
            //    Filter
            //    SerializeFromObject
            //    Project
            //    Filter
            //    SerializeFromObject
            (14, 4, 4, 8, 1, 0, 1)
          } else {
            // only 2 in the above example have projects and filters, 3 have sort and 1 has SMJ
            // - The children of wholestages will be:
            //    Sort
            //    SortMergeJoin
            //    Sort
            //    Project
            //    Filter
            //    SerializeFromObject
            //    Sort
            //    Project
            //    Filter
            //    SerializeFromObject
            (11, 6, 6, 10, 3, 1, 0)
          }
        assert(planInfo.execInfo.size == execInfoSize)
        assert(wholeStages.size == wholeStagesSize)
        assert(wholeStages.filter(_.isSupported).size == numSupported)
        assert(wholeStages.forall(_.duration.nonEmpty))
        assert(allChildren.size == expChildren)
        assertSizeAndSupported(2, filters)
        assertSizeAndSupported(2, projects)
        assertSizeAndSupported(numSort, sorts)
        assertSizeAndSupported(numSMJ, smj)
        assertSizeAndSupported(numBHJ, bhj)
      }
    }
  }

  test("Parse Execs within WholeStageCodeGen in Order") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "Execs within WSCG ") { spark =>
        import spark.implicits._
        val df = Seq(("foo", 1L, 1.2), ("foo", 2L, 2.2), ("bar", 2L, 3.2),
          ("bar", 2L, 4.2)).toDF("x", "y", "z")
        df.cube($"x", ceil($"y")).count
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      app.sqlPlans.foreach { case (sqlID, plan) =>
        val planInfo = SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        verifyPlanExecToStageMap(planInfo)
        val allExecInfo = planInfo.execInfo
        // Note that:
        // Spark320+ will generate the following execs:
        //   AdaptiveSparkPlan, WholeStageCodegen, AQEShuffleRead, Exchange, WholeStageCodegen
        // Other Sparks:
        //   WholeStageCodegen, Exchange, WholeStageCodegen
        val wholeStages = planInfo.execInfo.filter(_.exec.contains("WholeStageCodegen"))
        assert(wholeStages.size == 2)
        // Expanding the children of WholeStageCodegen
        val allExecs = allExecInfo.map(x => if (x.exec.startsWith("WholeStage")) {
          x.children.getOrElse(Seq.empty)
        } else {
          Seq(x)
        }).flatten.reverse
        val expectedOrder = if (ToolUtils.isSpark320OrLater()) {
          // Order should be: LocalTableScan, Expand, HashAggregate, Exchange,
          // AQEShuffleRead, HashAggregate, AdaptiveSparkPlan
          Seq("LocalTableScan", "Expand", "HashAggregate", "Exchange", "AQEShuffleRead",
            "HashAggregate", "AdaptiveSparkPlan")
        } else {
          // Order should be: LocalTableScan, Expand, HashAggregate, Exchange, HashAggregate
          Seq("LocalTableScan", "Expand", "HashAggregate", "Exchange", "HashAggregate")
        }
        assert(allExecs.map(_.exec) == expectedOrder)
      }
    }
  }

  test("HashAggregate") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "sqlMetric") { spark =>
        import spark.implicits._
        spark.range(10).
            groupBy('id % 3 as "group").agg(sum("id") as "sum")
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      app.sqlPlans.foreach { case (sqlID, plan) =>
        val planInfo = SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        verifyPlanExecToStageMap(planInfo)
        val wholeStages = planInfo.execInfo.filter(_.exec.contains("WholeStageCodegen"))
        assert(wholeStages.size == 2)
        val numSupported = wholeStages.filter(_.isSupported).size
        assert(numSupported == 2)
        assert(wholeStages.forall(_.duration.nonEmpty))
        val allChildren = wholeStages.flatMap(_.children).flatten
        val hashAggregate = allChildren.filter(_.exec == "HashAggregate")
        assertSizeAndSupported(2, hashAggregate, checkDurations = false)
      }
    }
  }

  test("FileSourceScan") {
    val eventLog = s"$profileLogDir/eventlog_dsv1.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size == 7)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    val json = allExecInfo.filter(_.exec.contains("Scan json"))
    val orc = allExecInfo.filter(_.exec.contains("Scan orc"))
    val parquet = allExecInfo.filter(_.exec.contains("Scan parquet"))
    val text = allExecInfo.filter(_.exec.contains("Scan text"))
    val csv = allExecInfo.filter(_.exec.contains("Scan csv"))
    assertSizeAndSupported(2, json)
    assertSizeAndNotSupported(1, text)
    for (t <- Seq(parquet, csv)) {
      assertSizeAndSupported(1, t)
    }
    assertSizeAndSupported(2, orc)
  }

  test("BatchScan") {
    val eventLog = s"$profileLogDir/eventlog_dsv2.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size == 9)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    // Note that the text scan from this file is v1 so ignore it
    val json = allExecInfo.filter(_.exec.contains("BatchScan json"))
    val orc = allExecInfo.filter(_.exec.contains("BatchScan orc"))
    val parquet = allExecInfo.filter(_.exec.contains("BatchScan parquet"))
    val csv = allExecInfo.filter(_.exec.contains("BatchScan csv"))
    assertSizeAndSupported(3, json)
    assertSizeAndSupported(1, csv)
    for (t <- Seq(orc, parquet)) {
      assertSizeAndSupported(2, t)
    }
  }

  test("Subquery exec should be ignored without impact on performance") {
    TrampolineUtil.withTempDir { outputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "subqueryTest") { spark =>
          import spark.implicits._
          val df = spark.sparkContext.makeRDD(1 to 10000, 6).toDF
          df.createOrReplaceTempView("t1")
          val df2WithSubquery =
            spark.sql("select * from t1 where value < (select avg(value) from t1)")
          df2WithSubquery
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        app.sqlPlans.size shouldBe 2
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val subqueryExecs = allExecInfo.filter(_.exec.contains(s"Subquery"))
        val summaryRecs = subqueryExecs.flatten { sqExec =>
          sqExec.isSupported  shouldNot be(true)
          sqExec.getUnsupportedExecSummaryRecord(0)
        }
        summaryRecs.size shouldBe 1
        val summaryRec = summaryRecs.head
        summaryRec.opType shouldBe OpTypes.Exec
        summaryRec.opAction shouldBe OpActions.IgnoreNoPerf
      }
    }
  }

  test("InsertIntoHadoopFsRelationCommand") {
    val dataWriteCMD = DataWritingCommandExecParser.insertIntoHadoopCMD
    TrampolineUtil.withTempDir { outputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, dataWriteCMD) { spark =>
          import spark.implicits._
          val df = spark.sparkContext.makeRDD(1 to 10000, 6).toDF
          val dfWithStrings = df.select(col("value").cast("string"))
          dfWithStrings.write.text(s"$outputLoc/testtext")
          df.write.parquet(s"$outputLoc/testparquet")
          df.write.orc(s"$outputLoc/testorc")
          df.write.json(s"$outputLoc/testjson")
          df.write.csv(s"$outputLoc/testcsv")
          df
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        assert(app.sqlPlans.size == 6)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val writeExecs = allExecInfo.filter(_.exec.contains(s"$dataWriteCMD"))
        val text = writeExecs.filter(_.expr.contains("text"))
        val json = writeExecs.filter(_.expr.contains("json"))
        val orc = writeExecs.filter(_.expr.contains("orc"))
        val parquet = writeExecs.filter(_.expr.contains("parquet"))
        val csv = writeExecs.filter(_.expr.contains("csv"))
        for (t <- Seq(json, csv, text)) {
          assertSizeAndNotSupported(1, t)
        }
        for (t <- Seq(orc, parquet)) {
          assertSizeAndSupported(1, t)
        }
      }
    }
  }

  runConditionalTest("WriteFilesExec is marked as Supported",
    execsSupportedSparkGTE340) {
    val dataWriteCMD = DataWritingCommandExecParser.insertIntoHadoopCMD
    TrampolineUtil.withTempDir { outputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, dataWriteCMD) { spark =>
          import spark.implicits._
          val df = spark.sparkContext.makeRDD(1 to 10000, 6).toDF
          val dfWithStrings = df.select(col("value").cast("string"))
          dfWithStrings.write.text(s"$outputLoc/testtext")
          df.write.parquet(s"$outputLoc/testparquet")
          df.write.orc(s"$outputLoc/testorc")
          df.write.json(s"$outputLoc/testjson")
          df.write.csv(s"$outputLoc/testcsv")
          df
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        assert(app.sqlPlans.size == 6)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val writeExecs = allExecInfo.filter(_.exec.contains(s"$dataWriteCMD"))
        val text = writeExecs.filter(_.expr.contains("text"))
        val json = writeExecs.filter(_.expr.contains("json"))
        val orc = writeExecs.filter(_.expr.contains("orc"))
        val parquet = writeExecs.filter(_.expr.contains("parquet"))
        val csv = writeExecs.filter(_.expr.contains("csv"))
        for (t <- Seq(json, csv, text)) {
          assertSizeAndNotSupported(1, t)
        }
        for (t <- Seq(orc, parquet)) {
          assertSizeAndSupported(1, t)
        }
        // For Spark.3.4.0+, the new operator WriteFilesExec is added.
        // Q tool handles this operator as supported regardless of the type of the exec operation.
        val writeFileExecs = allExecInfo.filter(_.exec.contains(EX_NAME_WRITE_FILES))
        // we have 5 write operations, so we should have 5 WriteFilesExec.
        assertSizeAndSupported(5, writeFileExecs)
      }
    }
  }

  test("CreateDataSourceTableAsSelectCommand") {
    // using event log to not deal with enabling hive support
    val eventLog = s"$qualLogDir/createdatasourcetable_eventlog.zstd"
    val app = createAppFromEventlog(eventLog)
    val pluginTypeChecker = new PluginTypeChecker()
    assert(app.sqlPlans.size == 1)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    val parquet = {
      allExecInfo.filter(_.exec.contains("CreateDataSourceTableAsSelectCommand"))
    }
    assertSizeAndNotSupported(1, parquet.toSeq)
  }

  test("Stages and jobs failure") {
    val eventLog = s"$profileLogDir/tasks_executors_fail_compressed_eventlog.zstd"
    val app = createAppFromEventlog(eventLog)
    val stats = app.aggregateStats()
    assert(stats.nonEmpty)
  }

  test("InMemoryTableScan") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "InMemoryTableScan") { spark =>
        import spark.implicits._
        val df = spark.sparkContext.makeRDD(1 to 10000, 6).toDF
        val dfc = df.cache()
        dfc
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val tableScan = allExecInfo.filter(_.exec == ("InMemoryTableScan"))
      assertSizeAndSupported(1, tableScan.toSeq)
    }
  }

  test("BroadcastExchangeExec, SubqueryBroadcastExec and Exchange") {
    val eventLog = s"$qualLogDir/nds_q86_test"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size > 0)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    val broadcasts = allExecInfo.filter(_.exec == "BroadcastExchange")
    assertSizeAndSupported(3, broadcasts.toSeq,
      expectedDur = Seq(Some(1154), Some(1154), Some(1855)))
    val subqueryBroadcast = allExecInfo.filter(_.exec == "SubqueryBroadcast")
    assertSizeAndSupported(1, subqueryBroadcast.toSeq, expectedDur = Seq(Some(1175)))
    val exchanges = allExecInfo.filter(_.exec == "Exchange")
    assertSizeAndSupported(2, exchanges.toSeq, expectedDur = Seq(Some(15688), Some(8)))
  }

  test("ReusedExchangeExec") {
    // The eventlog has a ReusedExchange that causes multiple nodes to be duplicate.
    // This unit test is to check that we set the duplicate nodes as shouldRemove
    // to avoid adding them multiple times to the speedups.
    val eventLog = s"$qualLogDir/nds_q86_test"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    // reusedExchange is added as a supportedExec
    val reusedExchangeExecs = allExecInfo.filter(_.exec == "ReusedExchange")
    assertSizeAndSupported(1, reusedExchangeExecs)
    val nodesWithRemove = allExecInfo.filter(_.shouldRemove)
    // There are 7 nodes that are duplicated including 1 ReusedExchangeExec node and
    // 1 WholeStageCodegen.
    assert(1 == nodesWithRemove.count(
      exec => exec.exec.contains("WholeStageCodegen")))
    assert(1 == nodesWithRemove.count(exec => exec.exec.equals("ReusedExchange")))
  }

  test("CustomShuffleReaderExec") {
    // this is eventlog because CustomShuffleReaderExec only available before 3.2.0
    val eventLog = s"$qualLogDir/customshuffle_eventlog.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size > 0)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    val reader = allExecInfo.filter(_.exec == "CustomShuffleReader")
    assertSizeAndSupported(2, reader.toSeq)
  }

  test("AQEShuffleReadExec") {
    // this is eventlog because AQEShuffleReadExec only available after 3.2.0
    val eventLog = s"$qualLogDir/aqeshuffle_eventlog.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size > 0)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    val reader = allExecInfo.filter(_.exec == "AQEShuffleRead")
    assertSizeAndSupported(2, reader.toSeq)
  }

  test("Parsing various Execs - Coalesce, CollectLimit, Expand, Range, Sample" +
      "TakeOrderedAndProject and Union") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val df1 = spark.sparkContext.makeRDD(1 to 1000, 6).toDF
        df1.coalesce(1).collect // Coalesce
        spark.range(10).where(col("id") === 2).collect // Range
        // TakeOrderedAndProject
        df1.orderBy($"value", ceil($"value"), round($"value")).limit(10).collect
        df1.limit(10).collect // CollectLimit
        df1.union(df1).collect // Union
        df1.rollup(ceil(col("value"))).agg(ceil(col("value"))).collect // Expand
        df1.sample(0.1) // Sample
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 7)
      val supportedExecs = Array("Coalesce", "Expand", "Range", "Sample",
        "TakeOrderedAndProject", "Union")
      val unsupportedExecs = Array("CollectLimit")
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      for (execName <- supportedExecs) {
        val execs = allExecInfo.filter(_.exec == execName)
        assertSizeAndSupported(1, execs, expectedDur = Seq.empty, extraText = execName)
      }
      for (execName <- unsupportedExecs) {
        val execs = allExecInfo.filter(_.exec == execName)
        assertSizeAndNotSupported(1, execs)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
    }
  }

  test("Parse Execs - CartesianProduct and Generate") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val genDf = spark.sparkContext.parallelize(List(List(1, 2, 3), List(4, 5, 6)), 4).toDF
        val joinDf1 = spark.sparkContext.makeRDD(1 to 10, 4).toDF
        val joinDf2 = spark.sparkContext.makeRDD(1 to 10, 4).toDF
        genDf.select(explode($"value")).collect
        joinDf1.crossJoin(joinDf2)
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 2)
      val supportedExecs = Array("CartesianProduct", "Generate")
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      for (execName <- supportedExecs) {
        val supportedExec = allExecInfo.filter(_.exec == execName)
        assertSizeAndSupported(1, supportedExec.toSeq)
      }
    }
  }

  test("Parse Execs and supported exprs - BroadcastHashJoin, " +
    "BroadcastNestedLoopJoin and ShuffledHashJoin") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val df1 = Seq((1, "ABC", 1.2), (2, "DEF", 2.3), (3, "GHI", 1.4)).toDF(
          "emp_id", "name", "dept")
        val df2 = Seq(("Fin", 1.2, "ABC"), ("IT", 2.3, "DEF")).toDF(
          "dept_name", "dept_id", "name")
        // BroadcastHashJoin
        df1.join(df2, df1("name") === df2("name") &&
          ceil(df1("dept")) === ceil(df2("dept_id")), "left_outer").collect
        // ShuffledHashJoin
        df1.createOrReplaceTempView("t1")
        df2.createOrReplaceTempView("t2")
        spark.sql("select /*+ SHUFFLE_HASH(t1) */ * from t1 " +
          "INNER JOIN t2 ON t1.name = t2.name AND CEIL(t1.DEPT) = CEIL(t2.DEPT_ID)").collect
        // BroadcastNestedLoopJoin
        df1.join(df2, ceil(df1("dept")) <= ceil(df2("dept_id"))
          && floor(df1("dept")) <= floor(df2("dept_id")), "inner")
      }

      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 5)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val bhj = allExecInfo.filter(_.exec == "BroadcastHashJoin")
      assertSizeAndSupported(1, bhj)
      val broadcastNestedJoin = allExecInfo.filter(_.exec == "BroadcastNestedLoopJoin")
      assertSizeAndSupported(1, broadcastNestedJoin)
      val shj = allExecInfo.filter(_.exec == "ShuffledHashJoin")
      // sjh will get sql time metrics "time in aggregation build"
      // but the UT might fail everytime due to the time difference.
      // Therefore, we should disable the verification of the durations
      assertSizeAndSupported(1, shj, checkDurations = false)
    }
  }

  test("Expressions not supported in  BroadcastHashJoin, BroadcastNestedLoopJoin " +
    "and ShuffledHashJoin") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val df1 = Seq((1, "ABC", 1.2), (2, "DEF", 2.3), (3, "GHI", 1.4)).toDF(
          "emp_id", "name", "dept")
        val df2 = Seq(("Fin", 1.2, "ABC"), ("IT", 2.3, "DEF")).toDF(
          "dept_name", "dept_id", "name")
        // BroadcastHashJoin
        // hex is not supported in GPU yet.
        df1.join(df2, df1("name") === df2("name") &&
          hex(df1("dept")) === hex(df2("dept_id")), "left_outer").collect
        // ShuffledHashJoin
        df1.createOrReplaceTempView("t1")
        df2.createOrReplaceTempView("t2")
        spark.sql("select /*+ SHUFFLE_HASH(t1) */ * from t1 " +
          "INNER JOIN t2 ON t1.name = t2.name AND HEX(t1.DEPT) = HEX(t2.DEPT_ID)").collect
        // BroadcastNestedLoopJoin
        df1.join(df2, ceil(df1("dept")) <= ceil(df2("dept_id"))
          && hex(df1("dept")) <= hex(df2("dept_id")), "inner")
      }

      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 5)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val bhj = allExecInfo.filter(_.exec == "BroadcastHashJoin")
      assertSizeAndNotSupported(1, bhj)
      val broadcastNestedJoin = allExecInfo.filter(_.exec == "BroadcastNestedLoopJoin")
      assertSizeAndNotSupported(1, broadcastNestedJoin)
      val shj = allExecInfo.filter(_.exec == "ShuffledHashJoin")
      // sjh will get sql time metrics "time in aggregation build"
      // but the UT might fail everytime due to the time difference.
      // Therefore, we should disable the verification of the durations
       assertSizeAndNotSupported(1, shj, checkDurations = false)
    }
  }

  test("HyperLogLogPlusPlus is supported in aggregations(non databricks support only)") {
    // Currently only supported for non-databricks environments
    // Being tracked here - https://github.com/NVIDIA/cudf-spark/issues/12388
    // Current test is a generic one for just testing the functionality
    // Can be extended later to support environments based testing
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) =
        ToolTestUtils.generateEventLog(eventLogDir, "ApproxCountDistinctTest") { spark =>
          import spark.implicits._
          // Test data with duplicates
          val data = Seq(1, 1, 2, 3, 3, 3)
          val df = data.toDF("value")

          // Apply approx_count_distinct with tolerance
          df.agg(approx_count_distinct("value", 0.05).alias("distinct_count"))
        }
      // Verify execution plan
      val app = createAppFromEventlog(eventLog)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", new PluginTypeChecker(), app)
      }
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val aggExprs = execInfo.filter(_.exec == "HashAggregate")
      assertSizeAndSupported(2, aggExprs, checkDurations = false)
    }
  }

  test("Expressions not supported in SortMergeJoin") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val broadcastConfs = Map("spark.sql.autoBroadcastJoinThreshold" -> "-1")
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric",
        Some(broadcastConfs)) { spark =>
        import spark.implicits._
        val df1 = Seq((1, "ABC", 1.2), (2, "DEF", 2.3), (3, "GHI", 1.4)).toDF(
          "emp_id", "name", "dept")
        val df2 = Seq(("Fin", 1.2, "ABC"), ("IT", 2.3, "DEF")).toDF(
          "dept_name", "dept_id", "name")
        // hex is not supported in GPU yet.
        df1.join(df2, df1("name") === df2("name") &&
          hex(df1("dept")) === hex(df2("dept_id")), "left_outer")
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val smj = allExecInfo.filter(_.exec == "SortMergeJoin")
      assertSizeAndNotSupported(1, smj)
    }
  }

  test("Parse Exec - SortAggregate") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val aggConfs = Map("spark.sql.execution.useObjectHashAggregateExec" -> "false")
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric",
        Some(aggConfs)) { spark =>
        import spark.implicits._
        val df1 = Seq((1, "a"), (1, "aa"), (1, "a"), (2, "b"),
                         (2, "b"), (3, "c"), (3, "c")).toDF("num", "letter")
        df1.groupBy("num").agg(collect_list("letter").as("collected_letters"))
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val sortAggregate = execInfo.filter(_.exec == "SortAggregate")
      assertSizeAndSupported(2, sortAggregate)
    }
  }

  test("Parse Exec - ObjectHashAggregate") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val df1 = Seq((1, "a"), (1, "aa"), (1, "a"), (2, "b"),
          (2, "b"), (3, "c"), (3, "c")).toDF("num", "letter")
        df1.groupBy("num").agg(collect_list("letter").as("collected_letters"))
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val objectHashAggregate = execInfo.filter(_.exec == "ObjectHashAggregate")
      // OHA will get sql time metrics "time in aggregation build"
      // but the UT might fail everytime due to the time difference.
      // Therefore, we should disable the verification of the durations
      assertSizeAndSupported(2, objectHashAggregate, checkDurations = false)
    }
  }

  test("WindowExec and expressions within WindowExec") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val metrics = Seq(
          (0, 0, 0), (1, 0, 1), (2, 5, 2), (3, 0, 3), (4, 0, 1), (5, 5, 3), (6, 5, 0)
        ).toDF("id", "device", "level")
        val rangeWithTwoDevicesById = Window.partitionBy('device).orderBy('id)
        metrics.withColumn("sum", sum('level) over rangeWithTwoDevicesById)
            .withColumn("row_number", row_number.over(rangeWithTwoDevicesById))
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val windowExecs = execInfo.filter(_.exec == "Window")
      assertSizeAndSupported(1, windowExecs)
    }
  }

  test("Parse Pandas execs - AggregateInPandas, ArrowEvalPython, " +
      "FlatMapGroupsInPandas, MapInPandas, WindowInPandas") {
    val eventLog = s"$qualLogDir/pandas_execs_eventlog.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size > 0)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    verifyExecToStageMapping(parsedPlans.toSeq, app)
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    val flatMapGroups = allExecInfo.filter(_.exec == "FlatMapGroupsInPandas")
    assertSizeAndSupported(1, flatMapGroups)
    val aggregateInPandas = allExecInfo.filter(_.exec == "AggregateInPandas")
    assertSizeAndSupported(1, aggregateInPandas)
    // this event log had UDF for ArrowEvalPath so shows up as not supported
    val arrowEvalPython = allExecInfo.filter(_.exec == "ArrowEvalPython")
    // Note that we do not assert speedupFactor == 1, since some speedup is
    // observed for Pandas execs due to accelerated data transfer
    assert(arrowEvalPython.size == 1)
    assert(arrowEvalPython.forall(!_.isSupported))
    val mapInPandas = allExecInfo.filter(_.exec == "MapInPandas")
    assertSizeAndSupported(1, mapInPandas)
    // WindowInPandas configured off by default
    val windowInPandas = allExecInfo.filter(_.exec == "WindowInPandas")
    assertSizeAndNotSupported(1, windowInPandas)
  }

  // GlobalLimit and LocalLimit is not in physical plan when collect is called on the dataframe.
  // We are reading from static eventlogs to test these execs.
  test("Parse execs - LocalLimit and GlobalLimit") {
    val eventLog = s"$qualLogDir/global_local_limit_eventlog.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog)
    assert(app.sqlPlans.size == 1)
    val supportedExecs = Array("GlobalLimit", "LocalLimit")
    app.sqlPlans.foreach { case (sqlID, plan) =>
      val planInfo = SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      verifyPlanExecToStageMap(planInfo)
      // GlobalLimit and LocalLimit are inside WholeStageCodegen. So getting the children of
      // WholeStageCodegenExec
      val wholeStages = planInfo.execInfo.filter(_.exec.contains("WholeStageCodegen"))
      val allChildren = wholeStages.flatMap(_.children).flatten
      for (execName <- supportedExecs) {
        val supportedExec = allChildren.filter(_.exec == execName)
        assertSizeAndSupported(1, supportedExec)
      }
    }
  }

  test("Expression not supported in FilterExec") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "Expressions in FilterExec") { spark =>
        import spark.implicits._
        val df1 = spark.sparkContext.parallelize(List(10, 20, 30, 40)).toDF
        df1.filter(hex($"value") === "A") // hex is not supported in GPU yet.
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      app.sqlPlans.foreach { case (sqlID, plan) =>
        val planInfo = SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        verifyPlanExecToStageMap(planInfo)
        val wholeStages = planInfo.execInfo.filter(_.exec.contains("WholeStageCodegen"))
        assert(wholeStages.size == 1)
        val allChildren = wholeStages.flatMap(_.children).flatten
        val filters = allChildren.filter(_.exec == "Filter")
        assertSizeAndNotSupported(1, filters)
      }
    }
  }

  test("Expression not supported in Expand") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "Expressions in ExpandExec") { spark =>
        import spark.implicits._
        val df = Seq(("foo", 1L, 1.2), ("foo", 2L, 2.2), ("bar", 2L, 3.2),
          ("bar", 2L, 4.2)).toDF("x", "y", "z")
        df.cube($"x", ceil($"y"), hex($"z")).count // hex is not supported in GPU yet.
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      app.sqlPlans.foreach { case (sqlID, plan) =>
        val planInfo = SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        verifyPlanExecToStageMap(planInfo)
        val wholeStages = planInfo.execInfo.filter(_.exec.contains("WholeStageCodegen"))
        assert(wholeStages.size == 2)
        val allChildren = wholeStages.flatMap(_.children).flatten
        val filters = allChildren.filter(_.exec == "Expand")
        assertSizeAndNotSupported(1, filters)
      }
    }
  }

  test("Expression not supported in TakeOrderedAndProject") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "Expressions in TakeOrderedAndProject") { spark =>
        import spark.implicits._
        val df = Seq(("foo", 1L, 1.2), ("foo", 2L, 2.2), ("bar", 2L, 3.2),
          ("bar", 2L, 4.2)).toDF("x", "y", "z")
        df.orderBy(hex($"y"), $"z").limit(2) // hex is not supported in GPU yet.
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val takeOrderedAndProject = execInfo.filter(_.exec == "TakeOrderedAndProject")
      assertSizeAndNotSupported(1, takeOrderedAndProject)
    }
  }

  test("json_tuple is supported in Generate") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
        "Expressions in Generate") { spark =>
        import spark.implicits._
        val jsonString =
          """{"Zipcode":123,"ZipCodeType":"STANDARD",
            |"City":"ABCDE","State":"YZ"}""".stripMargin
        val data = Seq((1, jsonString))
        val df = data.toDF("id", "jsonValues")
        df.select(col("id"), json_tuple(col("jsonValues"), "Zipcode", "ZipCodeType", "City"))
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val generateExprs = execInfo.filter(_.exec == "Generate")
      assertSizeAndSupported(1, generateExprs)
    }
  }

  test("get_json_object is supported by default in Project with RAPIDS 24.10") {
    TrampolineUtil.withTempDir { parquetoutputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "Expressions in Generate") { spark =>
          import spark.implicits._
          val jsonString =
            """{"Zipcode":123,"ZipCodeType":"STANDARD",
              |"City":"ABCDE","State":"YZ"}""".stripMargin
          val data = Seq((1, jsonString))
          val df1 = data.toDF("id", "jValues")
          df1.write.parquet(s"$parquetoutputLoc/parquetfile")
          val df2 = spark.read.parquet(s"$parquetoutputLoc/parquetfile")
          df2.select(col("id"), get_json_object(col("jValues"), "$.ZipCodeType").as("ZipCodeType"))
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        assert(app.sqlPlans.size == 2)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val projectExprs = execInfo.filter(_.exec == "Project")
        assertSizeAndSupported(1, projectExprs)
      }
    }
  }

  test("Expressions supported in SortAggregateExec") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val aggConfs = Map("spark.sql.execution.useObjectHashAggregateExec" -> "false")
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric",
        Some(aggConfs)) { spark =>
        import spark.implicits._
        val df1 = Seq((1, "a"), (1, "aa"), (1, "a"), (2, "b"),
          (2, "b"), (3, "c"), (3, "c")).toDF("num", "letter")
        df1.groupBy("num").agg(collect_list("letter").as("collected_letters"),
          count("letter").as("letter_count"))
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val sortAggregate = execInfo.filter(_.exec == "SortAggregate")
      assertSizeAndSupported(2, sortAggregate)
    }
  }

  test("Expressions supported in SortExec") {
    TrampolineUtil.withTempDir { parquetoutputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "ProjectExprsSupported") { spark =>
          import spark.implicits._
          val df1 = Seq((1.7, "a"), (1.6, "aa"), (1.1, "b"), (2.5, "a"), (2.2, "b"),
            (3.2, "a"), (10.6, "c")).toDF("num", "letter")
          df1.write.parquet(s"$parquetoutputLoc/testsortExec")
          val df2 = spark.read.parquet(s"$parquetoutputLoc/testsortExec")
          df2.sort("num").collect
          df2.orderBy("num").collect
          df2.select(round(col("num")), col("letter")).sort(round(col("num")), col("letter").desc)
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        assert(app.sqlPlans.size == 4)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val sortExec = allExecInfo.filter(_.exec.contains("Sort"))
        assert(sortExec.size == 3)
        assertSizeAndSupported(3, sortExec)
      }
    }
  }

  test("Expressions not supported in ProjectExec") {
    TrampolineUtil.withTempDir { parquetoutputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "ProjectExprsNotSupported") { spark =>
          import spark.implicits._
          val df1 = spark.sparkContext.parallelize(List(10, 20, 30, 40)).toDF
          df1.write.parquet(s"$parquetoutputLoc/testtext")
          val df2 = spark.read.parquet(s"$parquetoutputLoc/testtext")
          df2.select(hex($"value") === "A")
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        assert(app.sqlPlans.size == 2)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "test desc", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        parsedPlans.foreach { pInfo =>
          assert(pInfo.sqlDesc == "test desc")
        }
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val wholeStages = allExecInfo.filter(_.exec.contains("WholeStageCodegen"))
        assert(wholeStages.forall(_.duration.nonEmpty))
        val allChildren = wholeStages.flatMap(_.children).flatten
        val projects = allChildren.filter(_.exec == "Project")
        assertSizeAndNotSupported(1, projects)
      }
    }
  }

  /**
   * Helper function to write a DataFrame to Parquet and then read it back.
   *
   * @param spark The SparkSession instance.
   * @param df    The input DataFrame to be written.
   * @param path  The file path to store the Parquet file.
   * @return A DataFrame read from the Parquet file.
   */
  private def writeAndReadParquet(spark: SparkSession, df: DataFrame, path: String): DataFrame = {
    df.write.parquet(path)
    spark.read.parquet(path)
  }

  /**
   * Table-driven test cases for verifying Spark SQL expressions in ProjectExec.
   */
  val projectExecTestCases: TableFor2[String, (File => (SparkSession => DataFrame))] = Table(
    ("Expression", "FileToSparkSession"),
    // MonthsBetween is supported in ProjectExec
    ("MonthsBetween", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq(("2024-12-01", "2024-01-01"),
                    ("2024-12-01", "2023-12-01"),
                    ("2024-12-01", "2024-12-01")).toDF("date1", "date2")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // months_between should be part of ProjectExec
      df2.select(months_between(df2("date1"), df2("date2")))
    }}),
    // TruncDate is supported in ProjectExec
    ("TruncDate", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq("2024-12-15", "2024-01-10", "2023-11-05").toDF("date")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // trunc should be part of ProjectExec
      df2.select(trunc(df2("date"), "month"))
    }}),
    // TruncTimestamp is supported in ProjectExec
    ("TruncTimestamp", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val data = Seq("2024-12-15 14:30:45",
                      "2024-01-10 08:15:00",
                      "2023-11-05 20:45:30").toDF("timestamp")
      val df1 = data.withColumn("timestamp", to_timestamp(col("timestamp")))
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // date_trunc should be part of ProjectExec
      df2.select(date_trunc("month", df2("timestamp")))
    }}),
    // Ceil is supported in ProjectExec
    ("Ceil", { parquetOutputLoc => { spark =>
      import spark.implicits._
      import org.apache.spark.sql.types.StringType
      val df1 = Seq(9.9, 10.2, 11.6, 12.5).toDF("value")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // ceil should be part of ProjectExec
      df2.select(df2("value").cast(StringType), ceil(df2("value")), df2("value"))
    }}),
    // Translate is supported in ProjectExec
    ("Translate", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq("", "abc", "ABC", "AaBbCc").toDF("value")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // translate should be part of ProjectExec
      df2.select(translate(df2("value"), "ABC", "123"))
    }}),
    // Timestamp functions are supported in ProjectExec
    ("TimestampFunctions", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val init_df = Seq((1230219000123123L, 1230219000123L, 1230219000.123))
      val df1 = init_df.toDF("micro", "millis", "seconds")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // timestamp functions should be part of ProjectExec
      df2.selectExpr("timestamp_micros(micro)", "timestamp_millis(millis)",
        "timestamp_seconds(seconds)")
    }}),
    // Flatten is supported in ProjectExec
    ("Flatten", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq(Seq(Seq(1, 2), Seq(3, 4))).toDF("value")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // flatten should be part of ProjectExec
      df2.select(flatten(df2("value")))
    }}),
    // Xxhash64 is supported in ProjectExec
    ("Xxhash64", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq("spark", "", "abc").toDF("value")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // xxhash64 should be part of ProjectExec
      df2.select(xxhash64(df2("value")))
    }}),
    // MapFromArrays is supported in ProjectExec
    ("MapFromArrays", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq((Array("a", "b", "c"), Array(1, 2, 3)),
        (Array("x", "y", "z"), Array(10, 20, 30))).toDF("keys", "values")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // map_from_arrays should be part of ProjectExec
      df2.select(map_from_arrays(df2("keys"), df2("values")).as("map"))
    }}),
    // ArrayPosition is supported in ProjectExec
    ("ArrayPosition", { parquetOutputLoc => { spark =>
      import spark.implicits._
      val df1 = Seq((1, Array(1, 2, 3), 2)).toDF("id", "array_col", "target")
      val df3 = df1.withColumn("array_col", col("array_col").cast("array<int>"))
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df3, s"$parquetOutputLoc/testtext")
      // array_position should be part of ProjectExec
      df2.select(array_position(col("array_col"), col("target")))
    }}),
    // MapFromEntries is supported in ProjectExec
    ("MapFromEntries", { parquetOutputLoc => { spark =>
      import spark.implicits._
      // Create DataFrame with array of structs (key-value pairs)
      val df1 = Seq(
        Seq(("a", 1), ("b", 2), ("c", 3)),
        Seq(("x", 10), ("y", 20))
      ).toDF("entries")
      // write df1 to parquet to transform LocalTableScan to ProjectExec
      val df2 = writeAndReadParquet(spark, df1, s"$parquetOutputLoc/testtext")
      // map_from_entries should be part of ProjectExec
      df2.select(map_from_entries(df2("entries")).as("map"))
    }})
  )

  /**
   * Tests whether a given Spark SQL expression is supported in ProjectExec.
   *
   * @param appName            Name of the Spark application.
   * @param fileToSparkSession Function that maps a temporary Parquet directory to a function that
   *                           takes a SparkSession and returns a DataFrame.
   * @param parquetOutputLoc   Temporary directory used for writing and reading Parquet files.
   */
  private def testExpressionInProjectExec(appName: String,
                                          fileToSparkSession: File => (SparkSession => DataFrame),
                                          parquetOutputLoc: File): Unit = {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) =
        ToolTestUtils.generateEventLog(eventLogDir, appName)(fileToSparkSession(parquetOutputLoc))
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 2)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val wholeStages = allExecInfo.filter(_.exec.contains("WholeStageCodegen"))
      assert(wholeStages.size == 1)
      assert(wholeStages.forall(_.duration.nonEmpty))
      val allChildren = wholeStages.flatMap(_.children).flatten
      val projects = allChildren.filter(_.exec == "Project")
      assertSizeAndSupported(1, projects)
    }
  }

  forAll(projectExecTestCases) { (exprName, fileToSparkSession) =>
    test(s"$exprName is supported in ProjectExec") {
      TrampolineUtil.withTempDir { parquetOutputLoc =>
        testExpressionInProjectExec(s"{$exprName}SupportedInProjectExec", fileToSparkSession,
          parquetOutputLoc)
      }
    }
  }

  test("ParseUrl is supported") {
    // Verify that each partToExtract expression causes the parse_url to be renamed.
    SQLPlanParser.unsupportedParseURLParts.foreach { part =>
      val partLC = part.toLowerCase
      val a1 = SQLPlanParser.processSpecialFunctions(s"parse_url(test1, $part, cast(test))")
      val a2 = SQLPlanParser.processSpecialFunctions(s"parse_url(test1, $partLC, cast(test))")
      val a3 = SQLPlanParser.processSpecialFunctions(s"parse_url(test1, $part)")
      val a4 = SQLPlanParser.processSpecialFunctions(s"parse_url(test1, $partLC)")
      a1 shouldEqual s"parse_url_$partLC(test1, $part, cast(test))"
      a2 shouldEqual s"parse_url_$partLC(test1, $partLC, cast(test))"
      a3 shouldEqual s"parse_url_$partLC(test1, $part)"
      a4 shouldEqual s"parse_url_$partLC(test1, $partLC)"
    }
    // verify that having the keywords in different argument does not affect the correctness
    val a5 = SQLPlanParser.processSpecialFunctions("parse_url(AUTHORITY, ANY_PART, query)")
    a5 shouldEqual "parse_url(AUTHORITY, ANY_PART, query)"
    // verify multiple calls
    val a6 = SQLPlanParser.processSpecialFunctions(
      "parse_url(AUTHORITY, ANY_PART, query), parse_url(AUTHORITY, REF, query), " +
        "parse_url(AUTHORITY, FILE, query)")
    a6 shouldEqual "parse_url(AUTHORITY, ANY_PART, query), parse_url_ref(AUTHORITY, REF, query), " +
      "parse_url_file(AUTHORITY, FILE, query)"
    // verify nested. Note it does not matter the order as long as we get it right
    val a7 = SQLPlanParser.processSpecialFunctions(
      "parse_url(parse_url(AUTHORITY, ANY_PART, query), REF, query)")
    val a8 = SQLPlanParser.processSpecialFunctions(
      "parse_url(parse_url(AUTHORITY, REF, query), ANY_PART, query)")
    a7 shouldEqual "parse_url_ref(parse_url(AUTHORITY, ANY_PART, query), REF, query)"
    a8 shouldEqual "parse_url_ref(parse_url(AUTHORITY, REF, query), ANY_PART, query)"
    TrampolineUtil.withTempDir { parquetoutputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "ParseURL") { spark =>
          import spark.implicits._
          val df1 = Seq("https://spark.apache.org/downloads.html?query=50",
            "https://docs.nvidia.com/spark-rapids/user-guide/23.12/spark-profiling-tool.html"
          ).toDF("url_col")
          // write df1 to parquet to transform LocalTableScan to ProjectExec
          df1.write.parquet(s"$parquetoutputLoc/testparse")
          val df2 = spark.read.parquet(s"$parquetoutputLoc/testparse")
          df2.selectExpr("*", "parse_url(`url_col`, 'HOST') as HOST",
            "parse_url(`url_col`,'QUERY') as QUERY", "parse_url(`url_col`, 'REF') as REF",
            "parse_url(`url_col`,'USERINFO') as NOT_QUERY")
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)

        assert(app.sqlPlans.size == 2)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val projects = allExecInfo.filter(_.exec.contains("Project"))
        assertSizeAndNotSupported(1, projects)
        val expectedExprss = Seq("parse_url_ref", "parse_url_userinfo")
        projects(0)
          .unsupportedExprs.map(_.getOpName) should contain theSameElementsAs expectedExprss
      }
    }
  }

  test("Parse SQL function Name in HashAggregateExec") {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "sqlmetric") { spark =>
        import spark.implicits._
        val df1 = Seq((1, "a"), (1, "aa"), (1, "a"), (2, "b"),
          (2, "b"), (3, "c"), (3, "c")).toDF("num", "letter")
        // Average is Expression name and `avg` is SQL function name.
        // SQL function name is in the eventlog as shown below.
        // HashAggregate(keys=[letter#187], functions=[avg(cast(num#186 as bigint))])
        df1.groupBy("letter").avg("num")
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      assert(app.sqlPlans.size == 1)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      val execInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val hashAggregate = execInfo.filter(_.exec == "HashAggregate")
      assertSizeAndSupported(2, hashAggregate, checkDurations = false)
    }
  }

  test("Parsing Conditional Expressions") {
    // scalastyle:off line.size.limit
    val expressionsMap: mutable.HashMap[String, Map[String, Int]] = mutable.HashMap(
      "(((lower(partition_act#90) = moduleview) && (isnotnull(productarr#22) && NOT (productarr#22 = []))) || (lower(moduletype#13) = saveforlater))" ->
        Map("lower" -> 2,
          "isnotnull" -> 1,
          "EqualTo" -> 3,
          "And" -> 2,
          "Not" -> 1,
          "Or" -> 1),
      "(IsNotNull(c_customer_id))" ->
        Map("IsNotNull" -> 1),
      "(isnotnull(names#15) AND StartsWith(names#15, OR))" ->
        Map("isnotnull" -> 1,
          "StartsWith" -> 1,
          "And" -> 1),
      "((isnotnull(s_state#68) AND (s_state#68 = TN)) OR (hex(cast(value#0 as bigint)) = B))" ->
        Map("isnotnull" -> 1,
          "And" -> 1,
          "Or" -> 1,
          "hex" -> 1,
          "EqualTo" -> 2),
      // Test that AND followed by '(' without space can be parsed
      "((isnotnull(s_state#68) AND(s_state#68 = TN)) OR (hex(cast(value#0 as bigint)) = B))" ->
        Map("isnotnull" -> 1,
          "And" -> 1,
          "Or" -> 1,
          "hex" -> 1,
          "EqualTo" -> 2),
      "(((isnotnull(d_year#498) AND isnotnull(d_moy#500)) AND (d_year#498 = 1998)) AND (d_moy#500 = 12))" ->
        Map("isnotnull" -> 2,
          "And" -> 3,
          "EqualTo" -> 2),
      "IsNotNull(d_year) AND IsNotNull(d_moy) AND EqualTo(d_year,1998) AND EqualTo(d_moy,12)" ->
        Map("IsNotNull" -> 2,
          "EqualTo" -> 2,
          "And" -> 3),
      // check that a predicate with a single variable name is fine
      "flagVariable" ->
        Map(),
      // check that a predicate with a single function call
      "isnotnull(c_customer_sk#412)" ->
        Map("isnotnull" -> 1),
      "((substr(ca_zip#457, 1, 5) IN (85669,86197,88274,83405,86475,85392,85460,80348,81792) OR ca_state#456 IN (CA,WA,GA)) OR (cs_sales_price#20 > 500.00))" ->
        Map("substr" -> 1,
          "In" -> 2,
          "Or" -> 2,
          "GreaterThan" -> 1),
      // test the operator is at the beginning of expression and not followed by space
      "NOT(isnotnull(d_moy))" ->
        Map("isnotnull" -> 1,
          "Not" -> 1),
      // test the shiftright operator(since spark-4.0)
      "((isnotnull(d_year#498) AND isnotnull(d_moy#500)) AND (d_year#498 >> 1) >= 100)" ->
        Map("isnotnull" -> 2,
          "And" -> 2,
          "ShiftRight" -> 1,
          "GreaterThanOrEqual" -> 1)
    )
    // scalastyle:on line.size.limit
    for ((condExpr, expectedExpressionCounts) <- expressionsMap) {
      val rawExpressions = SQLPlanParser.parseConditionalExpressions(condExpr)
      val expected = expectedExpressionCounts.map(e => ExprOpRef(OpRef.fromExpr(e._1), e._2))
        .toSeq.sortBy(_.getOpName)
      val actualExpressions = ExprOpRef.fromRawExprSeq(rawExpressions)
        .sortBy(_.getOpName)
      actualExpressions should ===(expected)
    }
  }

  runConditionalTest("SaveIntoDataSourceCommand is supported",
    checkDeltaLakeSparkRelease) {
    // This unit test enables deltaLake which generate SaveIntoDataSourceCommand
    TrampolineUtil.withTempDir { outputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val dataWriteCMD = DeltaLakeOps.execSaveIntoDataSourceCMD
        val deltaConfs = Map(
          "spark.sql.extensions" -> "io.delta.sql.DeltaSparkSessionExtension",
          "spark.sql.catalog.spark_catalog" ->
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "DeltaLakeWrites", Some(deltaConfs)) { spark =>
          import spark.implicits._
          val df = spark.sparkContext.makeRDD(1 to 10000, 6).toDF
          df.write.format("delta").mode("overwrite").save(s"$outputLoc/testdelta")
          df
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        // TODO: Spark shows a weird behavior as the entire plan with SQL 65 loses the metrics
        //       and the associated jobs. This causes some execs to be isolated without stage.
        //       The UI won't be able to visualize that job anymore. So, we need to investigate
        //       what happens in that query before we test the execs-to-stage mapping.

        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val deltaLakeWrites = allExecInfo.filter(_.exec.contains(s"$dataWriteCMD"))
        assertSizeAndSupported(1, deltaLakeWrites)
      }
    }
  }

  test("Parse conditional expression does not list ignored expressions") {
    val exprString = "((((((isnotnull(action#191L) AND isnotnull(content#192)) " +
      "AND isnotnull(content_name_16#197L)) AND (action#191L = 0)) AND NOT (content#192 = )) " +
      "AND (content_name_16#197L = 1)) AND NOT (split(split(split(replace(replace(replace" +
      "(replace(trim(replace(cast(unbase64(content#192) as string),  , ), Some( )), *., ), *, ), " +
      "https://, ), http://, ), /, -1)[0], :, -1)[0], \\?, -1)[0] = ))"
    val expected = Map(
      "isnotnull" -> 3,
      "split" -> 3,
      "replace" -> 5,
      "trim" -> 1,
      "unbase64" -> 1,
      "And" -> 6,
      "EqualTo" -> 4, // EqualTo comes from the = operator
      "Not" -> 2).map(e => ExprOpRef(OpRef.fromExpr(e._1), e._2))
    val rawExpressions = SQLPlanParser.parseFilterExpressions(exprString)
    val actualExpressions = ExprOpRef.fromRawExprSeq(rawExpressions)
    actualExpressions should ===(expected)
  }


  test("Parse aggregate expressions") {
    val exprString = "(keys=[], functions=[split(split(split(replace(replace(replace(replace(" +
      "trim(replace(cast(unbase64(content#192) as string),  , ), Some( )), *., ), *, ), " +
      "https://, ), http://, ), /, -1)[0], :, -1)[0], \\?, -1)[0]#199, " +
      "CASE WHEN (instr(replace(cast(unbase64(content#192) as string),  , ), *) = 0) " +
      "THEN concat(replace(cast(unbase64(content#192) as string),  , ), %) " +
      "ELSE replace(replace(replace(cast(unbase64(content#192) as string),  , ), %, " +
      "\\%), *, %) END#200])"
    val expected = Map(
      "replace" -> 10,
      "concat" -> 1,
      "instr" -> 1,
      "split" -> 3,
      "trim" -> 1,
      "unbase64" -> 4).map(e => ExprOpRef(OpRef.fromExpr(e._1), e._2))
    val rawExpressions = SQLPlanParser.parseAggregateExpressions(exprString)
    val actualExpressions = ExprOpRef.fromRawExprSeq(rawExpressions)
    actualExpressions should ===(expected)
  }

  runConditionalTest("promote_precision is supported for Spark LT 3.4.0: issue-517",
    ignoreExprForSparkGTE340) {
    // Spark-3.4.0 removed the promote_precision SQL function
    // the SQL generates the following physical plan
    // (1) Project [CheckOverflow((promote_precision(cast(dec1#24 as decimal(13,2)))
    //     + promote_precision(cast(dec2#25 as decimal(13,2)))), DecimalType(13,2))
    //     AS (dec1 + dec2)#30]
    // For Spark3.4.0, the promote_precision was removed from the plan.
    // (1) Project [(dec1#24 + dec2#25) AS (dec1 + dec2)#30]
    TrampolineUtil.withTempDir { parquetoutputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "projectPromotePrecision") { spark =>
          import spark.implicits._

          import org.apache.spark.sql.types.DecimalType
          val df = Seq(("12347.21", "1234154"), ("92233.08", "1")).toDF
            .withColumn("dec1", col("_1").cast(DecimalType(7, 2)))
            .withColumn("dec2", col("_2").cast(DecimalType(10, 0)))
          // write the df to parquet to transform localTableScan to projectExec
          df.write.parquet(s"$parquetoutputLoc/testPromotePrecision")
          val df2 = spark.read.parquet(s"$parquetoutputLoc/testPromotePrecision")
          df2.selectExpr("dec1+dec2")
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        // The promote_precision should be part of the project exec.
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val projExecs = allExecInfo.filter(_.exec.contains("Project"))
        assertSizeAndSupported(1, projExecs)
      }
    }
  }

  test("current_database is not listed as unsupported: issue-547") {
    // current_database is a scalar value that does not cause any CPU fallbacks
    // we should not list it as unsupported expression if it appears in the SQL.
    TrampolineUtil.withTempDir { parquetoutputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir,
          "ignoreCurrentDatabase") { spark =>
          import spark.implicits._
          val df1 = spark.sparkContext.parallelize(List(10, 20, 30, 40)).toDF
          df1.write.parquet(s"$parquetoutputLoc/ignore_current_database")
          val df2 = spark.read.parquet(s"$parquetoutputLoc/ignore_current_database")
          // Note that the current_database will show up in project only if it is used as a column
          // name. For example:
          // > SELECT current_database(), current_database() as my_db;
          //   +------------------+-------+
          //   |current_database()|  my_db|
          //   +------------------+-------+
          //   |           default|default|
          //   |           default|default|
          //   |           default|default|
          //   |           default|default|
          //   +------------------+-------+
          //   == Physical Plan ==
          //   *(1) Project [default AS current_database()#10, default AS my_db#9]
          //   +- *(1) ColumnarToRow
          //      +- FileScan parquet [] Batched: true, DataFilters: [], Format: Parquet, Location:
          //         InMemoryFileIndex(1 paths)[file:/tmp_folder/T/toolTest..., PartitionFilters: []
          //         PushedFilters: [], ReadSchema: struct<>
          df2.selectExpr("current_database()", "current_database() as my_db")
        }
        val pluginTypeChecker = new PluginTypeChecker()
        val app = createAppFromEventlog(eventLog)
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        // The current_database should be part of the project-exec and the parser should ignore it.
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val projExecs = allExecInfo.filter(_.exec.contains("Project"))
        assertSizeAndSupported(1, projExecs)
      }
    }
  }

  test("ArrayBuffer should be ignored in Expand: issue-554") {
    // ArrayBuffer appears in some SQL queries.
    // It is a non-Spark expression and it should not be considered as SQL-Function
    val exprString = "[" +
      "ArrayBuffer(cast((cast(ds#1241L as double) / 100.0) as bigint), null, null, 0," +
      "            cast(if ((ret_type#1236 = 2)) 1 else 0 as bigint))," +
      "ArrayBuffer(cast((cast(ds#1241L as double) / 100.0) as bigint), wuid#1234, null, 1, null)," +
      "ArrayBuffer(cast((cast(ds#1241L as double) / 100.0) as bigint), null," +
      "            if ((if ((ret_type#1236 = 2)) 1 else 0 = 1)) wuid#1234 else null, 2, null)," +
      "[" +
      "CAST((CAST(supersql_t12.`ds` AS DOUBLE) / 100.0D) AS BIGINT)#1297L," +
      "supersql_t12.`wuid`#1298," +
      "(IF(((IF((supersql_t12.`ret_type` = 2), 1, 0)) = 1), supersql_t12.`wuid`," +
      "CAST(NULL AS STRING)))#1299," +
      "gid#1296," +
      "CAST((IF((supersql_t12.`ret_type` = 2), 1, 0)) AS BIGINT)#1300L]]"
    // Only "IF" should be picked up as a function name
    val expected = Array("IF", "IF", "IF")
    val expressions = SQLPlanParser.parseExpandExpressions(exprString)
    expressions should ===(expected)
  }

  test("SortMergeJoin with arguments should be marked as supported: issue-751") {
    val node = ToolsPlanGraph.constructGraphNode(
      1,
      "SortMergeJoin(skew=true)",
      "SortMergeJoin(skew=true) [trim(field_00#7407, None)], " +
        "[trim(field_01#7415, None)], LeftOuter", Seq[SQLPlanMetric]())
    SortMergeJoinExecParser.accepts(node.name) shouldBe true
    val pluginTypeChecker = new PluginTypeChecker()
    val execInfo = SortMergeJoinExecParser(node, pluginTypeChecker, 2, app = None).parse
    execInfo.isSupported shouldBe true
    execInfo.exec shouldEqual "SortMergeJoin"
  }

  private def testDeltaLakeOperator(
      nodeName: String,
      nodeDescr: String)(f: ExecInfo => Unit) : Unit = {
    val node = ToolsPlanGraph.constructGraphNode(1, nodeName, nodeDescr, Seq[SQLPlanMetric]())
    DataWritingCommandExecParser.isWritingCmdExec(node.name) shouldBe true
    val execInfo = DataWritingCommandExecParser.parseNode(node, new PluginTypeChecker(), 2)
    f(execInfo)
  }

  test("DeltaLake Op AppendDataExecV1 and OverwriteByExpressionExecV1") {
    // test AppendDataExecV1 with supported dataFormat
    val nodeName = "AppendDataExecV1"
    // scalastyle:off line.size.limit
    val nodeDesc =
      s"""|AppendDataExecV1 [num_affected_rows#18560L, num_inserted_rows#18561L], DeltaTableV2(org.apache.spark.sql.SparkSession@5aa5327e,abfss://abfs_path,Some(CatalogTable(
          |Catalog: spark_catalog
          |Database: database
          |Table: tableName
          |Owner: root
          |Created Time: Wed Sep 15 16:47:47 UTC 2021
          |Last Access: UNKNOWN
          |Created By: Spark 3.1.1
          |Type: EXTERNAL
          |Provider: delta
          |Table Properties: [bucketing_version=2, delta.lastCommitTimestamp=1631724453000, delta.lastUpdateVersion=0, delta.minReaderVersion=1, delta.minWriterVersion=2]
          |Location: abfss://abfs_path
          |Serde Library: org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe
          |InputFormat: org.apache.hadoop.mapred.SequenceFileInputFormat
          |OutputFormat: org.apache.hadoop.hive.ql.io.HiveSequenceFileOutputFormat
          |Partition Provider: Catalog
          |Schema: root
          | |-- field_00: string (nullable = true)
          | |-- field_01: string (nullable = true)
          | |-- field_02: string (nullable = true)
          | |-- field_03: string (nullable = true)
          | |-- field_04: string (nullable = true)
          | |-- field_05: string (nullable = true)
          | |-- field_06: string (nullable = true)
          | |-- field_07: string (nullable = true)
          | |-- field_08: string (nullable = true)
          |)),Some(spark_catalog.adl.tableName),None,Map()), Project [from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_00#15200, 20240112 AS field_01#15201, load_func00 AS field_02#15202, completed AS field_03#15203, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_04#15204, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_05#15205, ddsdmsp AS field_06#15206, rename_01 AS field_07#15207, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyyMMdd, Some(Etc/UTC)) AS field_08#15208], org.apache.spark.sql.execution.datasources.v2.DataSourceV2Strategy$$$$Lambda$$12118/1719387317@5a111bfa, com.databricks.sql.transaction.tahoe.catalog.WriteIntoDeltaBuilder$$$$anon$$1@24257336
          |""".stripMargin
    // scalastyle:on line.size.limit
    testDeltaLakeOperator(nodeName, nodeDesc) { execInfo =>
      execInfo.exec shouldEqual nodeName
      execInfo.isSupported shouldBe true
      execInfo.exec shouldEqual "AppendDataExecV1"
      execInfo.expr shouldEqual s"Format: Delta"
      execInfo.udf shouldBe false
      execInfo.shouldIgnore shouldBe false
    }

    // test OverwriteByExpressionExecV1 with supported dataFormat
    // scalastyle:off line.size.limit
    val overwriteNodeName = "OverwriteByExpressionExecV1"
    val overwriteNodeDesc =
      s"""|OverwriteByExpressionExecV1 [num_affected_rows#25429L, num_inserted_rows#25430L], DeltaTableV2(org.apache.spark.sql.SparkSession@5aa5327e,abfss:abfs_path,Some(CatalogTable(
          |Catalog: sparkCatalog
          |Database: dataBaseName
          |Table: tableName
          |Owner: root
          |Created Time: Fri Jun 24 03:44:47 UTC 2022
          |Last Access: UNKNOWN
          |Created By: Spark 3.1.2
          |Type: EXTERNAL
          |Provider: delta
          |Table Properties: [bucketing_version=2, delta.lastCommitTimestamp=1656042273000, delta.lastUpdateVersion=0, delta.minReaderVersion=1, delta.minWriterVersion=2]
          |Location: abfss://abfs_path
          |Serde Library: org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe
          |InputFormat: org.apache.hadoop.mapred.SequenceFileInputFormat
          |OutputFormat: org.apache.hadoop.hive.ql.io.HiveSequenceFileOutputFormat
          |Partition Provider: Catalog
          |Schema: root
          | |-- field_00: timestamp (nullable = true)
          | |-- field_01: string (nullable = true)
          | |-- field_02: string (nullable = true)
          | |-- field_03: string (nullable = true)
          | |-- field_04: string (nullable = true)
          | |-- field_05: string (nullable = true)
          | |-- field_06: string (nullable = true)
          | |-- field_07: string (nullable = true)
          | |-- field_08: string (nullable = true)
          | |-- field_09: string (nullable = true)
          | |-- field_10: string (nullable = true)
          | |-- field_11: string (nullable = true)
          | |-- field_12: string (nullable = true)
          | |-- field_13: string (nullable = true)
          | |-- field_14: string (nullable = true)
          | |-- field_15: string (nullable = true)
          | |-- field_16: string (nullable = true)
          | |-- field_17: string (nullable = true)
          | |-- field_18: string (nullable = true)
          | |-- field_19: string (nullable = true)
          | |-- field_20: string (nullable = true)
          | |-- field_21: string (nullable = true)
          | |-- field_22: string (nullable = true)
          | |-- field_23: string (nullable = true)
          |)),Some(sparkCatalog.dataBaseName.tableName),None,Map()), Project [current_timestamp() AS field_00#24837, rename01 AS field_01#24838, field_02#25049, field_03#25050, field_04#25051, field_05#25052, field_06#25053, field_07#25054, field_08#25055, field_09#25056, field_10#25057, field_11#25058, field_12#25059, field_13#25060, field_14#25061, field_15#25062, field_16#25063, field_17#25064, field_18#25065, field_19#25066, intrn_type#25067, field_21#25068, field_22#25069, field_23#25070, ... 80 more fields], org.apache.spark.sql.execution.datasources.v2.DataSourceV2Strategy$$$$Lambda$$11293/1556987120@3cae5e3a, com.databricks.sql.transaction.tahoe.catalog.WriteIntoDeltaBuilder$$$$anon$$1@6d82eaa
          |""".stripMargin
    // scalastyle:on line.size.limit
    testDeltaLakeOperator(overwriteNodeName, overwriteNodeDesc) { execInfo =>
      execInfo.exec shouldEqual overwriteNodeName
      execInfo.isSupported shouldBe true
      execInfo.exec shouldEqual "OverwriteByExpressionExecV1"
      execInfo.expr shouldEqual s"Format: Delta"
      execInfo.udf shouldBe false
      execInfo.shouldIgnore shouldBe false
    }

    // test that missing pattern won't cause failures to the parser
    val nodeNameForMissingPattern = "AppendDataExecV1"
    // scalastyle:off line.size.limit
    val nodeDescrForMissingPattern =
      s"""|AppendDataExecV1 [num_affected_rows#18560L, num_inserted_rows#18561L], DeltaTableV2(org.apache.spark.sql.SparkSession@5aa5327e,abfss://abfs_path,Some(CatalogTable(
          |)),Some(spark_catalog.adl.tableName),None,Map()), Project [from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_00#15200, 20240112 AS field_01#15201, load_func00 AS field_02#15202, completed AS field_03#15203, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_04#15204, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_05#15205, ddsdmsp AS field_06#15206, rename_01 AS field_07#15207, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyyMMdd, Some(Etc/UTC)) AS field_08#15208], org.apache.spark.sql.execution.datasources.v2.DataSourceV2Strategy$$$$Lambda$$12118/1719387317@5a111bfa, com.databricks.sql.transaction.tahoe.catalog.WriteIntoDeltaBuilder$$$$anon$$1@24257336
          |""".stripMargin
    // scalastyle:on line.size.limit
    testDeltaLakeOperator(nodeNameForMissingPattern, nodeDescrForMissingPattern) { execInfo =>
      execInfo.exec shouldEqual nodeNameForMissingPattern
      execInfo.isSupported shouldBe true
      execInfo.exec shouldEqual "AppendDataExecV1"
      execInfo.expr shouldEqual s"Format: Delta"
      execInfo.udf shouldBe false
      execInfo.shouldIgnore shouldBe false
    }

    // test the case when the operator is not delta lake
    val nodeNameForNonDelta = "AppendDataExecV1"
    // scalastyle:off line.size.limit
    val nodeDescForNonDelta =
      s"""|AppendDataExecV1 [num_affected_rows#18560L, num_inserted_rows#18561L]
          |""".stripMargin
    // scalastyle:on line.size.limit
    testDeltaLakeOperator(nodeNameForNonDelta, nodeDescForNonDelta) { execInfo =>
      execInfo.exec shouldEqual s"$nodeName unknown"
      execInfo.isSupported shouldBe false
      execInfo.expr shouldEqual s"Format: unknown"
      execInfo.udf shouldBe false
      execInfo.shouldIgnore shouldBe false
    }
  }

  test("BloomFilters are supported") {
    // BloomFilter was added in Spark 3.3.0, but we do not care about the version here because
    // we use one parser to rule all spark-versions.
    // scalastyle:off line.size.limit
    // The following two expressions are copied from the Spark explain files
    val aggDescr = "ObjectHashAggregate(keys=[], functions=[partial_bloom_filter_agg(xxhash64(d_week_seq#41, 42), 335, 8990, 0, 0)])"
    val filterDescr = "Filter (((isnotnull(c_customer_sk#1) AND isnotnull(c_current_addr_sk#3)) AND isnotnull(c_current_cdemo_sk#2)) AND might_contain(Subquery scalar-subquery#4, [id=#5], xxhash64(c_current_addr_sk#3, 42)))"
    // scalastyle:on line.size.limit
    val aggNode = ToolsPlanGraph.constructGraphNode(
      1,
      "ObjectHashAggregate",
      aggDescr, Seq[SQLPlanMetric]())
    val aggExprArr =
      SQLPlanParser.parseAggregateExpressions(aggNode.desc.replaceFirst("ObjectHashAggregate", ""))
    val filterNode = ToolsPlanGraph.constructGraphNode(
      2,
      "Filter",
      filterDescr, Seq[SQLPlanMetric]())
    val filterExprArray = SQLPlanParser.parseFilterExpressions(
      filterNode.desc.replaceFirst("Filter ", ""))
    val pluginTypeChecker = new PluginTypeChecker()
    pluginTypeChecker.getNotSupportedExprs(aggExprArr) shouldBe 'empty
    pluginTypeChecker.getNotSupportedExprs(filterExprArray) shouldBe 'empty
  }

  test("KnownNullable is supported") {
    // scalastyle:off line.size.limit
    // The following expression is copied from the Spark explain files
    val projectDescr = "Project [named_struct(start, precisetimestampconversion(precisetimestampconversion(t#0, TimestampType, LongType), LongType, TimestampType), end, knownnullable(precisetimestampconversion(precisetimestampconversion(cast(t#0 + cast(10 minutes as interval) as timestamp), TimestampType, LongType), LongType, TimestampType))) AS session_window#0, d#0, t#0, s#0, x#0L, wt#0]"
    // scalastyle:on line.size.limit
    val projectNode = ToolsPlanGraph.constructGraphNode(
      2,
      "Project",
      projectDescr, Seq[SQLPlanMetric]())
    val filterExprArray = SQLPlanParser.parseFilterExpressions(
      projectNode.desc.replaceFirst("Project ", ""))
    val pluginTypeChecker = new PluginTypeChecker()
    pluginTypeChecker.getNotSupportedExprs(filterExprArray) shouldBe 'empty
  }

  /**
   * Helper method to run tests for WindowGroupLimit expressions
   * @param windowExpr the expression to be evaluated (i.e., rank, row_number).
   * @param skipSqlID the SQL ID to skip the stage verification.
   */
  def runWindowGroupLimitTest(windowExpr: String, skipSqlID: Long): Unit = {
    val windowGroupLimitExecCmd = "WindowGroupLimit"
    val tbl_name = s"foobar_tbl_test_$windowExpr"
    val appName = s"WindowGroupLimitExecTest_$windowExpr"
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, appName) { spark =>
        withTable(spark, tbl_name) {
          spark.sql(s"CREATE TABLE $tbl_name (foo STRING, bar STRING) USING PARQUET")
          val query =
            s"""
            SELECT foo, bar FROM (
                SELECT foo, bar,
                    $windowExpr() OVER (PARTITION BY foo ORDER BY bar) as rank
                FROM $tbl_name)
            WHERE rank <= 2"""
          spark.sql(query)
        }
      }
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      // Note that the generated plan, there are skipped stages that causes some execs to appear
      // without their relevant stages. so we skip the stage verification here.
      verifyExecToStageMapping(parsedPlans.toSeq, app, Some( planInfo =>
        if (planInfo.sqlID == skipSqlID) {
          // Nodes should not have any stages
          val allExecInfos = planInfo.execInfo.flatMap { e =>
            e.children.getOrElse(Seq.empty) :+ e
          }
          // exclude all stages higher than 8 because those ones belong to a skipped stage
          allExecInfos.filter(_.nodeId <= 8).forall(_.stages.nonEmpty)
        })
      )
      val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
      val windowGroupLimitExecs = allExecInfo.filter(_.exec.contains(windowGroupLimitExecCmd))
      // We should have two WindowGroupLimitExec operators (Partial and Final).
      assertSizeAndSupported(2, windowGroupLimitExecs)
    }
  }

  runConditionalTest("WindowGroupLimit expression rank is supported",
    execsSupportedSparkGTE350) {
    runWindowGroupLimitTest("RANK", skipSqlID = 73)
  }

  runConditionalTest("WindowGroupLimit expression row_number is supported",
    execsSupportedSparkGTE350) {
    runWindowGroupLimitTest("ROW_NUMBER", skipSqlID = 76)
  }

  runConditionalTest("CheckOverflowInsert should not exist in Physical Plan",
    execsSupportedSparkGTE331) {
    // This test verifies that the 'CheckOverflowInsert' expression exists in the logical plan
    // but is absent in the physical plan as of Spark 3.5.0. If in future Spark versions,
    // 'CheckOverflowInsert' expression appears in the physical plan, this test will fail,
    // and we should to handle it appropriately.
    val overflowExprStr = "CheckOverflowInTableInsert"
    val tableName = "foobar_tbl"
    var planDf: Option[DataFrame] = None
    TrampolineUtil.withTempDir { eventLogDir =>
      ToolTestUtils.generateEventLog(eventLogDir, overflowExprStr) { spark =>
        withTable(spark, tableName) {
          spark.sql(s"CREATE TABLE $tableName (foo INT) USING PARQUET")
          // Store the Spark Plan in a variable
          planDf = Some(spark.sql(s"INSERT INTO $tableName VALUES (2L)"))
          planDf.get
        }
      }
      // toJSON contains information about all expressions in all plan nodes.
      // We can use it to check if the expression exists in any node.
      val overflowExprInLogicalPlan = planDf.get.queryExecution.optimizedPlan
        .toJSON.contains(overflowExprStr)
      val overflowExprInPhysicalPlan = planDf.get.queryExecution.executedPlan
        .toJSON.contains(overflowExprStr)

      // Assert CheckOverflowInsert exists in Logical Plan but not in Physical Plan
      assert(overflowExprInLogicalPlan,
        "CheckOverflowInTableInsert should exist in the logical plan.")
      assert(!overflowExprInPhysicalPlan,
        "CheckOverflowInTableInsert should not exist in the physical plan.")
    }
  }

  test("MinBy and MaxBy are supported") {
    // Test that aggregates minBy and MAxBy are marked as supported by the qualification tool.
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "minMaxBy") { spark =>
        import spark.implicits._
        val testData = Seq((1, 2, 3), (1, 4, 6), (2, 3, 4)).toDF("a", "b", "c")
        testData.createOrReplaceTempView("t1")
        spark.sql("SELECT a, min_by(b, c), max_by(b, c) FROM t1 GROUP BY a")
      }
      // validate that the eventlog contains MinBy and MaxBy
      val reader = FSUtils.readFileContentAsUTF8(eventLog)
      assert(reader.contains("min_by"))
      assert(reader.contains("max_by"))
      // run the qualification tool
      val pluginTypeChecker = new PluginTypeChecker()
      val app = createAppFromEventlog(eventLog)
      val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
        SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
      }
      verifyExecToStageMapping(parsedPlans.toSeq, app)
      // we should have 2 hash aggregates with min_by and max_by expressions
      // if the min_by and max_by were not recognized, the test would fail
      val hashAggExecs =
        getAllExecsFromPlan(parsedPlans.toSeq).filter(_.exec.equals("HashAggregate"))
      assertSizeAndSupported(2, hashAggExecs, checkDurations = false)
    }
  }

  test("array_join is supported") {
    TrampolineUtil.withTempDir { outputLoc =>
      TrampolineUtil.withTempDir { eventLogDir =>
        val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, "arrayjoin") { spark =>
          import spark.implicits._
          val df = Seq(
            (List("a", "b", "c"), List("b", "c")),
            (List("a", "a"), List("b", "c")),
            (List("aa"), List("b", "c"))
          ).toDF("x", "y")
          df.write.parquet(s"$outputLoc/test_arrayjoin")
          val df2 = spark.read.parquet(s"$outputLoc/test_arrayjoin")
          val df3 = df2.withColumn("arr_join", array_join(col("x"), "."))
          df3
        }
        val app = createAppFromEventlog(eventLog)
        assert(app.sqlPlans.size == 2)
        val pluginTypeChecker = new PluginTypeChecker()
        val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
          SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
        }
        verifyExecToStageMapping(parsedPlans.toSeq, app)
        val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
        val projectExecs = allExecInfo.filter(_.exec == "Project")
        assertSizeAndSupported(1, projectExecs)
      }
    }
  }

  test("Parse MergeRows expressions from physicalPlanDescription") {
    val testCases = Table(
      ("description", "physPlan", "expectedExprs"),

      // Basic: keep only (typical UPDATE/INSERT)
      ("keep only",
        """(10) MergeRows
          |Input [2]: [_c0#478, _c1#479]
          |Arguments: [keep(true, _c0#494)], [keep(true, _c0#478)]
          |
          |(11) Project""".stripMargin,
        Seq("keep")),

      // All three expression types (merge-on-read with DELETE)
      ("all expressions",
        """(10) MergeRows
          |Input [1]: [_c0#478]
          |Arguments: [keep(true, _c0#494)], [discard(cond)], [split(cond, out)]
          |
          |(11) Project""".stripMargin,
        Seq("keep", "discard", "split")),

      // Filters out non-merge functions (isnotnull, cast, concat, etc.)
      ("filter other functions",
        """(10) MergeRows
          |Input [1]: [_c0#478]
          |Arguments: isnotnull(src), [keep(true, _c0)], cast(id), [discard(c)]
          |
          |(11) Project""".stripMargin,
        Seq("keep", "discard")),

      // Multi-line Arguments section (ensures regex captures all lines)
      ("multi-line Arguments",
        """(10) MergeRows
          |Input [4]: [_c0#478, _c1#479]
          |Arguments: isnotnull(__row_from_source#522),
          |           isnotnull(__row_from_target#520),
          |           [keep(true, _c0#494)],
          |           [discard(condition)]
          |
          |(11) Project""".stripMargin,
        Seq("keep", "discard")),

      // Name-based matching works regardless of node ID
      ("mismatched node ID",
        """(99) MergeRows
          |Input [2]: [_c0#478, _c1#479]
          |Arguments: [keep(true, _c0#494)], [split(cond, out)]
          |
          |(100) Project""".stripMargin,
        Seq("keep", "split")),

      // Negative: No MergeRows node
      ("no MergeRows",
        """(10) Project
          |Input [1]: [_c0#523]
          |
          |(11) Filter""".stripMargin,
        Seq.empty),

      // Deduplication (multiple keep expressions return single "keep")
      ("deduplication",
        """(10) MergeRows
          |Input [4]: [_c0#494, _c1#495]
          |Arguments: [keep(true, _c0#494)], [keep(true, _c1#495)], [keep(false, _c0#494)]
          |
          |(11) Project""".stripMargin,
        Seq("keep"))
    )

    forAll(testCases) { (desc, physPlan, expectedExprs) =>
      val actualExprs = SQLPlanParser.parseMergeRowsExpressions(physPlan)
      assert(actualExprs.sorted === expectedExprs.sorted, s"Failed for: $desc")
    }
  }
}
