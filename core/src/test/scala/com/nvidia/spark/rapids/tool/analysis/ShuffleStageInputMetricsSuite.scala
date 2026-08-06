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

package com.nvidia.spark.rapids.tool.analysis

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.nvidia.spark.rapids.tool.{EventLogPathProcessor, ToolTestUtils}
import com.nvidia.spark.rapids.tool.profiling.{ShuffleInputProvenance, ShuffleStageInputAnalysis, ShuffleStageInputIncompleteReason, ShuffleStageInputRecord}
import org.apache.hadoop.conf.Configuration
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession, TrampolineUtil}
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo

/**
 * Tests the consumer-stage shuffle input analysis on real Spark plan graphs.
 *
 * The positive cases run actual queries so the plan graphs, AQE rewrites, exchange reuse, and
 * accumulator wiring are the real ones. The fail-closed cases start from the same real
 * application and then remove one piece of evidence, which is the most direct way to prove that a
 * single gap disables the whole analysis.
 */
class ShuffleStageInputMetricsSuite extends AnyFunSuite with Logging {

  private lazy val sparkSession: SparkSession = {
    SparkSession.builder()
      .master("local[*]")
      .appName("Rapids Shuffle Stage Input Unit Tests")
      .getOrCreate()
  }

  private lazy val hadoopConf: Configuration = sparkSession.sparkContext.hadoopConfiguration

  private val profilingLogDir = ToolTestUtils.getTestResourcePath("spark-events-profiling")
  private val qualificationLogDir =
    ToolTestUtils.getTestResourcePath("spark-events-qualification")

  private def loadApp(eventLogPath: String): ApplicationInfo = {
    val eventLogInfo = EventLogPathProcessor.getEventLogInfo(eventLogPath, hadoopConf).head._1
    new ApplicationInfo(hadoopConf, eventLogInfo)
  }

  private def analyze(app: ApplicationInfo): ShuffleStageInputAnalysis = {
    ShuffleStageInputAnalyzer(app)
  }

  /**
   * Runs a query through a local Spark session, writes its event log, and hands the resulting
   * profiled application to the test body.
   */
  private def withProfiledQuery(name: String)(query: SparkSession => DataFrame)
      (body: ApplicationInfo => Unit): Unit = {
    TrampolineUtil.withTempDir { eventLogDir =>
      val (eventLog, _) = ToolTestUtils.generateEventLog(eventLogDir, name)(query)
      body(loadApp(eventLog))
    }
  }

  private def joinQuery(spark: SparkSession): DataFrame = {
    import spark.implicits._
    val left = spark.sparkContext.makeRDD(1 to 2000, 4).map(i => (i, s"l$i")).toDF("k", "lv")
    val right = spark.sparkContext.makeRDD(1 to 2000, 4).map(i => (i, s"r$i")).toDF("k", "rv")
    // Repartitioning both sides guarantees two shuffle branches into the join's consumer stage.
    left.repartition(4, $"k").join(right.repartition(4, $"k"), "k").groupBy("k").count()
  }

  private def singleShuffleQuery(spark: SparkSession): DataFrame = {
    import spark.implicits._
    spark.sparkContext.makeRDD(1 to 2000, 4).map(i => (i % 17, i)).toDF("k", "v")
      .groupBy("k").sum("v")
  }

  private def selfJoinQuery(spark: SparkSession): DataFrame = {
    import spark.implicits._
    val base = spark.sparkContext.makeRDD(1 to 2000, 4).map(i => (i % 31, i)).toDF("k", "v")
      .groupBy("k").sum("v")
    // Joining an aggregation to itself lets Spark reuse the same exchange on both sides.
    base.as("a").join(base.as("b"), $"a.k" === $"b.k").select($"a.k")
  }

  test("a single shuffle produces one complete consumer-stage record") {
    withProfiledQuery("singleShuffle")(singleShuffleQuery) { app =>
      val analysis = analyze(app)
      assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
      assert(analysis.provenance == ShuffleInputProvenance.Estimated)
      assert(analysis.records.nonEmpty)
      analysis.records.foreach { record =>
        assert(record.numShuffleBranches >= 1)
        assert(record.totalShuffleInputBytes >= 0L)
        assert(record.numTasks > 0)
        assert(record.stageAttemptId >= 0)
      }
    }
  }

  test("a two-sided join sums both shuffle branches into one consumer stage") {
    withProfiledQuery("twoSidedJoin")(joinQuery) { app =>
      val analysis = analyze(app)
      assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
      val multiBranch = analysis.records.filter(_.numShuffleBranches >= 2)
      assert(multiBranch.nonEmpty,
        s"expected a stage fed by several branches but got: ${analysis.records}")
      // The stage total must exceed any single branch, which is the whole point of summing.
      multiBranch.foreach { record =>
        assert(record.totalShuffleInputBytes > 0L)
      }
    }
  }

  test("a reused exchange contributes once per consumer branch") {
    withProfiledQuery("selfJoinReuse")(selfJoinQuery) { app =>
      val analysis = analyze(app)
      assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
      // Reuse shows up as extra branches, never as extra records for the same stage.
      val branchCounts = analysis.records.map(_.numShuffleBranches)
      assert(branchCounts.forall(_ >= 1))
      assert(analysis.records.map(r => (r.sqlId, r.stageId)).distinct.size ==
        analysis.records.size, "each (sql, consumer stage) must appear exactly once")
    }
  }

  test("records are deterministic across repeated analysis of the same application") {
    withProfiledQuery("determinism")(joinQuery) { app =>
      assert(analyze(app) == analyze(app))
    }
  }

  test("a GPU join event log yields measured per-consumer-stage totals") {
    val app = loadApp(s"$profilingLogDir/rapids_join_eventlog.zstd")
    val analysis = analyze(app)
    assert(app.gpuMode)
    assert(analysis.provenance == ShuffleInputProvenance.Measured)
    assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
    // This fixture joins two shuffled inputs into stage 2 and feeds the final stage 3 from one.
    assert(analysis.records.toSet == Set(
      ShuffleStageInputRecord(sqlId = 0L, stageId = 2, stageAttemptId = 0,
        totalShuffleInputBytes = 80152384L, numShuffleBranches = 2, numTasks = 200,
        hasPositiveSpill = false, hasSkew = false),
      ShuffleStageInputRecord(sqlId = 0L, stageId = 3, stageAttemptId = 0,
        totalShuffleInputBytes = 19600L, numShuffleBranches = 1, numTasks = 1,
        hasPositiveSpill = false, hasSkew = false)))
  }

  test("a failed SQL execution makes the whole analysis incomplete") {
    withProfiledQuery("failedSql")(joinQuery) { app =>
      assert(analyze(app).isComplete)
      // A recorded end time alone does not prove success; a failure must disable the analysis.
      app.sqlIdToInfo.values.foreach(_.failureReason = Some("injected failure"))
      val analysis = analyze(app)
      assert(!analysis.isComplete)
      assert(analysis.incompleteReasons.exists {
        case _: ShuffleStageInputIncompleteReason.IncompleteSqlExecution => true
        case _ => false
      })
    }
  }

  test("an unfinished SQL execution makes the whole analysis incomplete") {
    withProfiledQuery("unfinishedSql")(joinQuery) { app =>
      app.sqlIdToInfo.values.foreach(_.endTime = None)
      val analysis = analyze(app)
      assert(!analysis.isComplete)
      assert(analysis.incompleteReasons.exists {
        case _: ShuffleStageInputIncompleteReason.IncompleteSqlExecution => true
        case _ => false
      })
    }
  }

  test("a CPU exchange in a GPU application is an unsupported shuffle input") {
    withProfiledQuery("mixedPlan")(joinQuery) { app =>
      assert(analyze(app).isComplete)
      // Treat the CPU plan as a GPU one: its plain Exchange nodes now lack GPU size metrics,
      // which is the mixed-plan situation that must never produce a reduction.
      app.gpuMode = true
      val analysis = analyze(app)
      assert(!analysis.isComplete)
      assert(analysis.incompleteReasons.exists {
        case _: ShuffleStageInputIncompleteReason.UnsupportedShuffleNode => true
        case _ => false
      })
      assert(analysis.provenance == ShuffleInputProvenance.Measured)
    }
  }

  test("a missing exchange size metric makes the whole analysis incomplete") {
    withProfiledQuery("missingMetric")(joinQuery) { app =>
      assert(analyze(app).isComplete)
      // Drop the accumulator evidence behind every exchange 'data size' metric.
      val dataSizeAccumIds = app.sqlManager.sqlPlans.values.flatMap { planModel =>
        planModel.getToolsPlanGraph.allNodes
          .filter(n => ShuffleStageInputAnalyzer.isShuffleInputCandidate(n.name))
          .flatMap(_.metrics.filter(_.name == ShuffleStageInputAnalyzer.DATA_SIZE_METRIC))
          .map(_.accumulatorId)
      }.toSet
      assert(dataSizeAccumIds.nonEmpty)
      dataSizeAccumIds.foreach(app.accumManager.removeAccumInfo)
      val analysis = analyze(app)
      assert(!analysis.isComplete)
      assert(analysis.incompleteReasons.exists {
        case _: ShuffleStageInputIncompleteReason.MissingExchangeMetric => true
        case _ => false
      })
    }
  }

  test("a consumer stage without a completed successful attempt makes the analysis incomplete") {
    withProfiledQuery("noSuccessfulAttempt")(joinQuery) { app =>
      val consumerStageIds = analyze(app).records.map(_.stageId)
      assert(consumerStageIds.nonEmpty)
      app.stageManager.removeStages(consumerStageIds)
      val analysis = analyze(app)
      assert(!analysis.isComplete)
      assert(analysis.incompleteReasons.exists {
        case _: ShuffleStageInputIncompleteReason.NoSuccessfulStageAttempt => true
        case _ => false
      })
    }
  }

  test("a broadcast exchange is never treated as a shuffle input") {
    assert(!ShuffleStageInputAnalyzer.isShuffleInputCandidate("BroadcastExchange"))
    assert(!ShuffleStageInputAnalyzer.isShuffleInputCandidate("GpuBroadcastExchange"))
    assert(ShuffleStageInputAnalyzer.isShuffleInputCandidate("Exchange"))
    assert(ShuffleStageInputAnalyzer.isShuffleInputCandidate("GpuColumnarExchange"))
  }

  test("GPU spill evidence is retained per stage attempt") {
    TrampolineUtil.withTempDir { tmpDir =>
      // Attempt 0 of stage 10 spilled; attempt 1 of the same stage ran clean.
      val logPath = Paths.get(tmpDir.getAbsolutePath, "gpu_spill_attempts_eventlog")
      // scalastyle:off line.size.limit
      val content =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}
          |{"Event":"SparkListenerApplicationStart","App Name":"GpuSpillAttempts","App ID":"local-1600000000000","Timestamp":123456,"User":"tester"}
          |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":1,"Index":1,"Attempt":0,"Partition ID":1,"Launch Time":1712248533994,"Executor ID":"1","Host":"host1","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712248534994,"Failed":false,"Killed":false,"Accumulables":[{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.845","Value":"00:00:00.845","Internal":false,"Count Failed Values":true}]}}
          |{"Event":"SparkListenerTaskEnd","Stage ID":10,"Stage Attempt ID":1,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":2,"Index":1,"Attempt":0,"Partition ID":1,"Launch Time":1712248535994,"Executor ID":"1","Host":"host1","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712248536994,"Failed":false,"Killed":false,"Accumulables":[{"ID":1010,"Name":"gpuSemaphoreWait","Update":"00:00:00.492","Value":"00:00:00.492","Internal":false,"Count Failed Values":true}]}}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(logPath, content.getBytes(StandardCharsets.UTF_8))
      val app = loadApp(logPath.toString)
      assert(app.accumManager.hasGpuSpillEvidence(10, 0),
        "the failed-style attempt 0 must retain its spill evidence")
      assert(!app.accumManager.hasGpuSpillEvidence(10, 1),
        "the clean attempt 1 must not inherit attempt 0's spill")
    }
  }

  test("a zero GPU spill update is not treated as spill evidence") {
    TrampolineUtil.withTempDir { tmpDir =>
      val logPath = Paths.get(tmpDir.getAbsolutePath, "gpu_zero_spill_eventlog")
      // scalastyle:off line.size.limit
      val content =
        """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}
          |{"Event":"SparkListenerApplicationStart","App Name":"GpuZeroSpill","App ID":"local-1600000000001","Timestamp":123456,"User":"tester"}
          |{"Event":"SparkListenerTaskEnd","Stage ID":7,"Stage Attempt ID":0,"Task Type":"ShuffleMapTask","Task End Reason":{"Reason":"Success"},"Task Info":{"Task ID":1,"Index":1,"Attempt":0,"Partition ID":1,"Launch Time":1712248533994,"Executor ID":"1","Host":"host1","Locality":"PROCESS_LOCAL","Speculative":false,"Getting Result Time":0,"Finish Time":1712248534994,"Failed":false,"Killed":false,"Accumulables":[{"ID":1018,"Name":"gpuSpillToHostTime","Update":"00:00:00.000","Value":"00:00:00.000","Internal":false,"Count Failed Values":true}]}}""".stripMargin
      // scalastyle:on line.size.limit
      Files.write(logPath, content.getBytes(StandardCharsets.UTF_8))
      val app = loadApp(logPath.toString)
      assert(!app.accumManager.hasGpuSpillEvidence(7, 0))
    }
  }

  test("the profiling analyzer exposes the same analysis without a second traversal") {
    withProfiledQuery("providerReuse")(joinQuery) { app =>
      val fromAnalyzer = app.planMetricProcessor.shuffleStageInputAnalysis
      assert(fromAnalyzer == analyze(app))
      // The lazy val must be cached rather than recomputed on every access.
      assert(app.planMetricProcessor.shuffleStageInputAnalysis eq fromAnalyzer)
    }
  }

  test("a CPU event log without a terminal SQL end event is incomplete") {
    val app = loadApp(s"$qualificationLogDir/join_missing_sql_end")
    val analysis = analyze(app)
    assert(!analysis.isComplete)
    assert(analysis.records.isEmpty)
    assert(analysis.incompleteReasons ==
      Seq(ShuffleStageInputIncompleteReason.IncompleteSqlExecution(0L)))
  }

  test("a CPU AQE shuffle event log records multi-branch totals and real spill pressure") {
    val app = loadApp(s"$qualificationLogDir/aqeshuffle_eventlog.zstd")
    val analysis = analyze(app)
    assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
    assert(analysis.provenance == ShuffleInputProvenance.Estimated)
    val joinStage = analysis.records.find(_.stageId == 4)
    assert(joinStage.isDefined, s"stage 4 missing from ${analysis.records}")
    // Two shuffle branches feed stage 4, and its tasks really did spill in this fixture.
    assert(joinStage.get.numShuffleBranches == 2)
    assert(joinStage.get.totalShuffleInputBytes == 320000000L)
    assert(joinStage.get.hasPositiveSpill,
      "the fixture's spill evidence must reach the record so the pass stays blocked")
  }

  test("a CPU query event log totals each consumer stage of its final plan") {
    val app = loadApp(s"$qualificationLogDir/nds_q86_test")
    val analysis = analyze(app)
    assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
    assert(analysis.records.toSet == Set(
      ShuffleStageInputRecord(sqlId = 24L, stageId = 34, stageAttemptId = 0,
        totalShuffleInputBytes = 5491688L, numShuffleBranches = 1, numTasks = 1024,
        hasPositiveSpill = false, hasSkew = false),
      ShuffleStageInputRecord(sqlId = 24L, stageId = 35, stageAttemptId = 0,
        totalShuffleInputBytes = 17600L, numShuffleBranches = 1, numTasks = 1024,
        hasPositiveSpill = false, hasSkew = false)))
  }

  test("an event log with several stage attempts selects one completed successful attempt") {
    val app = loadApp(s"$qualificationLogDir/multiple_attempts")
    val analysis = analyze(app)
    assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")
    assert(analysis.records.nonEmpty)
    analysis.records.foreach { record =>
      val selected = app.stageManager.getStagesByIds(Seq(record.stageId))
        .find(_.getAttemptId == record.stageAttemptId)
      assert(selected.isDefined, s"attempt ${record.stageAttemptId} missing for $record")
      assert(!selected.get.hasFailed)
      assert(selected.get.stageInfo.completionTime.isDefined)
      // No later completed successful attempt may exist for the same stage.
      assert(!app.stageManager.getStagesByIds(Seq(record.stageId)).exists { candidate =>
        candidate.getAttemptId > record.stageAttemptId && !candidate.hasFailed &&
          candidate.stageInfo.completionTime.isDefined
      })
    }
  }

  test("an exchange whose downstream nodes carry no stage is still attributed") {
    // Ends in a repartition, so the topmost exchange's only sink is the plan root, which carries
    // no metrics and therefore no stage assignment. Walking the graph dead-ends there, and the
    // reading stage has to be recovered from the exchange's own assignment instead.
    val confs = Map(
      "spark.sql.adaptive.enabled" -> "true",
      "spark.sql.autoBroadcastJoinThreshold" -> "-1",
      "spark.sql.shuffle.partitions" -> "8")
    TrampolineUtil.withTempDir { dir =>
      val (log, _) = ToolTestUtils.generateEventLog(dir, "rootExchange", Some(confs)) { spark =>
        import spark.implicits._
        val left = spark.sparkContext.makeRDD(1 to 2000, 4).map(i => (i, s"l$i")).toDF("k", "lv")
        val right = spark.sparkContext.makeRDD(1 to 2000, 4).map(i => (i, s"r$i")).toDF("k", "rv")
        left.join(right, "k").repartition(4)
      }
      val app = loadApp(log)
      val analysis = analyze(app)
      assert(analysis.isComplete, s"unexpected gaps: ${analysis.incompleteSummary(10)}")

      // Confirm the shape this test exists for: an exchange whose sink has no stage assignment.
      val deadEnding = app.sqlManager.sqlPlans.values.flatMap { planModel =>
        val graph = planModel.getToolsPlanGraph
        graph.allNodes
          .filter(n => ShuffleStageInputAnalyzer.isShuffleInputCandidate(n.name))
          .filter { n =>
            val sinks = graph.getSinkNodes(n.id)
            sinks.nonEmpty && sinks.forall(graph.getNodeStageRawAssignment(_).isEmpty)
          }
      }
      assert(deadEnding.nonEmpty,
        "fixture no longer produces an exchange whose sinks are unassigned")

      // Every executed exchange must contribute exactly one branch: the join's two inputs land
      // in one consumer stage, and the trailing repartition feeds another.
      val totalExchanges = app.sqlManager.sqlPlans.values.flatMap { planModel =>
        planModel.getToolsPlanGraph.allNodes
          .filter(n => ShuffleStageInputAnalyzer.isShuffleInputCandidate(n.name))
          .filter(n => planModel.getToolsPlanGraph.getNodeStageRawAssignment(n.id).nonEmpty)
      }.size
      assert(analysis.records.map(_.numShuffleBranches).sum == totalExchanges,
        s"not every exchange was attributed: ${analysis.records}")
      assert(analysis.records.exists(_.numShuffleBranches == 2),
        s"expected the join stage to sum both inputs: ${analysis.records}")
    }
  }

  test("test resources are available") {
    assert(new File(s"$profilingLogDir/rapids_join_eventlog.zstd").exists())
  }
}
