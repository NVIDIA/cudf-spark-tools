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

import scala.collection.mutable
import scala.util.Try

import com.nvidia.spark.rapids.tool.profiling.{ShuffleInputProvenance, ShuffleStageInputAnalysis, ShuffleStageInputIncompleteReason, ShuffleStageInputRecord}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.plangraph.{SparkPlanGraphNode, ToolsPlanGraph}
import org.apache.spark.sql.rapids.tool.store.{StageModel, TaskModel}

/**
 * Builds the raw consumer-stage shuffle input inventory used by the downward shuffle-partition
 * pass.
 *
 * Sizing shuffle exchanges in isolation is not enough, because a join stage can consume several
 * shuffled inputs at once. This analyzer therefore attributes each exchange's uncompressed
 * `data size` to every downstream branch that consumes it and totals those branches per consumer
 * stage. An exchange reused through two branches into the same stage contributes twice, because
 * each branch processes the data.
 *
 * Lowering a global partition count is more dangerous than overestimating it, so the analysis is
 * all-or-nothing: an unsupported shuffle node, a missing size metric, an ambiguous consumer
 * mapping, a SQL execution that did not terminate successfully, or a consumer stage without a
 * completed successful attempt marks the whole application incomplete rather than silently
 * dropping the affected stage.
 */
class ShuffleStageInputAnalyzer(app: AppBase) extends Logging {

  import ShuffleStageInputAnalyzer._

  /** Accumulated bytes and branch count for one (SQL execution, consumer stage). */
  private case class BranchTotals(bytes: Long, branches: Int) {
    // Saturating rather than wrapping: a total that overflowed to a small positive value would
    // understate the requirement, which is the one direction this pass must never fail in.
    def add(moreBytes: Long): BranchTotals = {
      val sum = bytes + moreBytes
      val saturated = if (sum < bytes) Long.MaxValue else sum
      BranchTotals(saturated, branches + 1)
    }
  }

  private val provenance: ShuffleInputProvenance = {
    if (app.gpuMode) ShuffleInputProvenance.Measured else ShuffleInputProvenance.Estimated
  }

  /**
   * Shuffle exchange node names this analyzer can size for the current application.
   *
   * A GPU application must expose GPU columnar exchanges; a plain CPU `Exchange` that executed in
   * a GPU plan means the GPU metrics do not cover all the shuffled data, which is exactly the
   * situation that makes a downward decision unsafe.
   *
   * TODO: a CPU exchange inside a GPU run could in principle be sized from its CPU data size, the
   * way qualification already sizes an all-CPU run. It is not handled today because the input-size
   * factor that converts CPU bytes to estimated GPU bytes is application-level, so a single run
   * cannot apply one factor to its CPU exchanges and another to its GPU ones.
   */
  private def isSupportedShuffleExchange(nodeName: String): Boolean = {
    if (app.gpuMode) {
      nodeName.contains(COLUMNAR_EXCHANGE_NAME)
    } else {
      nodeName == EXCHANGE_NAME
    }
  }

  def build(): ShuffleStageInputAnalysis = {
    val records = mutable.ArrayBuffer.empty[ShuffleStageInputRecord]
    val reasons = mutable.ArrayBuffer.empty[ShuffleStageInputIncompleteReason]

    app.sqlManager.sqlPlans.values.foreach { planModel =>
      Try(planModel.getToolsPlanGraph).toOption match {
        case Some(graph) => analyzeSql(planModel.id, graph, records, reasons)
        case None =>
          // No final plan graph means there is nothing trustworthy to size.
          reasons += ShuffleStageInputIncompleteReason.IncompleteSqlExecution(planModel.id)
      }
    }
    ShuffleStageInputAnalysis(records.toSeq, reasons.toSeq, provenance,
      appHasFailedStage = app.stageManager.getFailedStages.nonEmpty)
  }

  /**
   * Edge index for one plan graph.
   *
   * `ToolsPlanGraph.getSinkNodes` rescans every edge per call, so the walk builds both directions
   * once. The incoming direction is what identifies an exchange's producing stage without relying
   * on stage-number ordering.
   */
  private class EdgeIndex(graph: ToolsPlanGraph) {
    private val outgoing: Map[Long, Seq[Long]] =
      graph.edges.groupBy(_.fromId).map { case (from, es) => from -> es.map(_.toId).toSeq }.toMap
    private val incoming: Map[Long, Seq[Long]] =
      graph.edges.groupBy(_.toId).map { case (to, es) => to -> es.map(_.fromId).toSeq }.toMap

    /** Downstream consumer branches of a node. A reused exchange has one entry per reuse site. */
    def sinksOf(nodeId: Long): Seq[Long] = outgoing.getOrElse(nodeId, Seq.empty)
    def sourcesOf(nodeId: Long): Seq[Long] = incoming.getOrElse(nodeId, Seq.empty)
  }

  /**
   * Stages that write into an exchange, taken from the stage assignment of its child nodes.
   *
   * The exchange's own assignment spans both the writing and the reading side, so it cannot
   * distinguish them. Walking one edge upstream identifies the producing side from the graph
   * itself rather than from stage-number ordering.
   */
  private def producerStagesOf(
      graph: ToolsPlanGraph, edgeIndex: EdgeIndex, nodeId: Long): Set[Int] = {
    edgeIndex.sourcesOf(nodeId).flatMap(graph.getNodeStageLogicalAssignment).toSet
  }

  private def analyzeSql(
      sqlId: Long,
      graph: ToolsPlanGraph,
      records: mutable.ArrayBuffer[ShuffleStageInputRecord],
      reasons: mutable.ArrayBuffer[ShuffleStageInputIncompleteReason]): Unit = {
    // A node that was never assigned to a stage did not execute (for example a branch that AQE
    // replaced), so it carries no shuffle data and is not evidence of missing coverage.
    val executedShuffleNodes = graph.allNodes.filter { node =>
      isShuffleInputCandidate(node.name) && graph.getNodeStageRawAssignment(node.id).nonEmpty
    }
    if (executedShuffleNodes.isEmpty) {
      return
    }
    if (!app.sqlIdToInfo.get(sqlId).exists(_.isTerminalSuccess)) {
      reasons += ShuffleStageInputIncompleteReason.IncompleteSqlExecution(sqlId)
      return
    }

    val sqlStageIds = stageIdsOfSql(sqlId)
    val edgeIndex = new EdgeIndex(graph)
    val branchTotals = mutable.LinkedHashMap.empty[Int, BranchTotals]

    executedShuffleNodes.foreach { node =>
      val producerStages = producerStagesOf(graph, edgeIndex, node.id)
      if (!isSupportedShuffleExchange(node.name)) {
        reasons += ShuffleStageInputIncompleteReason.UnsupportedShuffleNode(
          sqlId, node.id, node.name)
      } else if (producerStages.isEmpty) {
        // Without a producing side there is no way to tell which end of the exchange a
        // downstream stage sits on, so the branch cannot be attributed safely.
        reasons += ShuffleStageInputIncompleteReason.UnresolvedConsumerStage(
          sqlId, node.id, node.name)
      } else {
        // Resolve the exchange metric once, before attributing it to any branch, so that a
        // reused exchange cannot be counted from a different stage's accumulator rows.
        resolveExchangeDataSize(node, producerStages) match {
          case None =>
            reasons += ShuffleStageInputIncompleteReason.MissingExchangeMetric(
              sqlId, node.id, node.name)
          case Some(dataSize) =>
            attributeBranches(sqlId, graph, edgeIndex, node, producerStages, sqlStageIds,
              dataSize, branchTotals, reasons)
        }
      }
    }

    branchTotals.foreach { case (stageId, totals) =>
      selectStageAttempt(stageId) match {
        case None =>
          reasons += ShuffleStageInputIncompleteReason.NoSuccessfulStageAttempt(sqlId, stageId)
        case Some(stageModel) =>
          records += buildRecord(sqlId, stageId, stageModel, totals)
      }
    }
  }

  /**
   * Attributes an exchange's resolved bytes to every distinct downstream branch.
   *
   * A reused exchange is represented as extra outgoing edges from the single original node, so the
   * branch count here is what preserves reuse multiplicity, including two branches that land in
   * the same consumer stage.
   */
  private def attributeBranches(
      sqlId: Long,
      graph: ToolsPlanGraph,
      edgeIndex: EdgeIndex,
      node: SparkPlanGraphNode,
      producerStages: Set[Int],
      sqlStageIds: Set[Int],
      dataSize: Long,
      branchTotals: mutable.LinkedHashMap[Int, BranchTotals],
      reasons: mutable.ArrayBuffer[ShuffleStageInputIncompleteReason]): Unit = {
    val branches = edgeIndex.sinksOf(node.id)
    val walkedStages = if (branches.isEmpty) {
      // The topmost exchange of a plan has no downstream node at all.
      Seq(Set.empty[Int])
    } else {
      branches.map(resolveConsumerStages(graph, edgeIndex, _, producerStages, sqlStageIds))
    }
    // A branch that reaches no assigned stage still has a reading stage: near the root of a plan
    // the downstream nodes often carry no metrics, so the walk dead-ends. That stage is
    // recoverable from the exchange's own assignment, which spans both sides of the shuffle,
    // once the writing side is removed.
    val fallbackStages = consumerStagesFromOwnAssignment(graph, node.id, producerStages,
      sqlStageIds)
    walkedStages.foreach { walked =>
      val consumerStages = if (walked.nonEmpty) walked else fallbackStages
      if (consumerStages.isEmpty) {
        reasons += ShuffleStageInputIncompleteReason.UnresolvedConsumerStage(
          sqlId, node.id, node.name)
      } else {
        consumerStages.foreach { consumerStageId =>
          val current = branchTotals.getOrElse(consumerStageId, BranchTotals(0L, 0))
          branchTotals(consumerStageId) = current.add(dataSize)
        }
      }
    }
  }

  /**
   * Consumer stages derived from the exchange's own stage assignment.
   *
   * A shuffle exchange is assigned to both the stage that writes it (from its write metrics) and
   * the stage that reads it (from its read metrics), so removing the producing side leaves the
   * reading side. This is the fallback for branches whose graph walk reaches no assigned node.
   */
  private def consumerStagesFromOwnAssignment(
      graph: ToolsPlanGraph,
      nodeId: Long,
      producerStages: Set[Int],
      sqlStageIds: Set[Int]): Set[Int] = {
    graph.getNodeStageRawAssignment(nodeId).filter { stageId =>
      !producerStages.contains(stageId) && sqlStageIds.contains(stageId)
    }
  }

  /**
   * Walks one outgoing branch to the first downstream node assigned to a stage other than the
   * exchange's producing stage.
   *
   * The walk follows graph edges rather than comparing stage numbers, because stage ids are not
   * ordered by data flow. A candidate stage must also be confirmed by the node's own raw
   * assignment and must belong to this SQL execution's jobs.
   *
   * A branch can legitimately reach several stages at once: when AQE splits a skewed join, the
   * one consuming operator is assigned to every split stage. The full exchange size is then
   * attributed to each of them. That deliberately overstates each split stage, because AQE divided
   * the data between them, and overstating can only raise the partition requirement and make a
   * reduction less likely. Understating it is the outcome that would be unsafe.
   *
   * TODO: the split could be estimated instead of duplicated, for example by apportioning the
   * exchange size across the split stages by their per-stage shuffle read metrics. That would
   * tighten the requirement on skewed joins, where duplication is at its most conservative.
   *
   * @return the consumer stages of this branch, or an empty set when none could be resolved
   */
  private def resolveConsumerStages(
      graph: ToolsPlanGraph,
      edgeIndex: EdgeIndex,
      sinkId: Long,
      producerStages: Set[Int],
      sqlStageIds: Set[Int]): Set[Int] = {
    val visited = mutable.HashSet.empty[Long]
    var frontier = List(sinkId)
    var depth = 0
    while (frontier.nonEmpty && depth < MAX_CONSUMER_WALK_DEPTH) {
      val candidates = frontier.flatMap { nodeId =>
        val raw = graph.getNodeStageRawAssignment(nodeId)
        graph.getNodeStageLogicalAssignment(nodeId).filter { stageId =>
          !producerStages.contains(stageId) && sqlStageIds.contains(stageId) &&
            raw.contains(stageId)
        }
      }.toSet
      if (candidates.nonEmpty) {
        return candidates
      }
      visited ++= frontier
      frontier = frontier.flatMap(edgeIndex.sinksOf).distinct.filterNot(visited.contains)
      depth += 1
    }
    Set.empty
  }

  /**
   * Reads the uncompressed `data size` of a single exchange node.
   *
   * Stage-scoped accumulator evidence is preferred so that a reused exchange is not inflated by
   * rows belonging to another stage. A present zero is a real measurement and is returned as
   * such; only the complete absence of accumulator evidence returns None.
   */
  private def resolveExchangeDataSize(
      node: SparkPlanGraphNode, producerStages: Set[Int]): Option[Long] = {
    node.metrics.find(_.name == DATA_SIZE_METRIC).flatMap { metric =>
      val accumInfoOpt = app.accumManager.accumInfoMap.get(metric.accumulatorId)
      val fromStages = accumInfoOpt.flatMap { accumInfo =>
        val stageValues = producerStages.toSeq.sorted.flatMap(accumInfo.getTotalForStage)
        // Fall back to the accumulator's own maximum when the producing stage cannot be pinned
        // down, which keeps the existing driver/task maximum semantics.
        stageValues.reduceOption(_ max _).orElse(accumInfo.getMaxTotalAcrossStages)
      }
      // Local-mode plans report exchange sizes through driver accumulator updates instead.
      val fromDriver = app.driverAccumMap.get(metric.accumulatorId)
        .flatMap(_.map(_.value).reduceOption(_ max _))
      Seq(fromStages, fromDriver).flatten.reduceOption(_ max _)
    }
  }

  /**
   * Picks the attempt whose evidence represents the consumer stage: the highest completed attempt
   * that did not fail. Earlier failed attempts are ignored, so a stage that failed once and then
   * succeeded cleanly is still eligible.
   */
  private def selectStageAttempt(stageId: Int): Option[StageModel] = {
    app.stageManager.getStagesByIds(Seq(stageId))
      .filter(sm => !sm.hasFailed && sm.stageInfo.completionTime.isDefined)
      .reduceOption((left, right) => if (right.getAttemptId > left.getAttemptId) right else left)
  }

  private def buildRecord(
      sqlId: Long,
      stageId: Int,
      stageModel: StageModel,
      totals: BranchTotals): ShuffleStageInputRecord = {
    val attemptId = stageModel.getAttemptId
    val tasks = app.taskManager.getTasks(stageId, attemptId, Some(countsTowardTotals)).toSeq
    val spillCandidates =
      app.taskManager.getTasks(stageId, attemptId, Some(countsTowardSpillGate)).toSeq
    val hasTaskSpill = spillCandidates.exists(spilled)
    val hasSpill = hasTaskSpill || app.accumManager.hasGpuSpillEvidence(stageId, attemptId)
    ShuffleStageInputRecord(
      sqlId = sqlId,
      stageId = stageId,
      stageAttemptId = attemptId,
      totalShuffleInputBytes = totals.bytes,
      numShuffleBranches = totals.branches,
      numTasks = stageModel.stageInfo.numTasks,
      hasPositiveSpill = hasSpill,
      hasSkew = hasShuffleReadSkew(tasks.map(_.sr_totalBytesRead)))
  }

  /**
   * Matches the existing shuffle-skew heuristic: a task reading more than three times the attempt
   * average and more than 100 MB.
   */
  private def hasShuffleReadSkew(shuffleReadBytes: Seq[Long]): Boolean = {
    if (shuffleReadBytes.isEmpty) {
      false
    } else {
      val average = shuffleReadBytes.sum.toDouble / shuffleReadBytes.size
      shuffleReadBytes.exists(bytes => bytes > 3 * average && bytes > SKEW_MIN_READ_BYTES)
    }
  }

  /** Stage ids reachable from the jobs that belong to this SQL execution. */
  private def stageIdsOfSql(sqlId: Long): Set[Int] = {
    app.jobIdToInfo.values.collect {
      case job if job.sqlID.contains(sqlId) => job.stageIds
    }.flatten.toSet
  }
}

object ShuffleStageInputAnalyzer {
  /** Uncompressed shuffle bytes written by an exchange, as reported by Spark and RAPIDS. */
  val DATA_SIZE_METRIC = "data size"
  private val EXCHANGE_NAME = "Exchange"
  private val COLUMNAR_EXCHANGE_NAME = "ColumnarExchange"
  private val BROADCAST_MARKER = "Broadcast"
  /** Bound on how far a branch walk looks for a consumer stage before failing closed. */
  private val MAX_CONSUMER_WALK_DEPTH = 16
  /** Matches AppSparkMetricsAnalyzer.shuffleSkewCheck. */
  private val SKEW_MIN_READ_BYTES = 100L * 1024L * 1024L

  /**
   * Tasks whose metrics describe the work the recommendation governs. Speculative duplicates and
   * failed attempts did not produce the stage's output, so their bytes must not inflate the
   * requirement the candidate is sized from.
   */
  private[analysis] def countsTowardTotals(task: TaskModel): Boolean = {
    task.successful && !task.speculative
  }

  /**
   * Tasks whose spill evidence blocks a reduction. Deliberately broader than
   * [[countsTowardTotals]]: a task that spilled and then failed is exactly the memory pressure
   * this pass must not reduce into, so excluding it would hide the signal the gate exists to see.
   * Speculative duplicates are still excluded, since they re-run work already counted elsewhere.
   */
  private[analysis] def countsTowardSpillGate(task: TaskModel): Boolean = !task.speculative

  private[analysis] def spilled(task: TaskModel): Boolean = {
    task.memoryBytesSpilled > 0L || task.diskBytesSpilled > 0L
  }

  /**
   * Any exchange that shuffles data. Broadcast exchanges are excluded because they replicate a
   * small side rather than partitioning it, so they do not drive the partition count.
   */
  def isShuffleInputCandidate(nodeName: String): Boolean = {
    nodeName.contains(EXCHANGE_NAME) && !nodeName.contains(BROADCAST_MARKER)
  }

  def apply(app: AppBase): ShuffleStageInputAnalysis = {
    new ShuffleStageInputAnalyzer(app).build()
  }
}
