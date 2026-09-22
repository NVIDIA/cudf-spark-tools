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

package org.apache.spark.sql.rapids.tool.plangraph

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.execution.metric.SQLMetricInfo
import org.apache.spark.sql.rapids.tool.util.stubs.{EnclosingClusterPolicy, SparkPlanInfo}
import org.apache.spark.sql.rapids.tool.util.stubs.EnclosingClusterPolicy.{Break, Inherit}

class ToolsPlanGraphSuite extends AnyFunSuite {

  private class TestSparkPlanInfo(
      name: String,
      planChildren: Seq[SparkPlanInfo],
      planMetrics: Seq[SQLMetricInfo],
      override val enclosingClusterPolicy: EnclosingClusterPolicy)
    extends SparkPlanInfo(name, name, planChildren, Map.empty, planMetrics)

  private case class NodeView(
      name: String,
      desc: String,
      metrics: Seq[(String, Long, String)])

  private case class GraphView(
      topLevelIds: Seq[Long],
      nodes: Seq[(Long, NodeView)],
      clusters: Seq[(Long, Seq[Long])],
      edges: Seq[(Long, Long)])

  private def plan(
      name: String,
      children: Seq[SparkPlanInfo] = Seq.empty,
      metrics: Seq[SQLMetricInfo] = Seq.empty,
      policy: EnclosingClusterPolicy = Inherit): SparkPlanInfo = {
    new TestSparkPlanInfo(name, children, metrics, policy)
  }

  private def graphView(graph: SparkPlanGraph): GraphView = {
    val nodeViews = graph.allNodes.map { node =>
      val metrics = node.metrics.map { metric =>
        (metric.name, metric.accumulatorId, metric.metricType)
      }.toSeq
      node.id -> NodeView(node.name, node.desc, metrics)
    }.toSeq.sortBy(_._1)
    val clusterViews = graph.nodes.collect {
      case cluster: SparkPlanGraphCluster =>
        cluster.id -> cluster.nodes.map(_.id).toSeq
    }.toSeq.sortBy(_._1)
    val edgeViews = graph.edges.map(edge => (edge.fromId, edge.toId)).toSeq
      .sortBy(edge => (edge._1, edge._2))
    GraphView(graph.nodes.map(_.id).toSeq, nodeViews, clusterViews, edgeViews)
  }

  private def directBoundaryPlan(policy: EnclosingClusterPolicy): SparkPlanInfo = {
    val boundary = plan(
      "Exchange",
      children = Seq(plan("Filter", children = Seq(plan("Scan")))),
      metrics = Seq(new SQLMetricInfo("boundary metric", 99L, "sum")),
      policy = policy)
    val reusedBoundary = plan("ReusedExchange", children = Seq(boundary))
    plan("WholeStageCodegen",
      children = Seq(plan("Project", children = Seq(boundary, reusedBoundary))))
  }

  private def queryStageBoundaryPlan(policy: EnclosingClusterPolicy): SparkPlanInfo = {
    val nestedStage = plan(
      "WholeStageCodegen",
      children = Seq(plan("Filter", children = Seq(plan("Scan")))))
    val boundary = plan(
      "Exchange",
      children = Seq(plan("ShuffleQueryStage", children = Seq(nestedStage))),
      policy = policy)
    plan("WholeStageCodegen",
      children = Seq(plan("Project", children = Seq(boundary))))
  }

  private def nestedWholeStagePlan(policy: EnclosingClusterPolicy): SparkPlanInfo = {
    val nested = plan(
      "WholeStageCodegen",
      children = Seq(plan("Filter", children = Seq(plan("Scan")))),
      policy = policy)
    plan("WholeStageCodegen",
      children = Seq(plan("Project", children = Seq(nested))))
  }

  private def topLevelBoundaryPlan(policy: EnclosingClusterPolicy): SparkPlanInfo = {
    plan("Exchange",
      children = Seq(plan("Filter", children = Seq(plan("Scan")))),
      policy = policy)
  }

  private def reusedSubqueryPlan(policy: EnclosingClusterPolicy): SparkPlanInfo = {
    val reusedSubquery = plan(
      "ReusedSubquery",
      children = Seq(plan("Filter")),
      policy = policy)
    plan("WholeStageCodegen",
      children = Seq(plan("Project", children = Seq(reusedSubquery))))
  }

  test("Break moves an ordinary subtree out of an enclosing cluster and preserves reuse") {
    val inherited = graphView(ToolsPlanGraph(directBoundaryPlan(Inherit)))
    val broken = graphView(ToolsPlanGraph(directBoundaryPlan(Break)))
    val expectedNodes = Seq(
      0L -> NodeView("WholeStageCodegen", "WholeStageCodegen", Seq.empty),
      1L -> NodeView("Project", "Project", Seq.empty),
      2L -> NodeView("Exchange", "Exchange", Seq(("boundary metric", 99L, "sum"))),
      3L -> NodeView("Filter", "Filter", Seq.empty),
      4L -> NodeView("Scan", "Scan", Seq.empty))
    val expectedEdges = Seq((2L, 1L), (2L, 1L), (3L, 2L), (4L, 3L))

    assert(inherited.nodes == expectedNodes)
    assert(broken.nodes == expectedNodes)
    assert(inherited.edges == expectedEdges)
    assert(broken.edges == expectedEdges)
    assert(inherited.topLevelIds == Seq(0L))
    assert(inherited.clusters == Seq(0L -> Seq(1L, 2L, 3L, 4L)))
    assert(broken.topLevelIds == Seq(0L, 2L, 3L, 4L))
    assert(broken.clusters == Seq(0L -> Seq(1L)))
  }

  test("Break preserves edges through an elided ShuffleQueryStage and nested cluster") {
    val inherited = graphView(ToolsPlanGraph(queryStageBoundaryPlan(Inherit)))
    val broken = graphView(ToolsPlanGraph(queryStageBoundaryPlan(Break)))
    val expectedNodes = Seq(
      0L -> NodeView("WholeStageCodegen", "WholeStageCodegen", Seq.empty),
      1L -> NodeView("Project", "Project", Seq.empty),
      2L -> NodeView("Exchange", "Exchange", Seq.empty),
      3L -> NodeView("WholeStageCodegen", "WholeStageCodegen", Seq.empty),
      4L -> NodeView("Filter", "Filter", Seq.empty),
      5L -> NodeView("Scan", "Scan", Seq.empty))
    val expectedEdges = Seq((2L, 1L), (4L, 2L), (5L, 4L))

    assert(inherited.nodes == expectedNodes)
    assert(broken.nodes == expectedNodes)
    assert(inherited.edges == expectedEdges)
    assert(broken.edges == expectedEdges)
    assert(inherited.topLevelIds == Seq(0L, 3L))
    assert(inherited.clusters == Seq(0L -> Seq(1L, 2L), 3L -> Seq(4L, 5L)))
    assert(broken.topLevelIds == Seq(0L, 2L, 3L))
    assert(broken.clusters == Seq(0L -> Seq(1L), 3L -> Seq(4L, 5L)))
  }

  test("Break composes with WholeStageCodegen canonical dispatch") {
    val inherited = graphView(ToolsPlanGraph(nestedWholeStagePlan(Inherit)))
    val broken = graphView(ToolsPlanGraph(nestedWholeStagePlan(Break)))
    val expected = GraphView(
      topLevelIds = Seq(0L, 2L),
      nodes = Seq(
        0L -> NodeView("WholeStageCodegen", "WholeStageCodegen", Seq.empty),
        1L -> NodeView("Project", "Project", Seq.empty),
        2L -> NodeView("WholeStageCodegen", "WholeStageCodegen", Seq.empty),
        3L -> NodeView("Filter", "Filter", Seq.empty),
        4L -> NodeView("Scan", "Scan", Seq.empty)),
      clusters = Seq(0L -> Seq(1L), 2L -> Seq(3L, 4L)),
      edges = Seq((3L, 1L), (4L, 3L)))

    assert(inherited == expected)
    assert(broken == expected)
  }

  test("Break at a top-level ordinary node is placement neutral") {
    val inherited = graphView(ToolsPlanGraph(topLevelBoundaryPlan(Inherit)))
    val broken = graphView(ToolsPlanGraph(topLevelBoundaryPlan(Break)))
    val expected = GraphView(
      topLevelIds = Seq(0L, 1L, 2L),
      nodes = Seq(
        0L -> NodeView("Exchange", "Exchange", Seq.empty),
        1L -> NodeView("Filter", "Filter", Seq.empty),
        2L -> NodeView("Scan", "Scan", Seq.empty)),
      clusters = Seq.empty,
      edges = Seq((1L, 0L), (2L, 1L)))

    assert(inherited == expected)
    assert(broken == expected)
  }

  test("Break is applied before ReusedSubquery dispatch") {
    val inherited = graphView(ToolsPlanGraph(reusedSubqueryPlan(Inherit)))
    val broken = graphView(ToolsPlanGraph(reusedSubqueryPlan(Break)))
    val expectedNodes = Seq(
      0L -> NodeView("WholeStageCodegen", "WholeStageCodegen", Seq.empty),
      1L -> NodeView("Project", "Project", Seq.empty),
      2L -> NodeView("Filter", "Filter", Seq.empty))

    assert(inherited == GraphView(
      topLevelIds = Seq(0L),
      nodes = expectedNodes,
      clusters = Seq(0L -> Seq(1L, 2L)),
      edges = Seq((2L, 1L))))
    assert(broken == GraphView(
      topLevelIds = Seq(0L, 2L),
      nodes = expectedNodes,
      clusters = Seq(0L -> Seq(1L)),
      edges = Seq((2L, 1L))))
  }
}
