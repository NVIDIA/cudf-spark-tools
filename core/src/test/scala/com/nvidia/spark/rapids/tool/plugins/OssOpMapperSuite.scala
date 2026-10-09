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

package com.nvidia.spark.rapids.tool.plugins

import com.nvidia.spark.rapids.tool.planparser.auron.AuronOssOpMapper
import com.nvidia.spark.rapids.tool.planparser.db.PhotonOssOpMapper
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.scheduler.SparkListenerEvent
import org.apache.spark.sql.execution.{SparkPlanInfo => UpstreamSparkPlanInfo}
import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.plangraph.{PhotonSparkPlanGraphCluster, ToolsPlanGraph}
import org.apache.spark.sql.rapids.tool.util.stubs.EnclosingClusterPolicy
import org.apache.spark.sql.rapids.tool.util.stubs.EnclosingClusterPolicy.{Break, Inherit}

class OssOpMapperSuite extends AnyFunSuite {

  private class TestApp extends AppBase(None, None) {
    override def processEvent(event: SparkListenerEvent): Boolean = false
  }

  private val app = new TestApp

  private def upstream(
      name: String,
      children: Seq[UpstreamSparkPlanInfo] = Seq.empty): UpstreamSparkPlanInfo = {
    new UpstreamSparkPlanInfo(name, name, children, Map.empty, Seq.empty)
  }

  private def assertPolicy(
      mapper: OssOpMapperTrait,
      name: String,
      expected: EnclosingClusterPolicy): Unit = {
    val plan = upstream(name)
    assert(mapper.enclosingClusterPolicy(plan) == expected)
    assert(mapper.toPlatformAwarePlan(plan, app).enclosingClusterPolicy == expected)
  }

  test("Photon mapper extracts boundary tokens and preserves policy on converted plans") {
    Seq(
      "PhotonShuffleExchangeSource",
      "PhotonShuffleExchangeSource (42)").foreach { name =>
      assertPolicy(PhotonOssOpMapper, name, Break)
    }

    Seq(
      "PhotonShuffleExchangeSink",
      "PhotonShuffleMapStage",
      "PhotonUnknownOperator").foreach { name =>
      assertPolicy(PhotonOssOpMapper, name, Inherit)
    }
  }

  test("Photon cluster boundaries stay aligned with the loaded mapping inventory") {
    val mapping = PhotonOssOpMapper.ossSparkMapping
    val missingBoundaries = PhotonOssOpMapper.CLUSTER_BOUNDARY_OPERATORS.diff(mapping.keySet)
    val exchangeOperators = mapping.collect {
      case (operator, "Exchange") => operator
    }.toSet

    assert(missingBoundaries.isEmpty)
    assert(exchangeOperators == Set(
      "PhotonShuffleExchangeSink",
      "PhotonShuffleExchangeSource"))
  }

  test("non-Photon mappers inherit cluster placement and preserve it on converted plans") {
    Seq[(OssOpMapperTrait, String)](
      AuronOssOpMapper -> "NativeShuffleExchange",
      GpuOssOpMapper -> "GpuShuffleExchange",
      OssOpMapper -> "WholeStageCodegen (1)").foreach { case (mapper, name) =>
      assertPolicy(mapper, name, Inherit)
    }
  }

  test("mapped Photon WholeStageCodegen remains a Photon cluster") {
    val upstreamPlan = upstream(
      "PhotonShuffleMapStage",
      children = Seq(upstream("Project")))
    assert(PhotonOssOpMapper.enclosingClusterPolicy(upstreamPlan) == Inherit)

    val mappedPlan = PhotonOssOpMapper.toPlatformAwarePlan(upstreamPlan, app)
    assert(mappedPlan.enclosingClusterPolicy == Inherit)
    assert(mappedPlan.nodeName == "WholeStageCodegen")

    val graph = ToolsPlanGraph(mappedPlan)
    assert(graph.nodes.size == 1, s"Expected one Photon cluster, found: ${graph.nodes}")
    val cluster = graph.nodes.head match {
      case value: PhotonSparkPlanGraphCluster => value
      case other => fail(s"Expected a Photon cluster, found: $other")
    }
    assert(cluster.name == "WholeStageCodegen")
    assert(cluster.platformName == "PhotonShuffleMapStage")
    assert(cluster.nodes.map(_.name).toSeq == Seq("Project"))
  }
}
