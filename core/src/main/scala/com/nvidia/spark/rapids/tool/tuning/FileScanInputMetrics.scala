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

import scala.util.Try

import com.nvidia.spark.rapids.tool.profiling.{DataSourceProfileResult, StageAggTaskMetricsProfileResult}

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.plangraph.ToolsPlanGraph

private[tool] object FileScanInputMetrics {
  private val cachePlanNodes = Seq("InMemoryTableScan", "InMemoryRelation")

  def latestPlanGraph(
      appInfo: AppBase,
      sqlId: Long,
      version: Int): Option[ToolsPlanGraph] = {
    appInfo.sqlManager.applyToPlanModel(sqlId) { planModel =>
      if (planModel.plan.version == version) {
        Try(planModel.getToolsPlanGraph).toOption
      } else {
        None
      }
    }.flatten
  }

  def maxInputBytes(
      dataSources: Seq[DataSourceProfileResult],
      stageMetrics: Seq[StageAggTaskMetricsProfileResult],
      planGraphForSqlVersion: (Long, Int) => Option[ToolsPlanGraph]): Option[Double] = {
    val candidateStages = dataSources.iterator
      .filter(_.fromFinalPlan)
      .flatMap { dataSource =>
        planGraphForSqlVersion(dataSource.sqlID, dataSource.version).iterator.flatMap { graph =>
          graph.getNodeStageRawAssignment(dataSource.nodeId).iterator.filter { stageId =>
            val stageNodeIds = graph.getStageNodesByRawAssignment(stageId)
            val stageNodes = graph.allNodes.filter(node => stageNodeIds.contains(node.id))
            stageNodes.nonEmpty && !stageNodes.exists { node =>
              Seq(node.name, node.desc).exists { text =>
                Option(text).exists(nonNullText => cachePlanNodes.exists(nonNullText.contains))
              }
            }
          }.map(_.toLong)
        }
      }.toSet

    stageMetrics.iterator
      .filter(metric => candidateStages.contains(metric.id))
      .map(_.inputBytesReadMax)
      .filter(_ > 0L)
      .reduceOption((left: Long, right: Long) => math.max(left, right))
      .map(_.toDouble)
  }
}
