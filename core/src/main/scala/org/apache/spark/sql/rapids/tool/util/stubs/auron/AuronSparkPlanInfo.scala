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

package org.apache.spark.sql.rapids.tool.util.stubs.auron

import org.apache.spark.sql.execution.metric.SQLMetricInfo
import org.apache.spark.sql.rapids.tool.util.stubs.{EnclosingClusterPolicy, PWSparkPlanInfo,
  SparkPlanInfo}

/**
 * Represents execution plan information for Auron engine nodes.
 *
 * @param actualName The Auron-specific node name (e.g., "NativeProject")
 * @param actualDesc The Auron-specific description
 * @param sparkName  The equivalent Spark node name (e.g., "Project")
 * @param sparkDesc  The equivalent Spark description
 * @param children   Child execution plan nodes in the plan tree
 * @param metadata   Additional metadata associated with the Auron plan node
 * @param metrics    SQL metrics collected for this Auron plan node
 * @param enclosingClusterPolicy How this node interacts with an active enclosing cluster
 */
case class AuronSparkPlanInfo(
    actualName: String,
    actualDesc: String,
    sparkName: String,
    sparkDesc: String,
    override val children: Seq[SparkPlanInfo],
    override val metadata: Map[String, String],
    override val metrics: Seq[SQLMetricInfo],
    override val enclosingClusterPolicy: EnclosingClusterPolicy
) extends PWSparkPlanInfo(
  actualName, actualDesc, sparkName, sparkDesc, children, metadata, metrics,
  enclosingClusterPolicy) {

}
