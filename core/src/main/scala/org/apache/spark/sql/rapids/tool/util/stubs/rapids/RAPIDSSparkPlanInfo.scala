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

package org.apache.spark.sql.rapids.tool.util.stubs.rapids

import org.apache.spark.sql.execution.metric.SQLMetricInfo
import org.apache.spark.sql.rapids.tool.util.stubs.{PWSparkPlanInfo, SparkPlanInfo}

/**
 * Represents execution plan information for NVIDIA RAPIDS GPU-accelerated Spark nodes.
 *
 * cuDF plugin enables GPU acceleration of Spark workloads.
 * This class captures both the GPU-specific node information (e.g., "GpuProject",
 * "GpuColumnarToRow") and the equivalent CPU Spark node information for compatibility
 * and cross-platform analysis.
 *
 * @param actualName The RAPIDS GPU-specific node name (e.g., "GpuProject")
 * @param actualDesc The RAPIDS GPU-specific description
 * @param sparkName  The equivalent CPU Spark node name (e.g., "Project")
 * @param sparkDesc  The equivalent CPU Spark description
 * @param children   Child execution plan nodes in the plan tree
 * @param metadata   Additional metadata associated with the RAPIDS plan node
 * @param metrics    SQL metrics collected for this RAPIDS plan node
 */
case class RAPIDSSparkPlanInfo(
    actualName: String,
    actualDesc: String,
    sparkName: String,
    sparkDesc: String,
    override val children: Seq[SparkPlanInfo],
    override val metadata: Map[String, String],
    override val metrics: Seq[SQLMetricInfo]
) extends PWSparkPlanInfo(
  actualName, actualDesc, sparkName, sparkDesc, children, metadata, metrics) {

}
