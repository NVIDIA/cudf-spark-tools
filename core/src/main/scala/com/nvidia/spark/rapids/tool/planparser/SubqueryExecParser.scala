/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

import com.nvidia.spark.rapids.tool.planparser.ops.{OpTypes, UnsupportedReasonRef}
import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.plangraph.SparkPlanGraphNode

// SubQuery is simply a "collect" execution.
// It points to the actual execution. cuDF plugin usually skips that exec. Here we
// can represent it to be ignored with shouldRemove set to true.
// The reason we are implementing this as class is for future extensibility and to read the metrics.
case class SubqueryExecParser(
    override val node: SparkPlanGraphNode,
    override val checker: PluginTypeChecker,
    override val sqlID: Long,
    override val app: Option[AppBase]
) extends GenericExecParser(
    node,
    checker,
    sqlID,
    // set the reason why it shows up as unsupported.
    unsupportedReason = Some(UnsupportedReasonRef.getOrCreate(
      "Subquery is a collect execution pointing to an actual one")),
    app = app
) {
  /** Subquery uses driver-side metrics */
  override protected def useDriverMetrics: Boolean = true

  /**
   * Duration based on time to collect data on the driver.
   * See [[GenericExecParser.durationSqlMetrics]] for details.
   */
  override protected val durationSqlMetrics: Set[String] = Set(
    "time to collect",
    "time to collect (ms)")

  // Subquery is always marked as not supported (isSupported = false)
  override def pullSupportedFlag(registeredName: Option[String] = None): Boolean = false

  setUnsupportedReason("Subquery is a collect execution pointing to an actual one")

  // TODO: Should we also collect the "data size" metric?
}

object SubqueryExecParser extends GroupParserTrait {
  val execName = "Subquery"

  override def accepts(nodeName: String): Boolean = {
    nodeName.equals(execName)
  }

  /**
   * Create an ExecParser for the given node.
   *
   * @param node     spark plan graph node
   * @param checker  plugin type checker
   * @param sqlID    SQL ID
   * @param execName optional exec name override
   * @param opType   optional op type override
   * @param app      optional AppBase instance
   * @return an ExecParser for the given node
   */
  override def createExecParser(
      node: SparkPlanGraphNode,
      checker: PluginTypeChecker,
      sqlID: Long,
      execName: Option[String] = None,
      opType: Option[OpTypes.Value] = None,
      app: Option[AppBase]): ExecParser = {
    SubqueryExecParser(node, checker, sqlID, app)
  }
}
