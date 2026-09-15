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

import com.nvidia.spark.rapids.tool.PlatformNames
import com.nvidia.spark.rapids.tool.planparser.db.PhotonOssOpMapper
import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker


class PhotonPlanParserSuite extends BasePlanParserSuite {

  // scalastyle:off line.size.limit
  // Test cases for Photon nodes. We could add more operators here.
  val photonOpTestCases: Seq[(String, String)] = Seq(
    "PhotonBroadcastNestedLoopJoin" -> "BroadcastNestedLoopJoin",  // Case: Photon specific parser
    "PhotonProject" -> "Project",                                  // Case: Fallback to Spark CPU parser
    "PhotonShuffleMapStage" ->  "WholeStageCodegen"                // Case: WholeStageCodegen operator
  )
  // scalastyle:on line.size.limit

  /**
   * Test that verifies Photon-specific operators are correctly parsed and converted to their
   * Spark CPU equivalents in the execution plan.
   *
   * This test:
   * 1. Loads an event log from a Databricks environment running with Photon enabled
   * 2. Parses all SQL plans in the application using SQLPlanParser
   * 3. Extracts all execution operators (ExecInfo) from the parsed plans
   * 4. For each Photon operator in the test cases:
   *    - Verifies the corresponding Spark operator name exists in the parsed results
   *    - Verifies the original Photon operator name does NOT appear in the results
   *
   * This ensures that:
   * - Photon operators are recognized and handled by the parser
   * - They are correctly converted to Spark-equivalent names for compatibility and analysis
   * - The conversion applies to regular operators (e.g., PhotonProject -> Project),
   *   Photon-specific parsers (e.g., PhotonBroadcastNestedLoopJoin -> BroadcastNestedLoopJoin),
   *   and cluster operators (e.g., PhotonShuffleMapStage -> WholeStageCodegen)
   */
  test("Operator in photon is parsed as Spark equivalent") {
    val eventLog = s"$qualLogDir/nds_q88_photon_db_13_3.zstd"
    val pluginTypeChecker = new PluginTypeChecker()
    val app = createAppFromEventlog(eventLog, platformName = PlatformNames.DATABRICKS_AWS)
    assert(app.sqlPlans.nonEmpty)
    val parsedPlans = app.sqlPlans.map { case (sqlID, plan) =>
      SQLPlanParser.parseSQLPlan(plan, sqlID, "", pluginTypeChecker, app)
    }
    val allExecInfo = getAllExecsFromPlan(parsedPlans.toSeq)
    photonOpTestCases.foreach { case (photonName, sparkName) =>
      val sparkOpExists = allExecInfo.exists(_.exec.contains(sparkName))
      val photonOpExists = allExecInfo.exists(_.exec.contains(photonName))
      assert(sparkOpExists,
        s"Failed to find Spark operator $sparkName for Photon operator $photonName")
      assert(!photonOpExists,
        s"Failed to parse Photon operator $photonName as Spark operator $sparkName")
    }
  }

  // Photon nodes that appear in Databricks 15.4 and 17.3 plans and not in the 13.3 log above.
  // Each expected value is written out, so a wrong or missing entry in the mapping file fails
  // here rather than passing through the generic parser as an unsupported exec.
  val photonOpTestCasesNewerRuntimes: Seq[(String, String)] = Seq(
    "PhotonWriteStage" -> "WholeStageCodegen",
    "PhotonParquetWriter" -> "WriteFiles",
    "PhotonColumnarToRow" -> "ColumnarToRow",
    "PhotonMetadataSubquery" -> "Subquery",
    "PhotonRuntimeFilterSource" -> "Subquery",
    "PhotonRange" -> "Range",
    "PhotonJsonScan" -> "Scan"
  )

  test("Photon operators from Databricks 15.4 and 17.3 map to their Spark equivalents") {
    photonOpTestCasesNewerRuntimes.foreach { case (photonName, sparkName) =>
      assert(PhotonOssOpMapper.mapContentToOss(photonName) == sparkName,
        s"$photonName should map to $sparkName")
    }
    // The scan keeps its format suffix, as PhotonScan does, so the read parser sees "Scan json".
    assert(PhotonOssOpMapper.mapContentToOss("PhotonJsonScan json") == "Scan json")
    // A Photon node with no entry is left as it is; that is what the generic parser then reports
    // as unsupported.
    assert(PhotonOssOpMapper.mapContentToOss("PhotonNotARealNode") == "PhotonNotARealNode")
  }
}
