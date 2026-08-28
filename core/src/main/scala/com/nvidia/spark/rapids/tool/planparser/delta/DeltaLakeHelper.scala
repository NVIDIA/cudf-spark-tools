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

package com.nvidia.spark.rapids.tool.planparser.delta

import scala.util.matching.Regex

import com.nvidia.spark.rapids.tool.planparser.{DataWritingCmdWrapper, DataWritingCommandExecParser, ExecInfo, ExecParser, GenericExecParser, ReadParser}
import com.nvidia.spark.rapids.tool.planparser.ops.OpTypes
import com.nvidia.spark.rapids.tool.plugins.{AppPropVersionExtractorFromCPTrait, PropConditionOnSparkExtTrait}
import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker

import org.apache.spark.sql.rapids.tool.plangraph.{SparkPlanGraphCluster, SparkPlanGraphNode}
import org.apache.spark.sql.rapids.tool.store.SQLPlanModel
import org.apache.spark.sql.rapids.tool.util.StringUtils

// A class used to handle the DL writeOps such as:
// - AppendDataExecV1
// - OverwriteByExpressionExecV1
// - SaveIntoDataSourceCommand
// This is not a Case class so later we can overwrite it.
class DLWriteWithFormatAndSchemaParser(node: SparkPlanGraphNode,
    checker: PluginTypeChecker,
    sqlID: Long) extends ExecParser {
  private val cleanedUpNodeName = GenericExecParser.cleanupNodeName(node)
  // We use "DataWritingCommandExec" to get the speedupFactor of AppendDataExecV1
  val fullExecName: String = DataWritingCommandExecParser.getPhysicalExecName(cleanedUpNodeName)
  override def parse: ExecInfo = {
    // The node description has information about the table schema and its format.
    // We do not want the op to me marked as RDD or UDF if the node description contains some
    // expressions that match UDF/RDD.
    val (speedupFactor, isExecSupported) = if (checker.isExecSupported(fullExecName)) {
      (checker.getSpeedupFactor(fullExecName), true)
    } else {
      (1.0, false)
    }
    val dataFormat = DeltaLakeHelper.getWriteFormat
    val writeSupported = checker.isWriteFormatSupported(dataFormat)
    val finalSpeedupFactor = if (writeSupported) speedupFactor else 1.0

    // Do not remove prefix Execute to be consistent with other execs.
    ExecInfo.createExecNoNode(sqlID, cleanedUpNodeName,
      s"Format: $dataFormat", finalSpeedupFactor, None, node.id, OpTypes.WriteExec,
      isSupported = writeSupported && isExecSupported, children = None, expressions = Seq.empty)
  }
}

object DeltaLakeHelper extends PropConditionOnSparkExtTrait
    with AppPropVersionExtractorFromCPTrait {
  // A DeltaLake app is identified using the following properties from spark properties.
  override val extensionRegxMap: Map[String, String] = Map(
    "spark.sql.extensions" -> ".*DeltaSparkSessionExtension.*",
    "spark.sql.catalog.spark_catalog" -> ".*DeltaCatalog.*"
  )
  // The regex pattern used to extract the DeltaLake jar version from classpath entries.
  // Matches patterns like:
  // - spark://server:41661/jars/io.delta_delta-spark_2.12-3.3.1.jar
  // - file:///path/to/io.delta_delta-storage-3.3.1.jar
  // - /local/path/io.delta_delta-core_2.12-3.3.1.jar
  // The pattern:
  // - .* matches any prefix (path before the jar name)
  // - io\.delta_ matches "io.delta_" (the jar prefix)
  // - delta- matches the literal "delta-"
  // - [\w.-]+ matches the middle part (spark_2.12, storage, core_2.12, etc.)
  // - -(\d+\.\d+\.\d+) captures the version number like "-3.3.1"
  // - \.jar matches ".jar"
  override val cpKeyRegex: Regex = """.*io\.delta_delta-[\w.-]+-(\d+\.\d+\.\d+)\.jar""".r

  // The pattern used to identify deltaLake metadata tables
  // Sometimes this pattern does not show up in the node description when the node.descr is
  // truncated. In that case, we might find the full pattern in the planInfo or the sql description.
  val DELTALAKE_META_KEYWORD: String = "/_delta_log"
  // DeltaLake scan Existing RDD for delta_log contains the following pattern.
  val DELTALAKE_META_SCAN_RDD_KEYWORD: String = "Delta Table State"
  // Another pattern that can be used to identify deltaLake metadata tables.
  val DELTALAKE_LOG_FILE_KEYWORD: String = "DeltaLogFileIndex"
  // we look for the serdeLibrary which is part of the node description:
  // Serde Library: org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe
  // private val serdeRegex = "Serde Library: ([\\w.]+)".r
  private val serdeRegex = "Serde Library: ([\\w.]+)".r
  // We look for the schema in the node description. It is in the following format
  // Schema: root
  //  |-- field_00: timestamp (nullable = true)
  //  |-- field_01: string (nullable = true)
  // the following regex matches lines that start with ( --|) and it will create two groups
  // [fieldName, type]
  private val schemaRegex =
    "\\s+\\|--\\s+([a-zA-Z]+(?:_[a-zA-Z]+)*):\\s+([a-zA-Z]+(?:_[a-zA-Z]+)*)\\s+\\(".r
  private val atomicCreateTableExec = DataWritingCommandExecParser.atomicCreateTableExec
  private val atomicReplaceTableExec = DataWritingCommandExecParser.atomicReplaceTableExec
  private val appendDataExecV1 = DataWritingCommandExecParser.appendDataExecV1
  private val overwriteByExprExecV1 = DataWritingCommandExecParser.overwriteByExprExecV1
  // define the list of writeExecs that also exist in Spark
  private val deltaExecsFromSpark = Set(
    appendDataExecV1,
    overwriteByExprExecV1,
    atomicReplaceTableExec,
    atomicCreateTableExec)

  // keywords used to verify that the operator provider is DeltaLake
  private val nodeDescKeywords = Set(
    // WriteIntoDeltaBuilder is a databricks keyword specific, it does not exist in open source.
    // "WriteIntoDeltaBuilder"
    // The following entries are for open source DeltaLake
    "DeltaTableV2")

  def acceptsExclusiveWriteOp(nodeName: String): Boolean = {
    DeltaLakeOps.isExclusiveDeltaWriteOp(nodeName)
  }

  def accepts(nodeName: String): Boolean = {
    acceptsExclusiveWriteOp(nodeName) || deltaExecsFromSpark.contains(nodeName)
  }

  def acceptsWriteOp(node: SparkPlanGraphNode): Boolean = {
    if (acceptsExclusiveWriteOp(node.name)) {
      true
    } else if (deltaExecsFromSpark.contains(node.name)) {
      if (node.name.contains(appendDataExecV1) || node.name.contains(overwriteByExprExecV1)) {
        // those two execs require the existence of the keywords in the node description
        nodeDescKeywords.forall(s => node.desc.contains(s))
      } else if (node.name.contains(atomicReplaceTableExec)
          || node.name.contains(atomicCreateTableExec)) {
        // AtomicReplace and AtomicCreate have different format.
        // To decide whether they are supported or not, we need to check whether the TableSpec
        // second argument is "delta" provider. The sample below shows a table Spec with
        // Delta Provider. If the argument is none, we assume the provider is not Delta.
        // For simplicity, we will match regex on "*delta*".
        //
        // AtomicReplaceTableAsSelectExec has a different format
        // AtomicReplaceTableAsSelect [num_affected_rows#ID_0L, num_inserted_rows#ID_1L],
        // com.databricks.sql.managedcatalog.UnityCatalogV2Proxy@XXXXX,
        // DB.VAR_2, Union false, false,
        // TableSpec(Map(),Some(delta),Map(),None,None,None,false,Set(),None,None,None),
        // [replaceWhere=VAR_1 IN ('WHATEVER')], true,
        // org.apache.spark.sql.execution.datasources.
        // v2.DataSourceV2Strategy$$Lambda$XXXXX
        node.desc.contains("delta")
      } else {
        false
      }
    } else {
      false
    }
  }

  def getWriteFormat: String = {
    // by default Delta Operators return write format
    "Delta"
  }

  def parseNode(node: SparkPlanGraphNode,
      checker: PluginTypeChecker,
      sqlID: Long): ExecInfo = {
    node match {
      case n if acceptsWriteOp(n) =>
        val opExec = new DLWriteWithFormatAndSchemaParser(node, checker, sqlID)
        opExec.parse
      case _ => throw new IllegalArgumentException(s"Unhandled Exec for node ${node.name}")
    }
  }

  /**
   * Get the write command wrapper for the given node deltaLake exec node.
   * This method should be called only if the node passes the `acceptsWriteOp` check.
   * @param node the deltaLake write exec
   * @return the write command wrapper
   */
  def getWriteCMDWrapper(node: SparkPlanGraphNode): Option[DataWritingCmdWrapper] = {
    val wcmd = DeltaLakeOps.getExecNoPrefix(node.name) match {
      case Some(cmd) => cmd
      case _ =>
        deltaExecsFromSpark.find(node.name.contains(_)) match {
          case Some(cmd) => cmd
          case _ => StringUtils.UNKNOWN_EXTRACT
        }
    }
    // The format must be delta
    Some(DataWritingCmdWrapper(wcmd, DataWritingCommandExecParser.dataWriteCMD, getWriteFormat))
  }

  // Kept for future use if we find that SerDe library can be used to deduce any information to
  // reflect on the support of the Op
  def getSerdeLibrary(nodeDesc: String): Option[String] = {
    val matches = serdeRegex.findAllMatchIn(nodeDesc).toSeq
    if (matches.nonEmpty) {
      Option(matches.head.group(1))
    } else {
      None
    }
  }

  // Retrieve the schema of a DeltaLake table from some ops like Overwrite and Append
  def getSchema(nodeDesc: String): String = {
    schemaRegex.findAllMatchIn(nodeDesc).map { m =>
      s"${m.group(1)}:${m.group(2)}"
    }.mkString(",")
  }

  /**
   * Check if the node is a Delta meta-data operation.
   *
   * @param sqlPIGEntry container that references the planInfo and the planGraph
   * @param sqlDesc the physical plan description
   * @param node the node to check
   * @return true if the node is a Delta meta-data operation
   */
  def isDeltaOpNode(sqlPIGEntry: SQLPlanModel,
      sqlDesc: String,
      node: SparkPlanGraphNode): Boolean = {
    node match {
      case _: SparkPlanGraphCluster =>
        // This is WholeStageCodeGen
        // It is consider a delta-op if it has the _delta_log in the physical description
        // TODO: Improve this by checking if any of the nodes array is Delta Op instead of
        //       using "contains" on SQLDesc which is duplicating the work done the nodes array
        sqlDesc.contains("_delta_log")
      case _ =>
        // Some nodes have a $ at the end of the name
        val normalizedName = node.name.stripSuffix("$")
        normalizedName match {
          case n if n.contains("MergeIntoCommandEdge") =>
            // this is a bit odd but GPU doesn't accelerate anyway, this might be due to
            // differences in data between runs
            // Execute MergeIntoCommandEdge (1)
            //   +- MergeIntoCommandEdge (2)
            sqlPIGEntry.getToolsPlanGraph.allNodes.size < 2
          case "LocalTableScan" =>
            sqlDesc.contains("stats_parsed.numRecords") ||
              sqlPIGEntry.getToolsPlanGraph.allNodes.size == 1
          case s if ReadParser.isScanNode(s) =>
            // cuDF plugin checks that a node is Delta Scan by looking at the type of the exec
            // "RDDScanExec" and checking for some string patterns in the Relation/schema
            // sqlDesc.contains("Delta Table State") ||
            //   sqlDesc.contains("Delta Table Checkpoint") ||
            //   sqlDesc.contains("delta_log")
            if (sqlDesc.contains(DELTALAKE_META_KEYWORD) ||
              sqlDesc.contains(DELTALAKE_META_SCAN_RDD_KEYWORD) ||
              sqlDesc.contains("Delta Table Checkpoint") ||
              sqlDesc.contains("_databricks_internal")) {
              true
            } else if (sqlDesc.contains("checkpoint")) {
              // double check it has parquet - regex are expensive though so only do
              // if necessary
              val checkpointRegEx = ".*Location:(.*)checkpoint(.*).parquet(.*)".r
              checkpointRegEx.findFirstIn(sqlDesc).isDefined
            } else {
              false
            }
          case _ => false
        }
    }
  }
}
