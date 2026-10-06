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

package com.nvidia.spark.rapids.tool.views.qualification

import com.nvidia.spark.rapids.tool.ToolTextFileWriter
import com.nvidia.spark.rapids.tool.planparser.UnsupportedExecSummary
import com.nvidia.spark.rapids.tool.planparser.ops.OperatorCounter
import com.nvidia.spark.rapids.tool.qualification.QualOutputWriter
import org.apache.hadoop.conf.Configuration
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization

import org.apache.spark.sql.rapids.tool.ToolUtils
import org.apache.spark.sql.rapids.tool.qualification.{EstimatedPerSQLSummaryInfo, QualificationSummaryInfo}


/**
 * This file contains the code for writing out the per-app tool's output to CSV files.
 * Each type of output is written out to a different CSV file.
 */

/**
 * A trait that defines how to write out the tool's output to a CSV file.
 * The tables are defined in `QualOutputTableDefinition`.
 */
trait AppQualTableTrait extends QualTableBuilderTrait[QualificationSummaryInfo] {

}

/**
 * Abstract class for generating the Qualification report per application.
 *
 * @param tableMeta table definition.
 * @param rootDirectory root directory of the report.
 * @param hadoopConf Hadoop configuration.
 */
abstract class AppQualTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTableTrait {
  /**
   * The label for the table.
   */
  def getLabel: String = tableMeta.label

  /**
   * The description for the table.
   */
  def getDescription: String = tableMeta.description.getOrElse("")

  /**
   * The file name for the table.
   */
  def getFileName: String = tableMeta.fileName

  /**
   * Writes out the header for the table.
   */
  def writeCSVHeader(fWriter: ToolTextFileWriter): Unit = {
    fWriter.writeLn(tableMeta.columns.map(_.name).mkString(delim))
  }

  /**
   * Generates the output report for the table.
   * @param rec the qualification summary information for one application.
   */
  override def generateOutputReport(rec: QualificationSummaryInfo): Unit = {
    if (generateFileEnabled(rec)) {
      val csvFileWriter = new ToolTextFileWriter(getDirectory(() => rec.appId),
        getFileName,
        getDescription, Option(hadoopConf))
      try {
        // write the header
        writeCSVHeader(csvFileWriter)
        appendDataToWriter(csvFileWriter, rec)
      } finally {
        csvFileWriter.close()
      }
    }
  }
}

// ClusterInfo table is JSON file format.
// It can be disabled by the CLI argument
class AppQualClusterInfoTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {

  implicit val formats: DefaultFormats.type = DefaultFormats

  override def writeCSVHeader(fWriter: ToolTextFileWriter): Unit = {
    // No headers for JSON file format
  }

  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    fWriter.writeLn(Serialization.writePretty(rec.clusterSummary))
  }
}

// UDF detection report (JSON). Reports all detected UDFs with metadata.
class AppQualUdfReportTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {

  implicit val formats: DefaultFormats.type = DefaultFormats

  override def writeCSVHeader(fWriter: ToolTextFileWriter): Unit = {
    // No headers for JSON file format
  }

  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    val udfReport = UdfReportBuilder.generateReport(rec)
    fWriter.writeLn(Serialization.writePretty(udfReport))
  }
}

// Per-App execs CSV report. It can be disabled by user CLI argument.
class AppQualExecTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {

  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    val booleanTrue = "true"
    val booleanFalse = "false"
    val zeroDurationStr = "0"
    val emptyString = formatStr("")
    // write one Plan at a time.
    rec.planInfo.foreach { pInfo =>
      val sqlIDStr = pInfo.sqlID.toString
      pInfo.execInfo.flatMap { eInfo =>
        eInfo.children.getOrElse(Seq.empty) :+ eInfo
      }.sortBy(eInfo => eInfo.nodeId).map { info =>
        val (childrenExecsStr, nodeIdsStr) = if (info.children.isDefined) {
          (formatStr(
            info.children.get.map(_.exec).mkString(":")),
            formatStr(
              info.children.get.map(_.nodeId).mkString(":")))
        } else {
          (emptyString, emptyString)
        }
        Seq(sqlIDStr, info.execRef.getOpNameCSV,
          formatStr(info.expr),
          if (info.duration.isDefined) info.duration.get.toString else zeroDurationStr,
          info.nodeId.toString,
          if (info.isSupported) booleanTrue else booleanFalse,
          formatStr(info.stages.toSeq.sorted.mkString(":")),
          childrenExecsStr,
          nodeIdsStr,
          if (info.shouldRemove) booleanTrue else booleanFalse,
          if (info.shouldIgnore) booleanTrue else booleanFalse,
          formatStr(info.getOpAction.toString)
        ).mkString(delim)
      }.foreach { r =>
        fWriter.writeLn(r)
      }
    }
  }
}


// Unsupported ops table per-App.
class AppQualUnsupportedOpsTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {

  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    var execIdGenerator = 0
    val appDuration = rec.estimatedInfo.appDur
    val emptyStages: Seq[Int] = Seq(-1)

    val stageInfoMap: Map[Int, String] = {
      rec.origPlanStageInfo.iterator.map { sInfo =>
        sInfo.stageId -> sInfo.stageWallclockDuration.toString
      }.toMap
    }

    def getStageDuration(stageId: Int): String = {
      if (stageId < 0) {
        "0"
      } else {
        stageInfoMap.getOrElse(stageId, "0")
      }
    }

    def constructDetailedUnsupportedRow(
      unSupExecInfo: UnsupportedExecSummary,
      stageId: Int): String = {
      // scalastyle:off line.size.limit
      // P.S: Use raw because it offers better performance compared to sInterpolation.
      raw"${unSupExecInfo.sqlId}$delim$stageId$delim${unSupExecInfo.execId.toString}$delim${unSupExecInfo.finalOpType}$delim${unSupExecInfo.unsupportedOperatorCSVFormat}$delim${unSupExecInfo.detailsCSV}$delim${getStageDuration(stageId)}$delim$appDuration$delim${unSupExecInfo.opAction}"
      // scalastyle:on line.size.limit
    }

    rec.planInfo.foreach { pInfo =>
      // TODO: the following can be optimized by filtering before flattening all children
      pInfo.execInfo.flatMap { e =>
        // need to remove the WholeStageCodegen wrappers since they aren't actual
        // execs that we want to get timings of
        if (e.isClusterNode) {
          e.children.getOrElse(Seq.empty).filter(c => !c.isSupported)
        } else if (!e.isSupported) {
          Seq(e)
        } else {
          Seq.empty
        }
      }.foreach {
        eInfo =>
          val eRows = eInfo.getUnsupportedExecSummaryRecord(execIdGenerator)
          execIdGenerator += 1
          val stageIds = if (eInfo.stages.isEmpty) {
            emptyStages
          } else {
            eInfo.stages
          }
          stageIds.foreach { stageId =>
            eRows.foreach { eRow =>
              fWriter.writeLn(constructDetailedUnsupportedRow(eRow, stageId))
            }
          }
      }
    }
  }
}

// Operators stats table
class AppQualOpsStatsTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {
  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    val supportedCSVStr = "true"
    val unsupportedCSVStr = "false"
    rec.planInfo.foreach { planInfo =>
      val sqlStr = planInfo.sqlID.toString
      OperatorCounter(planInfo).getOpsCountSummary
        .sortBy(oInfo => (-oInfo.opData.count, oInfo.opData.opRef.getOpName))
        .foreach { opInfo =>
          val supF = if (opInfo.isSupported) supportedCSVStr else unsupportedCSVStr
          val stageStr = formatStr(opInfo.opData.stages.mkString(":"))
          // scalastyle:off line.size.limit
          // P.S: Use raw because it offers better performance compared to sInterpolation.
          fWriter.writeLn(
            raw"$sqlStr$delim${opInfo.opData.getOpTypeCSV}$delim${opInfo.opData.getOpNameCSV}$delim${opInfo.opData.count}$delim$supF$delim$stageStr"
          )
          // scalastyle:on line.size.limit
        }
    }
  }
}

// Stages.csv table
class AppQualStagesTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {
  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    rec.stageInfo.sortBy(_.stageId).foreach { sInfo =>
      // scalastyle:off line.size.limit
      // P.S: Use raw because it offers better performance compared to sInterpolation.
      fWriter.writeLn(
        raw"${sInfo.stageId}$delim${sInfo.stageTaskTime}$delim${sInfo.unsupportedTaskDur}$delim${sInfo.estimated}"
      )
      // scalastyle:on line.size.limit
    }
  }
}

// Databricks specific table
class AppQualClusterTagsTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {
  override def generateFileEnabled(rec: QualificationSummaryInfo): Boolean = {
    super.generateFileEnabled(rec) && rec.allClusterTagsMap.nonEmpty
  }
  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    rec.allClusterTagsMap.foreach { case (pKey, pValue) =>
      // P.S: Use raw because it offers better performance compared to sInterpolation.
      fWriter.writeLn(raw"${formatStr(pKey)}$delim${formatStr(pValue)}")
    }
  }
}

// Generated only when ml-functions argument is passed
class AppQualMLFunctionsTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {
  override def generateFileEnabled(rec: QualificationSummaryInfo): Boolean = {
    super.generateFileEnabled(rec) && rec.mlFunctions.nonEmpty
  }
  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    rec.mlFunctions.foreach { mlFuncsOp =>
      mlFuncsOp.foreach { r =>
        // scalastyle:off line.size.limit
        // P.S: Use raw because it offers better performance compared to sInterpolation.
        fWriter.writeLn(raw"${r.stageId}$delim${formatStr(ToolUtils.renderTextField(r.mlOps, ";", delim))}$delim${r.duration}")
        // scalastyle:on line.size.limit
      }
    }
  }
}

// Generated only when ml-functions argument is passed
class AppQualMLFunctionsDurationsTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {
  override def generateFileEnabled(rec: QualificationSummaryInfo): Boolean = {
    super.generateFileEnabled(rec) && rec.mlFunctionsStageDurations.nonEmpty
  }
  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    rec.mlFunctionsStageDurations.foreach { mlFuncsOp =>
      mlFuncsOp.foreach { r =>
        // scalastyle:off line.size.limit
        // P.S: Use raw because it offers better performance compared to sInterpolation.
        fWriter.writeLn(raw"${formatStr(ToolUtils.renderTextField(r.stageIds, ";", delim))}$delim${formatStr(r.mlFuncName)}$delim${r.duration}")
        // scalastyle:on line.size.limit
      }
    }
  }
}

class AppQualPerSqlTable(
    tableMeta: QualOutputTableDefinition,
    override val rootDirectory: String,
    hadoopConf: Configuration) extends AppQualTable(tableMeta, rootDirectory, hadoopConf) {
  // load the maxSQLDescLength from the report-configurations.
  lazy val maxSQLDescLength: Int =
    reportConfigs.getOrElse("column.sql_description.max.length", "100").toInt
  def constructRowFromPerSqlSummary(sumInfo: EstimatedPerSQLSummaryInfo): String = {
    val rootID = sumInfo.rootExecutionID match {
      case Some(id) => formatStr(id.toString)
      case _ => formatStr("")
    }

    val sqlDescr =
      formatStr(
        QualOutputWriter.formatSQLDescription(sumInfo.sqlDesc, maxSQLDescLength, delim))
    // Use raw interpolation which has better performance compared to sInterpolation because it
    // does not process escaped characters.
    // scalastyle:off line.size.limit
    raw"$rootID$delim${sumInfo.sqlID}$delim$sqlDescr$delim${sumInfo.info.sqlDfDuration}$delim${sumInfo.info.gpuOpportunity}"
    // scalastyle:on line.size.limit
  }

  override def generateFileEnabled(rec: QualificationSummaryInfo): Boolean = {
    // only generate the file including the report if the perSQL info is defined.
    super.generateFileEnabled(rec) && rec.perSQLEstimatedInfo.nonEmpty
  }

  override def appendDataToWriter(fWriter: ToolTextFileWriter,
    rec: QualificationSummaryInfo): Unit = {
    rec.perSQLEstimatedInfo match {
      case Some(perSQLInfo) =>
        perSQLInfo.foreach { sumInfo =>
          fWriter.writeLn(constructRowFromPerSqlSummary(sumInfo))
        }
      case None =>
        logWarning("No per SQL summary information found.")
    }
  }
}

/**
 * This is the main object for generating the per application reports.
 * See the readme docs for the structure of the report.
 */
object QualPerAppReportGenerator extends QualReportGeneratorTrait[QualificationSummaryInfo] {
  /**
   * Selects the tables which are per app.
   */
  override val tableSelector: QualOutputTableDefinition => Boolean = { tDef =>
    tDef.isPerApp
  }

  /**
   * Returns the relative directory for the per app reports.
   */
  override def relativeDirectory(parent: String): String = {
    s"$parent/${QualReportGenConfProvider.PER_APP_SUBDIRECTORY}"
  }

  /**
   * Generates the report with the given table definition, root directory, hadoopConf,
   * reportConfigs, and the result of the tool.
   *
   * @param tableMeta the table definition.
   * @param rootDirectory the root directory of the report.
   * @param hadoopConf the hadoop configuration.
   * @param reportConfigs the report configurations. For example, it can contain the configurations
   *                      for the report enabled, report path, and so on.
   * @param qualSumInfo the result of the tool.
   */
  def generateReport(
    tableMeta: QualOutputTableDefinition,
    rootDirectory: String,
    hadoopConf: Configuration,
    reportConfigs: Map[String, String],
    qualSumInfo: QualificationSummaryInfo): Unit = {
    val table = tableMeta.label match {
      case "execCSVReport" =>
        new AppQualExecTable(tableMeta, rootDirectory, hadoopConf)
      case "unsupportedOpsCSVReport" =>
        new AppQualUnsupportedOpsTable(tableMeta, rootDirectory, hadoopConf)
      case "operatorsStatsCSVReport" =>
        new AppQualOpsStatsTable(tableMeta, rootDirectory, hadoopConf)
      case "stagesCSVReport" =>
        new AppQualStagesTable(tableMeta, rootDirectory, hadoopConf)
      case "perSqlCSVReport" =>
        new AppQualPerSqlTable(tableMeta, rootDirectory, hadoopConf)
      case "clusterInfoJSONReport" =>
        new AppQualClusterInfoTable(tableMeta, rootDirectory, hadoopConf)
      case "clusterTagsCSVReport" =>
        new AppQualClusterTagsTable(tableMeta, rootDirectory, hadoopConf)
      case "mlFunctionsCSVReport" =>
        new AppQualMLFunctionsTable(tableMeta, rootDirectory, hadoopConf)
      case "mlFunctionsDurationsCSVReport" =>
        new AppQualMLFunctionsDurationsTable(tableMeta, rootDirectory, hadoopConf)
      case "udfReportJSON" =>
        new AppQualUdfReportTable(tableMeta, rootDirectory, hadoopConf)
      case _ =>
        throw new IllegalArgumentException(
          s"Unknown table label ${tableMeta.label} for table ${tableMeta.description}")
    }
    table.withReportConfigs(reportConfigs, table.getLabel).generateOutputReport(qualSumInfo)
  }
}
