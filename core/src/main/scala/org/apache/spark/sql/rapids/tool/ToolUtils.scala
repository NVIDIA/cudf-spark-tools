/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.tool

import scala.util.{Failure, Success, Try}
import scala.util.matching.Regex

import com.nvidia.spark.rapids.tool.Platform
import com.nvidia.spark.rapids.tool.planparser.SubqueryExecParser
import com.nvidia.spark.rapids.tool.planparser.delta.DeltaLakeOps
import com.nvidia.spark.rapids.tool.profiling.ProfileUtils.replaceDelimiter
import com.nvidia.spark.rapids.tool.qualification.QualOutputWriter
import org.apache.maven.artifact.versioning.ComparableVersion

import org.apache.spark.internal.{config, Logging}
import org.apache.spark.sql.rapids.tool.plangraph.SparkPlanGraphNode
import org.apache.spark.sql.rapids.tool.util.SparkRuntime

object ToolUtils extends Logging {
  // List of recommended file-encodings on the GPUs.
  val SUPPORTED_ENCODINGS = Seq("UTF-8")
  // the prefix of keys defined by the cuDF plugin
  val PROPS_RAPIDS_KEY_PREFIX = "spark.rapids"
  // List of keys from sparkProperties that may point to cuDF plugin jars.
  // Note that we ignore "spark.yarn.secondary.jars" for now as it does not include a full path.
  val POSSIBLE_JARS_PROPERTIES = Set("spark.driver.extraClassPath",
    "spark.executor.extraClassPath",
    "spark.yarn.dist.jars",
    "spark.repl.local.jars")
  // matches file paths or names of JAR files related to the cuDF plugin, cuDF, or CUDA:
  // - .*rapids-4-spark.*\.jar: Matches any JAR file whose name contains rapids-4-spark.
  // - .*cuda[0-9]+.*\.jar: Matches any JAR file whose name contains cuda followed by one or more
  //   digits.
  // Examples: rapids-4-spark_2.12-25.08.0.jar; rapids-4-spark_2.12-25.08.0-cuda12-arm64.jar
  val RAPIDS_JAR_REGEX: Regex =
    "(.*rapids-4-spark.*\\.jar)|(.*cuda[0-9]+.*\\.jar)".r

  // Add more entries to this lookup table as necessary.
  // There is no need to list all supported versions.
  private val lookupVersionsMap = Map(
    "320" -> new ComparableVersion("3.2.0"), // introduced reusedExchange
    "330" -> new ComparableVersion("3.3.0"), // used to check for memoryOverheadFactor
    "331" -> new ComparableVersion("3.3.1"),
    "340" -> new ComparableVersion("3.4.0"), // introduces jsonProtocolChanges
    "350" -> new ComparableVersion("3.5.0"), // introduces windowGroupLimit
    "356" -> new ComparableVersion("3.5.6"),
    "357" -> new ComparableVersion("3.5.7")  // default build version
  )

  // Property to check the spark runtime version. We need this outside of test module as we
  // extend the support runtime for different platforms such as Databricks.
  lazy val sparkRuntimeVersion: String = {
    org.apache.spark.SPARK_VERSION
  }

  /**
   * Compare two version strings using
   * [[org.apache.maven.artifact.versioning.ComparableVersion#compareTo]]. The wrapped
   * `Int` follows the standard `compareTo` sign convention (negative / zero / positive).
   * Returns `None` if either string fails to parse, so callers cannot confuse a parse
   * failure with version equality.
   */
  def compareVersions(verA: String, verB: String): Option[Int] = {
    Try {
      val verObjA = new ComparableVersion(verA)
      val verObjB = new ComparableVersion(verB)
      verObjA.compareTo(verObjB)
    } match {
      case Success(compRes) => Some(compRes)
      case Failure(t) =>
        logError(s"exception comparing two versions [$verA, $verB]", t)
        None
    }
  }

  def runtimeIsSparkVersion(refVersion: String): Boolean = {
    compareVersions(refVersion, sparkRuntimeVersion).contains(0)
  }

  private def compareToSparkVersion(currVersion: String, lookupVersion: String): Int = {
    val lookupVersionObj = lookupVersionsMap(lookupVersion)
    val currVersionObj = new ComparableVersion(currVersion)
    currVersionObj.compareTo(lookupVersionObj)
  }

  def isSpark320OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "320") >= 0
  }

  def isSpark330OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "330") >= 0
  }

  def isSpark331OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "331") >= 0
  }

  def isSpark340OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "340") >= 0
  }

  def isSpark350OrLater(sparkVersion: String = sparkRuntimeVersion): Boolean = {
    compareToSparkVersion(sparkVersion, "350") >= 0
  }

  def isPluginEnabled(properties: Map[String, String]): Boolean = {
    val hasPlugin = properties.getOrElse(config.PLUGINS.key, "")
      .contains("com.nvidia.spark.SQLPlugin")
    val isEnabled = Try {
      properties.getOrElse("spark.rapids.sql.enabled", "true").toBoolean
    }.getOrElse(true) // Default to true if parsing fails, matching the default value
    hasPlugin && isEnabled
  }

  /**
   * Calculate the duration percent given the numerator and total values.
   * This is used to calculate the CPURatio which represents the percentage of CPU time to
   * the runTime.
   * There is an implicit check to ensure that the denominator is not zero. If it is, then the
   * ratio will be set to 0.
   * There is an option to force the cap to 100% if the calculated value is greater
   * than the total. This is possible to happen because the tasks CPUTime is measured in
   * nanoseconds, while the runtTime is measured in milliseconds. This leads to a loss of precision
   * causing the total percentage to exceed 100%.
   * @param numerator the numerator value.
   * @param total the total value.
   * @param forceCap if true, then the value is capped at 100%.
   * @return the calculated percentage.
   */
  def calculateDurationPercent(numerator: Long, total: Long, forceCap: Boolean = true): Double = {
    if (numerator == 0 || total == 0) {
      0.toDouble
    } else {
      val numeratorDec = BigDecimal.decimal(numerator)
      val totalDec = BigDecimal.decimal(total)
      val res = formatDoubleValue((numeratorDec / totalDec) * 100, 2)
      if (forceCap) {
        math.min(res, 100)
      } else {
        res
      }
    }
  }

  // given to duration values, calculate a human average
  // rounded to specified number of decimal places.
  def calculateAverage(first: Double, size: Long, places: Int): Double = {
    val firstDec = BigDecimal.decimal(first)
    val sizeDec = BigDecimal.decimal(size)
    if (firstDec == 0 || sizeDec == 0) {
      0.toDouble
    } else {
      val res = (firstDec / sizeDec)
      formatDoubleValue(res, places)
    }
  }

  def formatDoubleValue(bigValNum: BigDecimal, places: Int): Double = {
    bigValNum.setScale(places, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  def formatDoublePrecision(valNum: Double): String = {
    truncateDoubleToTwoDecimal(valNum).toString
  }

  def truncateDoubleToTwoDecimal(valNum: Double): Double = {
    // floor is applied after multiplying by 100. This keeps the number "as is" up-to two decimal.
    math.floor(valNum * 100) / 100
  }

  /**
   * Converts a sequence of elements to a single string that can be appended to a formatted text.
   * Delegates to [[com.nvidia.spark.rapids.tool.profiling.ProfileUtils.replaceDelimiter]] to
   * replace what is used as a text delimiter with something else.
   *
   * @param values the sequence of elements to join together.
   * @param separator the separator string to use.
   * @param txtDelimiter the delimiter used by the output file format (i.e., comma for CSV).
   * @return a string representation of the input sequence value. In the resulting string the string
   *         representations (w.r.t. the method toString) of all elements are separated by
   *         the string sep.
   */
  def renderTextField(values: Seq[Any], separator: String, txtDelimiter: String): String = {
    replaceDelimiter(values.mkString(separator), txtDelimiter)
  }

  def formatComplexTypes(
      values: Seq[String], fileDelimiter: String = QualOutputWriter.CSV_DELIMITER): String = {
    renderTextField(values, ";", fileDelimiter)
  }

  def formatPotentialProblems(
      values: Seq[String], fileDelimiter: String = QualOutputWriter.CSV_DELIMITER): String = {
    renderTextField(values, ":", fileDelimiter)
  }

  /**
   * Given a Spark property key, this predicate checks whether it is related to cuDF plugin
   * configurations.
   * Note that, "cuDF plugin-related properties" do not always have 'spark.rapids' prefix.
   *
   * @param sparkPropKey the spark property key
   * @return True if it is directly related to the cuDF plugin
   */
  def isRapidsPropKey(pKey: String): Boolean = {
    pKey.startsWith(PROPS_RAPIDS_KEY_PREFIX) || pKey.startsWith("spark.executorEnv.UCX") ||
      pKey.startsWith("spark.shuffle.manager") || pKey.equals("spark.shuffle.service.enabled")
  }

  /**
   * Checks if the given value is supported for all Ops or not.
   * @param fileEncoding the value being read from the Application configs
   * @return True if file encoding is supported
   */
  def isFileEncodingRecommended(fileEncoding: String): Boolean = {
    fileEncoding.matches("(?i)utf-?8")
  }

  /**
   * Collects the paths that points to cuDF plugin jars in a map of properties.
   * @param properties the map of properties to holding the app configuration.
   * @return set of unique file paths that matches cuDF plugin jars patterns.
   */
  def extractRAPIDSJarsFromProps(properties: collection.Map[String, String]): Set[String] = {
    properties.filterKeys(POSSIBLE_JARS_PROPERTIES.contains(_)).collect {
      case (_, pVal) if pVal.matches(RAPIDS_JAR_REGEX.regex) =>
        pVal.split(",").filter(_.matches(RAPIDS_JAR_REGEX.regex))
    }.flatten.toSet
  }
}

object JoinType {
  val Inner = "Inner"
  val Cross = "Cross"
  val LeftOuter = "LeftOuter"
  val RightOuter = "RightOuter"
  val FullOuter = "FullOuter"
  val LeftSemi = "LeftSemi"
  val LeftAnti = "LeftAnti"
  val ExistenceJoin = "ExistenceJoin"

  val supportedJoinTypeForBuildRight = Set(Inner, Cross, LeftOuter, LeftSemi,
    LeftAnti, FullOuter, ExistenceJoin)

  val supportedJoinTypeForBuildLeft = Set(Inner, Cross, RightOuter, FullOuter)

  val allsupportedJoinType = Set(Inner, Cross, LeftOuter, RightOuter, FullOuter, LeftSemi,
    LeftAnti, ExistenceJoin)
}

object BuildSide {
  val BuildLeft = "BuildLeft"
  val BuildRight = "BuildRight"
  // TODO: This is a workaround to deal with native converters in Auron.
  //       We need a better way to configure this for each added parser.
  val BroadcastLeft = "BroadcastLeft"
  val BroadcastRight = "BroadcastRight"
  // Currently, we use only those 2 buildSides relying on the fact that the OssOpMapper
  // converts Auron keywords to OSS equivalents.
  val allBuildSides = Set(BuildLeft, BuildRight)

  val supportedBuildSides =
    Map(
      BuildLeft -> JoinType.supportedJoinTypeForBuildLeft,
      BuildRight -> JoinType.supportedJoinTypeForBuildRight
    )
}



case class RDDCheckResult(
    nodeNameRDD: Boolean,
    nodeDescRDD: Boolean,
    expr: Set[String] = Set.empty) {
  def isRDD: Boolean = nodeNameRDD || nodeDescRDD
}

object RDDCheckHelper {
  // regular expression to search for RDDs in node descriptions
  private val dataSetRDDRegExDescLookup = Set(
    ".*\\$Lambda\\$.*".r,
    ".*\\.apply$".r
  )
  // regular expression to search for RDDs in node names
  private val dataSetOrRDDRegExLookup = Set(
    "ExistingRDD$".r,
    "^Scan ExistingRDD.*".r,
    "SerializeFromObject$".r,
    "DeserializeToObject$".r,
    "MapPartitions$".r,
    "MapElements$".r,
    "AppendColumns$".r,
    "AppendColumnsWithObject$".r,
    "MapGroups$".r,
    "FlatMapGroupsInR$".r,
    "FlatMapGroupsInRWithArrow$".r,
    "CoGroup$".r
  )

  def isDatasetOrRDDPlan(nodeName: String, nodeDesc: String): RDDCheckResult = {
    val nodeNameRdd = dataSetOrRDDRegExLookup.exists(regEx => nodeName.trim.matches(regEx.regex))
    // For optimization purpose, we do not want to search for matches inside node description
    // if it is not necessary.
    val nodeDescRdd = !nodeNameRdd &&
      dataSetRDDRegExDescLookup.exists(regEx => nodeDesc.matches(regEx.regex))
    // TODO: catch the expressions that match the regular expression so we can pass it later to
    //       the reporting
    RDDCheckResult(nodeNameRdd, nodeDescRdd)
  }
}


object ExecHelper {
  private val UDFRegExLookup = Set(
    ".*UDF.*".r
  )

  // Set containing execs that should be labeled as "shouldRemove"
  private val execsToBeRemoved = Set(
    "GenerateBloomFilter",      // Exclusive on AWS. Ignore it as metrics cannot be evaluated.
    "ReusedExchange",           // reusedExchange should not be added to speedups
    "ColumnarToRow",            // for now, assume everything is columnar
    // Our customer-integration team requested this to be added to the list of execs to be removed.
    "ResultQueryStage",
    // AdaptiveSparkPlan is not a real exec. It is a wrapper for the whole plan.
    // Our customer-integration team requested this to be added to the list of execs to be removed.
    "AdaptiveSparkPlan",          // according to request from our customer facing team
    SubqueryExecParser.execName,  // Subquery represents a simple collect
    // TODO: This is a hack now for Auron ConvertToNative. Ideally we should define this as part of
    //  the parser for Auron. Since, there is some considerable work involved to set that up, we
    //  are adding this here for now.
    // ConvertToNative a specific exec to Auron. This exec should be removed from the plan when
    // parsing it because it does not represent any actual operation for the GPU.
    "ConvertToNative"
  )

  def isUDF(node: SparkPlanGraphNode): Boolean = {
    UDFRegExLookup.exists(regEx => node.desc.matches(regEx.regex))
  }

  def shouldBeRemoved(nodeName: String): Boolean = {
    execsToBeRemoved.contains(nodeName)
  }

  ///////////////////////////////////////////
  // Start definitions of blank execs supported by the plugin
  val EX_NAME_WRITE_FILES: String = "WriteFiles"
  val LIST_OF_SUPPORTED_BLANK_EXECS: Set[String] = Set(
    EX_NAME_WRITE_FILES
  )

  ///////////////////////////////////////////
  // start definitions of execs to be ignored
  // Collect Limit replacement can be slower on the GPU. Disabled by default.
  private val CollectLimit = "CollectLimit"
  // Some DDL's and table commands which can be ignored
  private val ExecuteCreateViewCommand = "Execute CreateViewCommand"
  private val LocalTableScan = "LocalTableScan"
  private val ExecuteCreateDatabaseCommand = "Execute CreateDatabaseCommand"
  private val ExecuteDropDatabaseCommand = "Execute DropDatabaseCommand"
  private val ExecuteCreateTableAsSelectCommand = "Execute CreateTableAsSelectCommand"
  private val ExecuteCreateTableCommand = "Execute CreateTableCommand"
  private val ExecuteDropTableCommand = "Execute DropTableCommand"
  private val ExecuteCreateDataSourceTableAsSelectCommand = "Execute " +
    "CreateDataSourceTableAsSelectCommand"
  private val SetCatalogAndNamespace = "SetCatalogAndNamespace"
  private val ExecuteSetCommand = "Execute SetCommand"
  private val ResultQueryStage = "ResultQueryStage"
  private val ExecAddJarsCommand = "Execute AddJarsCommand"
  private val ScanJDBCRelation = "Scan JDBCRelation"
  private val ScanOneRowRelation = "Scan OneRowRelation"
  private val CommandResult = "CommandResult"
  private val ExecuteAlterTableRecoverPartitionsCommand =
    "Execute AlterTableRecoverPartitionsCommand"
  private val ExecuteCreateFunctionCommand = "Execute CreateFunctionCommand"
  private val CreateHiveTableAsSelectCommand = "Execute CreateFunctionCommand"
  private val ExecuteDeleteCommand = "Execute DeleteCommand"
  private val ExecuteDescribeTableCommand = "Execute DescribeTableCommand"
  private val ExecuteRefreshTable = "Execute RefreshTable"
  private val ExecuteRepairTableCommand = "Execute RepairTableCommand"
  private val ExecuteShowPartitionsCommand = "Execute ShowPartitionsCommand"
  private val ExecuteClearCacheCommand = "Execute ClearCacheCommand"

  def getAllIgnoreExecs: Set[String] = Set(CollectLimit,
    ExecuteCreateViewCommand, LocalTableScan, ExecuteCreateTableCommand,
    ExecuteDropTableCommand, ExecuteCreateDatabaseCommand, ExecuteDropDatabaseCommand,
    ExecuteCreateTableAsSelectCommand, ExecuteCreateDataSourceTableAsSelectCommand,
    SetCatalogAndNamespace, ExecuteSetCommand,
    ResultQueryStage,
    ExecAddJarsCommand,
    ScanJDBCRelation,
    ScanOneRowRelation,
    CommandResult,
    DeltaLakeOps.execDescribeDeltaHistoryCMD,
    DeltaLakeOps.execShowPartitionsDeltaCMD,
    ExecuteAlterTableRecoverPartitionsCommand,
    ExecuteCreateFunctionCommand,
    CreateHiveTableAsSelectCommand,
    ExecuteDeleteCommand,
    ExecuteDescribeTableCommand,
    ExecuteRefreshTable,
    ExecuteRepairTableCommand,
    ExecuteShowPartitionsCommand,
    ExecuteClearCacheCommand,
    SubqueryExecParser.execName
  )

  def shouldIgnore(execName: String): Boolean = {
    getAllIgnoreExecs.contains(execName)
  }
}

object MlOps {
  val sparkml = "spark.ml."
  val xgBoost = "spark.XGBoost"
  val pysparkLog = "py4j.GatewayConnection.run" // pyspark eventlog contains py4j
}

object MlOpsEventLogType {
  val pyspark = "pyspark"
  val scala = "scala"
}

object SupportedMLFuncsName {
  val funcName: Map[String, String] = Map(
    "org.apache.spark.ml.clustering.KMeans.fit" -> "KMeans",
    "org.apache.spark.ml.feature.PCA.fit" -> "PCA",
    "org.apache.spark.ml.regression.LinearRegression.train" -> "LinearRegression",
    "org.apache.spark.ml.classification.RandomForestClassifier.train" -> "RandomForestClassifier",
    "org.apache.spark.ml.regression.RandomForestRegressor.train" -> "RandomForestRegressor",
    "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier.train" -> "XGBoost"
  )
}

class AppEventlogProcessException(message: String) extends Exception(message)

case class GpuEventLogException(
    message: String = "Cannot parse event logs from GPU run: skipping this file")
    extends AppEventlogProcessException(message)

case class StreamingEventLogException(
    message: String = "Encountered Spark Structured Streaming Job: skipping this file!")
    extends AppEventlogProcessException(message)

case class IncorrectAppStatusException(
    message: String = "Application status is incorrect. Missing AppInfo")
    extends AppEventlogProcessException(message)

case class UnsupportedMetricNameException(metricName: String)
    extends AppEventlogProcessException(
      s"Unsupported metric name found in the event log: $metricName")

case class UnsupportedSparkRuntimeException(
    platform: Platform,
    sparkRuntime: SparkRuntime.SparkRuntime)
    extends AppEventlogProcessException(
     s"Platform '${platform.platformName}' does not support the runtime '$sparkRuntime'")

class InvalidMemoryUnitFormatException(message: String)
  extends IllegalArgumentException(message)

class MatchingInstanceTypeNotFoundException(message: String)
  extends Exception(message)

// Case class to represent a failed AppInfo creation
case class FailureApp(
    status: String,
    message: String
)
