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
package com.nvidia.spark.rapids.tool.qualification

import com.nvidia.spark.rapids.tool.PlatformNames
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.rogach.scallop.exceptions.ScallopException

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.AppFilterImpl

class QualificationArgs(arguments: Seq[String]) extends ScallopConf(arguments) with Logging {

  banner("""
NVIDIA cuDF plugin for Apache Spark qualification tool

Usage: java -cp rapids-4-spark-tools_2.12-<version>.jar:$SPARK_HOME/jars/*
       com.nvidia.spark.rapids.tool.qualification.QualificationMain [options]
       <eventlogs | eventlog directories ...>
    """)

  val outputDirectory: ScallopOption[String] =
    opt[String](required = false,
      descr = "Base output directory. Default is current directory for the default filesystem." +
        " The final output will go into a subdirectory called" +
        " qual_core_output. It will overwrite any existing directory" +
        " with the same name.",
      default = Some("."))
  val eventlog: ScallopOption[List[String]] =
    trailArg[List[String]](required = true,
      descr = "Event log filenames(space separated) or directories containing event logs." +
          " eg: s3a://<BUCKET>/eventlog1 /path/to/eventlog2")
  val filterCriteria: ScallopOption[String] =
    opt[String](required = false,
      descr = "Filter newest or oldest N eventlogs based on application start timestamp, " +
          "unique application name or filesystem timestamp. Filesystem based filtering " +
          "happens before any application based filtering." +
          "For application based filtering, the order in which filters are" +
          "applied is: application-name, start-app-time, filter-criteria." +
          "Application based filter-criteria are:" +
          "100-newest (for processing newest 100 event logs based on timestamp inside" +
          "the eventlog) i.e application start time)  " +
          "100-oldest (for processing oldest 100 event logs based on timestamp inside" +
          "the eventlog) i.e application start time)  " +
          "100-newest-per-app-name (select at most 100 newest log files for each unique " +
          "application name) " +
          "100-oldest-per-app-name (select at most 100 oldest log files for each unique " +
          "application name)" +
          "Filesystem based filter criteria are:" +
          "100-newest-filesystem (for processing newest 100 event logs based on filesystem " +
          "timestamp). " +
          "100-oldest-filesystem (for processing oldest 100 event logs based on filesystem " +
          "timestamp).")
  val applicationName: ScallopOption[String] =
    opt[String](required = false,
      descr = "Filter event logs by application name. The string specified can be a " +
          "regular expression, substring, or exact match. For filtering based on complement " +
          "of application name, use ~APPLICATION_NAME. i.e Select all event logs except the ones " +
          "which have application name as the input string.")
  val any: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Apply multiple event log filtering criteria and process only logs for which any " +
          "condition is satisfied." +
          "Example: <Filter1> <Filter2> <Filter3> --any -> result is <Filter1> OR " +
          "<Filter2> OR <Filter3>")
  val all: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Apply multiple event log filtering criteria and process only logs for which all " +
          "conditions are satisfied." +
          "Example: <Filter1> <Filter2> <Filter3> --all -> result is <Filter1> AND " +
          "<Filter2> AND <Filter3>. Default is all=true")
  val startAppTime: ScallopOption[String] =
    opt[String](required = false,
      descr = "Select event logs whose application start occurred within the past specified " +
        "time period. Valid time periods are min(minute),h(hours),d(days),w(weeks)," +
        "m(months). If a period is not specified it defaults to days.")
  val maxEventLogSize: ScallopOption[String] =
    opt[String](required = false,
      descr = "Process only application event logs whose size is less than or equal to the size " +
        "specified. Valid units of size are b(bytes),k(kilobytes),m(megabytes),g(gigabytes). " +
        "If no units are specified, the size is assumed to be m. Note, this does not support " +
        "event log rolling which puts multiple event logs for the same application into a " +
        "single directory.")
  val minEventLogSize: ScallopOption[String] =
    opt[String](required = false,
      descr = "Process only application event logs whose size is greater than or equal to the " +
        "size specified. Valid units of size are " +
        "b(bytes),k(kilobytes),m(megabytes),g(gigabytes). If no units are specified, the " +
        "size is assumed to be m. Note, this does not support event log rolling which puts " +
        "multiple event logs for the same application into a single directory.")
  val matchEventLogs: ScallopOption[String] =
    opt[String](required = false,
      descr = "Filter event logs whose filenames contain the input string. Filesystem " +
              "based filtering happens before any application based filtering.")
  val noRecursion: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Set to true to disable recursive search for event logs in the provided " +
        "directories. Default is false.",
      default = Some(false))
  val numThreads: ScallopOption[Int] =
    opt[Int](required = false,
      descr = "Number of thread to use for parallel processing. The default is the " +
        "number of cores on host divided by 4.")
  val fsStartTime: ScallopOption[String] =
    opt[String](required = false,
      descr = "Select event logs whose filesystem time stamp is newer than or starts on the " +
        "date/time specified. Valid format is yyyy-MM-dd hh:mm:ss.")
  val fsEndTime: ScallopOption[String] =
    opt[String](required = false,
      descr = "Select event logs whose filesystem time stamp is older than or ends on the " +
        "date/time specified. Valid format is yyyy-MM-dd hh:mm:ss.")
  val mlFunctions: ScallopOption[Boolean] =
    toggle("ml-functions",
      default = Some(false),
      prefix = "no-",
      descrYes = "Parse ML functions in the eventlog and generate a report. Disabled by default.",
      descrNo = "Do not parse ML functions in the eventlog.")
  val penalizeTransitions: ScallopOption[Boolean] =
    toggle("penalize-transitions",
      default = Some(true),
      prefix = "no-",
      descrYes = "Add penalty for ColumnarToRow and RowToColumnar transitions. " +
        "Enabled by default.",
      descrNo = "Do not add penalty for ColumnarToRow and RowToColumnar transitions.")
  val sparkProperty: ScallopOption[List[String]] =
    opt[List[String]](required = false,
      descr = "Filter applications based on certain Spark properties that were set during " +
          "launch of the application. It can filter based on key:value pair or just based on " +
          "keys. Multiple configs can be provided where the filtering is done if any of the" +
          "config is present in the eventlog. " +
          "filter on specific configuration: --spark-property=spark.eventLog.enabled:true" +
          "filter all eventlogs which has config: --spark-property=spark.driver.port" +
          "Multiple configs: --spark-property=spark.eventLog.enabled:true " +
          "--spark-property=spark.driver.port")
  val timeout: ScallopOption[Long] =
    opt[Long](required = false,
      descr = "Maximum time in seconds to wait for the event logs to be processed. " +
        "Default is 48 hours (172800 seconds) and must be greater than 3 seconds. If it " +
        "times out, it will report what it was able to process up until the timeout.",
      default = Some(172800))
  val userName: ScallopOption[String] =
    opt[String](required = false,
      descr = "Applications which a particular user has submitted." )
  val perSql : ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Report at the individual SQL query level.")
  val connectStatements: ScallopOption[Boolean] =
    toggle("connect-statements",
      default = Some(false),
      prefix = "no-",
      descrYes = "Write Spark Connect statementText sidecar files in raw_metrics. " +
        "Disabled by default.",
      descrNo = "Do not write Spark Connect statementText sidecar files.")
  val maxSqlDescLength: ScallopOption[Int] =
    opt[Int](required = false,
      descr = "Maximum length of the SQL description string output with the " +
        "per sql output. Default is 100.",
      default = Some(100))
  val platform: ScallopOption[String] =
    opt[String](required = false,
      descr = "Cluster platform where Spark CPU workloads were executed. Options include " +
        s"${PlatformNames.getAllNames.mkString(", ")}. " +
        s"Default is ${PlatformNames.DEFAULT}.",
      default = Some(PlatformNames.DEFAULT))
  val speedupFactorFile: ScallopOption[String] =
    opt[String](required = false,
      descr = "Custom speedup factor file used to get estimated GPU speedup that is specific " +
        "to the user's environment. If the file is not provided, it defaults to use the " +
        "speedup files included in the jar.")
  val autoTuner: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Toggle AutoTuner module.",
      default = Some(false))
  val clusterReport: ScallopOption[Boolean] =
    toggle("cluster-report",
      default = Some(true),
      prefix = "no-",
      descrYes = "Generate a cluster information file. Enabled by default.",
      descrNo = "Do not generate the cluster information file.")
  val targetClusterInfo: ScallopOption[String] =
    opt[String](required = false,
      descr = "File path containing the system information of the target cluster")
  val tuningConfigs: ScallopOption[String] =
    opt[String](
      required = false,
      descr =
        "File path for tuning configs to override defaults. " +
        "If not set, default configs are used. " +
        "Default: <repo>/core/src/main/resources/bootstrap/tuningConfigs.yaml"
    )

  validate(filterCriteria) {
    case crit if (crit.endsWith("-newest-filesystem") || crit.endsWith("-oldest-filesystem")
        || crit.endsWith("-newest-per-app-name") || crit.endsWith("-oldest-per-app-name")
        || crit.endsWith("-oldest") || crit.endsWith("-newest")) => Right(())
    case _ => Left("Error, the filter criteria must end with -newest, -oldest, " +
        "-newest-filesystem, -oldest-filesystem, -newest-per-app-name or -oldest-per-app-name")
  }

  validate(timeout) {
    case timeout if (timeout > 3) => Right(())
    case _ => Left("Error, timeout must be greater than 3 seconds.")
  }

  validate(startAppTime) {
    case time if (AppFilterImpl.parseAppTimePeriod(time) > 0L) => Right(())
    case _ => Left("Time period specified, must be greater than 0 and valid periods " +
      "are min(minute),h(hours),d(days),w(weeks),m(months).")
  }

  validate(fsStartTime) {
    case dateTime if (AppFilterImpl.parseDateTimePeriod(dateTime).isDefined) => Right(())
    case _ => Left("Time period specified must be format yyyy-MM-dd hh:mm:ss.")
  }

  validate(fsEndTime) {
    case dateTime if (AppFilterImpl.parseDateTimePeriod(dateTime).isDefined) => Right(())
    case _ => Left("Time period specified must be format yyyy-MM-dd hh:mm:ss")
  }

  /**
   * Assumption: Event-log files do not begin with a '-'.
   *
   * Inform the user that all options after the event-logs will be ignored.
   * Eg,  Arguments: '--output-directory result_dir /path-to-log --timeout 50 --num-threads 100'
   * Result: Option '--timeout 50 --num-threads 100' will be ignored.
   */
  validate(eventlog) { log =>
    // `log/eventlog` is a trailing argument.
    // Drop all elements in it until the first occurrence of '-'.
    val ignoredOptions = log.dropWhile(s => !s.startsWith("-"))
    // If any elements exist in the `ignored` list, these are additional options that
    // will be skipped by EventLogPathProcessor
    if (ignoredOptions.nonEmpty) {
      logWarning(s"Options provided after event logs will be ignored: " +
        s"${ignoredOptions.mkString(" ")}")
    }
    Right(())
  }

  verify()

  override def onError(e: Throwable) = e match {
    case ScallopException(message) =>
      if (args.contains("--help")) {
        printHelp
        System.exit(0)
      }
      errorMessageHandler(message)
    case ex => super.onError(ex)
  }
}

object QualificationArgs {
  def isOrderAsc(order: String): Boolean = {
    order.toLowerCase.startsWith("asc")
  }

  def isOrderDesc(order: String): Boolean = {
    order.toLowerCase.startsWith("desc")
  }
}
