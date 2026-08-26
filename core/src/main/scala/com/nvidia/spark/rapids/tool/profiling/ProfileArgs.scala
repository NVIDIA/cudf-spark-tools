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
package com.nvidia.spark.rapids.tool.profiling

import com.nvidia.spark.rapids.tool.PlatformNames
import org.rogach.scallop.{ScallopConf, ScallopOption}
import org.rogach.scallop.exceptions.ScallopException

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.AppFilterImpl

class ProfileArgs(arguments: Seq[String]) extends ScallopConf(arguments) with Logging {

  banner("""
Profiling tool for the NVIDIA cuDF plugin for Apache Spark

Usage: java -cp rapids-4-spark-tools_2.12-<version>.jar:$SPARK_HOME/jars/*
       com.nvidia.spark.rapids.tool.profiling.ProfileMain [options]
       [eventlogs | eventlog directories ...]
    """)

  val outputDirectory: ScallopOption[String] =
    opt[String](required = false,
      descr = "Base output directory. Default is current directory for the default filesystem." +
        " The final output will go into a subdirectory called" +
        " rapids_4_spark_profile. It will overwrite any existing files" +
        " with the same name.",
      default = Some("."))
  val driverlog: ScallopOption[String] =
    opt[String](required = false,
      descr = "Driver log filename - eg: /path/to/driverlog. Default is empty.")
  val eventlog: ScallopOption[List[String]] =
    trailArg[List[String]](required = false,
      descr = "Event log filenames(space separated) or directories containing event logs." +
          " eg: s3a://<BUCKET>/eventlog1 /path/to/eventlog2")
  val filterCriteria: ScallopOption[String] =
    opt[String](required = false,
      descr = "Filter newest or oldest N eventlogs for processing." +
          "eg: 100-newest-filesystem (for processing newest 100 event logs). " +
          "eg: 100-oldest-filesystem (for processing oldest 100 event logs).")
  val matchEventLogs: ScallopOption[String] =
    opt[String](required = false,
      descr = "Filter event logs whose filenames contain the input string.")
  val numOutputRows: ScallopOption[Int] =
    opt[Int](required = false,
      descr = "Number of output rows for each Application. Default is 1000.")
  val generateDot: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Generate query visualizations in DOT format. Default is false.")
  val printPlans: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Print the SQL physical plan descriptions to a file named 'planDescriptions.log'." +
        " Default is false.")
  val printPlanGraph: ScallopOption[Boolean] =
    opt[Boolean](
      required = false,
      descr = "Print the SQL Spark-Graph of the SQL plans to a file." +
        " Default is false.",
      default = Some(false))
  val platform: ScallopOption[String] =
    opt[String](required = false,
      descr = "Cluster platform where Spark GPU workloads were executed. Options include " +
        s"${PlatformNames.getAllNames.mkString(", ")}. " +
        s"Default is ${PlatformNames.DEFAULT}.",
      default = Some(PlatformNames.DEFAULT))
  val generateTimeline: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Write an SVG graph out for the full application timeline.")
  val noRecursion: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Set to true to disable recursive search for event logs in the provided " +
        "directories. Default is false.",
      default = Some(false))
  val numThreads: ScallopOption[Int] =
    opt[Int](required = false,
      descr = "Number of thread to use for parallel processing. The default is the " +
        "number of cores on host divided by 4.")
  val csv: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Output each table to a CSV file as well creating the summary text file.")
  val connectStatements: ScallopOption[Boolean] =
    toggle("connect-statements",
      default = Some(false),
      prefix = "no-",
      descrYes = "Write Spark Connect statementText sidecar files. Disabled by default.",
      descrNo = "Do not write Spark Connect statementText sidecar files.")
  val timeout: ScallopOption[Long] =
    opt[Long](required = false,
      descr = "Maximum time in seconds to wait for the event logs to be processed. " +
        "Default is 48 hours (172800 seconds) and must be greater than 3 seconds. If it " +
        "times out, it will report what it was able to process up until the timeout.",
      default = Some(172800))
  val startAppTime: ScallopOption[String] =
    opt[String](required = false,
      descr = "Filter event logs whose application start occurred within the past specified " +
        "time period. Valid time periods are min(minute),h(hours),d(days),w(weeks)," +
        "m(months). If a period is not specified it defaults to days.")
  val autoTuner: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Toggle AutoTuner module.",
      default = Some(false))
  val outputSqlIdsAligned: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Output the SQL Ids after being cleaned of delta log metadata operations to " +
        "allow aligning cpu/gpu runs.",
      default = Some(false))
  val enableDiagnosticViews: ScallopOption[Boolean] =
    opt[Boolean](required = false,
      descr = "Toggle diagnostic views generation. Disabled by default.",
      default = Some(false))
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
    case crit if crit.endsWith("-newest-filesystem") ||
        crit.endsWith("-oldest-filesystem") => Right(())
    case _ => Left("Error, the filter criteria must end with either -newest-filesystem " +
        "or -oldest-filesystem")
  }

  validate(timeout) {
    case timeout if timeout > 3 => Right(())
    case _ => Left("Error, timeout must be greater than 3 seconds.")
  }

  validate(startAppTime) {
    case time if AppFilterImpl.parseAppTimePeriod(time) > 0L => Right(())
    case _ => Left("Time period specified, must be greater than 0 and valid periods " +
      "are min(minute),h(hours),d(days),w(weeks),m(months).")
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

  // verify that either driverlog or eventlog is specified
  validateOpt(driverlog, eventlog) {
    case (None, None) => Left("Error, one of driverlog or eventlog must be specified")
    case _ => Right(())
  }
  verify()

  override def onError(e: Throwable): Unit = {
    e match {
      case ScallopException(message) =>
        if (args.contains("--help")) {
          printHelp
          System.exit(0)
        }
        errorMessageHandler(message)
      case ex => super.onError(ex)
    }
  }
}
