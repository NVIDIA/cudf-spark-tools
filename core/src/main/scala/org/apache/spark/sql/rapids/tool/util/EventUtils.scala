/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.tool.util

import java.lang.reflect.InvocationTargetException

import scala.collection.mutable
import scala.util.{Failure, Success, Try}
import scala.util.control.NonFatal
import scala.util.matching.Regex

import org.json4s.JInt
import org.json4s.jackson.JsonMethods.parse

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.ui.{SparkListenerSQLExecutionEnd, SparkListenerSQLExecutionStart}

/**
 * Utility containing the implementation of helpers used for parsing data from event.
 */
object EventUtils extends Logging {
  // A valid Spark catalog name must start with a letter, can contain letters, digits,
  // underscores, or dashes, and must not end with a dot.
  val SPARK_CATALOG_REGEX: Regex = """spark\\.sql\\.catalog\\.([A-Za-z][A-Za-z0-9_-]*)$""".r

  /**
   * Reads Spark 4's optional application exit code without depending on Spark 4 classes.
   *
   * TODO: Remove this JSON fallback and use SparkListenerApplicationEnd.exitCode when all
   * supported Tools builds use Spark 4 or later.
   * See https://github.com/NVIDIA/cudf-spark-tools/issues/1927.
   */
  def readApplicationExitCode(line: String): Option[Int] = {
    Try(parse(line) \ "ExitCode").toOption.collect {
      case JInt(value) if value.isValidInt => value.toInt
    }
  }

  // Set to keep track of missing classes
  private val missingEventClasses = mutable.HashSet[String]()

  // This is the prefix of the assertion error message we want to track when IPV6 is not valid.
  private val IPV6_ASSERTION_ERR_PREFIX =
    "assertion failed: Expected hostname or IPv6 IP enclosed in [] but got"
  // A cache to avoid reporting the same unknown exception more than once
  // The key is the exception message, and the value is the count of occurences
  private val cachedUnknownExceptions = new java.util.concurrent.ConcurrentHashMap[String, Int]()

  private def reportMissingEventClass(className: String): Unit = {
    if (!missingEventClasses.contains(className)) {
      missingEventClasses.add(className)
      logWarning(s"ClassNotFoundException while parsing an event: $className")
    }
  }

  /**
   * Handles AssertionError exceptions, specifically to catch and log the
   * "assertion failed: expected hostname or IPv6" error only once. If the error keeps occuring we
   * only report it once.
   * @param t the Throwable to handle
   */
  private def handleAssertionError(t: Throwable): Unit = {
    // report assertion failed: expected hostname or IPv6 only once
    val (keyErrMsg, dumpStack) = if (t.getMessage.startsWith(IPV6_ASSERTION_ERR_PREFIX)) {
      // use a constant key to track this specific error and suppress the entire stack.
      (IPV6_ASSERTION_ERR_PREFIX, false)
    } else {
      (t.getMessage, true)
    }
    val errCount =
      cachedUnknownExceptions.compute(keyErrMsg, (_, v) => Option(v).map(_ + 1).getOrElse(1))
    if (errCount == 1) {
      // only report the first occurrence
      val errMessage = "Assertion Error encountered while parsing an event. " +
        "Only the first occurrence will be logged; subsequent similar errors will be suppressed " +
        "for clarity.\n\t\t"
      if (dumpStack) {
        logError(errMessage, t)
      } else {
        logError(errMessage + s"${t.getMessage}")
      }
    }
  }

  // Legacy spark version used to have different Metric names. This map is to transform those
  // old names into new names.
  private val SPARK_LEGACY_METRICS_MAP = Map(
    // from pull https://github.com/apache/spark/pull/23551/files
    "number of files" -> "number of files read", // type sum
    "metadata time (ms)" -> "metadata time", // type sum spark2.x, but it was fixed to be timing
    "time to build (ms)" -> "time to build", // type timing
    "time to broadcast (ms)" -> "time to broadcast", // type timing
    "total time to update rows" -> "time to update",
    "total time to remove rows" -> "time to remove",
    "bytes of written output" -> "written output", // type sum
    // v2.0.0+
    "spill size total (min, med, max)" -> "spill size", // type size
    "peak memory total (min, med, max)" -> "peak memory", // type size
    "aggregate time total (min, med, max)" -> "time in aggregation build", // type timing
    "avg hash probe (min, med, max)" -> "avg hash probe bucket list iters", // type average
    "duration total (min, med, max)" -> "duration", // type timing
    "data size total (min, med, max)" -> "data size", // type size
    "sort time total (min, med, max)" -> "sort time", // type timing
    "scan time total (min, med, max)" -> "scan time", // type timing
    // v3.0.0 (https://github.com/apache/spark/commit/dddfeca175bdce5294debe00d4a993daef92ca60)
    "spill size total (min, med, max (stageId (attemptId): taskId))" -> "spill size", // type size
    "peak memory total (min, med, max (stageId (attemptId): taskId))" -> "peak memory", // type size
    "aggregate time total (min, med, max (stageId (attemptId): taskId))" ->
      "time in aggregation build", // type timing
    "avg hash probe (min, med, max (stageId (attemptId): taskId))" ->
      "avg hash probe bucket list iters", // type average
    "duration total (min, med, max (stageId (attemptId): taskId))" -> "duration", // type timing
    "data size total (min, med, max (stageId (attemptId): taskId))" -> "data size", // type size
    "sort time total (min, med, max (stageId (attemptId): taskId))" -> "sort time", // type timing
    "scan time total (min, med, max (stageId (attemptId): taskId))" -> "scan time", // type timing
    // v3.1.0 (https://github.com/apache/spark/commit/bc37fdc77130ce4f60806db0bb2b1b8914452040)
    "spill size total (min, med, max (stageId: taskId))" -> "spill size", // type size
    "peak memory total (min, med, max (stageId: taskId))" -> "peak memory", // type size
    "aggregate time total (min, med, max (stageId: taskId))" ->
      "time in aggregation build", // type timing
    "avg hash probe (min, med, max (stageId: taskId))" ->
      "avg hash probe bucket list iters", // type average
    "duration total (min, med, max (stageId: taskId))" -> "duration", // type timing
    "data size total (min, med, max (stageId: taskId))" -> "data size", // type size
    "sort time total (min, med, max (stageId: taskId))" -> "sort time", // type timing
    "scan time total (min, med, max (stageId: taskId))" -> "scan time", // type timing
    // Finally, in V3.1.0,
    // PR https://github.com/apache/spark/commit/c4e98c065c99d2cf840e6006ee5414fbaaba9937
    // introduced the current format in recent Spark versions.
    "aggregate time" -> "time in aggregation build",
    // v3.2.0 -> https://github.com/apache/spark/commit/dd677770d88324bdbac13e48ecb9597c0f176a81
    "number of tasks fall-backed to sort-based aggregation" -> "number of sort fallback tasks",
    // Misc: I cannot find the occurrence but it was used in Qual code
    "time to build hash map total" -> "time to build hash map",
    "time in aggregation build total" -> "time in aggregation build"
  )

  /**
   * Maps tool names (simple class name) to the accepted line prefixes for each tool.
   * This gets initialized once to avoid reconstructing the list for each single eventlog.
   */
  private lazy val acceptedLinesToolMap: Map[String, List[String]] = {
    val toolNames = Seq(
      "ApplicationInfo", "QualificationAppInfo", "FilterAppInfo"
    )
    toolNames.map(toolName => toolName -> EventLogReaderConf.getAcceptedLines(toolName)).toMap
  }

  /**
   * Legacy spark versions (i.e., 2.3.x) used to have different metric names. We need to call this
   * method before consuming the metric's name to make sure that all names are consistent.
   * Failing to do so may result in listing metrics that cannot be processed by consumer modules
   * such as Prediction mode (see https://github.com/NVIDIA/cudf-spark-tools/issues/1042).
   * @param metricName the name of the metric stored in the Spark AccumulableInfo object
   * @return the normalized metric name
   */
  def normalizeMetricName(metricName: String): String = {
    SPARK_LEGACY_METRICS_MAP.getOrElse(metricName, metricName)
  }

  /**
   * Used to parse (value/update) fields of the AccumulableInfo object. If the data is not
   * a valid long, it tries to parse it as a duration in the format of "hour:mm:ss.SSS".
   * Finally, if all the above fails, it tries to parse it as a GPU metric in human-readable format.
   *
   * @param data value stored in the (value/update) of the AccumulableInfo
   * @return valid parsed long of the content or the duration
   */
  @throws[NullPointerException]
  def parseAccumFieldToLong(data: Any): Option[Long] = {
    val strData = data.toString
    try {
      Some(strData.toLong)
    } catch {
      case _ : NumberFormatException =>
        // if the data is not a valid long, try to parse it as a duration.
        val durationValue = StringUtils.parseFromDurationToLongOption(strData)
        if (durationValue.isDefined) {
          return durationValue
        }
        // finally try to parse it as a human-readable GPU metric.
        StringUtils.parseFromGPUMemoryMetricToLongOption(strData)
      case NonFatal(_) =>
        None
    }
  }

  // A utility function used to read Spark properties and compare it to a given target.
  // Note that it takes a default argument as well in case the property is not available.
  def isPropertyMatch(properties: collection.Map[String, String], propKey: String,
      defValue: String, targetValue: String): Boolean = {
    properties.getOrElse(propKey, defValue).equals(targetValue)
  }


  // Reads rootExecutionId via reflection (Spark 3.4+). Returns None on older versions.
  def readRootIDFromSQLStartEvent(
      event: SparkListenerSQLExecutionStart): Option[Long] = {
    Try(invokeMethodOnEvent(event, "rootExecutionId")
      .asInstanceOf[Option[Long]]).getOrElse(None)
  }

  // Reads modifiedConfigs via reflection (Spark 3.3+, SPARK-34735).
  // Returns empty map on older versions.
  def readModifiedConfigsFromSQLStartEvent(
      event: SparkListenerSQLExecutionStart): Map[String, String] = {
    Try {
      Option(invokeMethodOnEvent(event, "modifiedConfigs"))
        .map(_.asInstanceOf[Map[String, String]])
    }.toOption.flatten.getOrElse(Map.empty)
  }

  // Reads errorMessage via reflection (Spark 3.4+). This is the only failure state that
  // SparkListenerSQLExecutionEnd serializes into the event log; its sibling `executionFailure`
  // is a live-listener-only field and is always empty when replaying a log.
  // Returns None on older versions or when the execution succeeded.
  def readErrorMessageFromSQLEndEvent(
      event: SparkListenerSQLExecutionEnd): Option[String] = {
    Try {
      Option(invokeMethodOnEvent(event, "errorMessage"))
        .map(_.asInstanceOf[Option[String]])
    }.toOption.flatten.flatten.filter(_.nonEmpty)
  }

  // Reads jobTags via reflection (Spark 3.5+, introduced for Connect support).
  // Returns empty set on older versions.
  def readJobTagsFromSQLStartEvent(
      event: SparkListenerSQLExecutionStart): Set[String] = {
    Try {
      Option(invokeMethodOnEvent(event, "jobTags"))
        .map(_.asInstanceOf[Set[String]])
    }.toOption.flatten.getOrElse(Set.empty)
  }

  @throws[com.fasterxml.jackson.core.JsonParseException]
  private def handleEventJsonParseEx(
      ex: com.fasterxml.jackson.core.JsonParseException): Unit = {
    // Spark 3.4- will throw a JsonParseException if the eventlog is incomplete (lines are broken)
    val exMsg = ex.getMessage
    if (exMsg != null && exMsg.contains("Unexpected end-of-input within/between Object entries")) {
      // In case the eventlog is incomplete (i.e., inprogress), we show a warning message
      // because we do not want to cause the entire app to fail.
      logWarning(s"Incomplete eventlog, $exMsg")
    } else {
      // This is a parser error thrown by spark-3.4+ which indicates the log is malformed
      if (exMsg != null) {
        // dump an error message to explain the details of the exception, without dumping the
        // entire stack. Otherwise, the logs become not friendly to read.
        logError(s"Eventlog parse exception: $exMsg")
      }
      throw ex
    }
  }

  private lazy val runtimeEventFromJsonMethod = {
    // Spark 3.4 and Databricks changed the signature on sparkEventFromJson
    // Note that it is preferred we use reflection rather than checking Spark-runtime
    // because some vendors may back-port features.
    val c = Class.forName("org.apache.spark.util.JsonProtocol")
    val m = Try {
      // versions prior to spark3.4
      c.getDeclaredMethod("sparkEventFromJson", classOf[org.json4s.JValue])
    } match {
      case Success(a) =>
        (line: String) =>
          a.invoke(null, parse(line)).asInstanceOf[org.apache.spark.scheduler.SparkListenerEvent]
      case Failure(_) =>
        // Spark3.4+ and databricks
        val b = c.getDeclaredMethod("sparkEventFromJson", classOf[String])
        (line: String) =>
          b.invoke(null, line).asInstanceOf[org.apache.spark.scheduler.SparkListenerEvent]
    }
    m
  }

  lazy val getEventFromJsonMethod:
    String => Option[org.apache.spark.scheduler.SparkListenerEvent] = {
    // At this point, the method is already defined.
    // Note that the Exception handling is moved within the method to make it easier
    // to isolate the exception reason.
    (line: String) => Try {
      runtimeEventFromJsonMethod.apply(line)
    } match {
      case Success(i) => Some(i)
      case Failure(e) =>
        e match {
          case i: InvocationTargetException =>
            val targetEx = i.getTargetException
            if (targetEx != null) {
              targetEx match {
                case j: com.fasterxml.jackson.core.io.JsonEOFException =>
                  // Spark3.41+ embeds JsonEOFException in the InvocationTargetException
                  // We need to show a warning message instead of failing the entire app.
                  logWarning(s"Incomplete eventlog, ${j.getMessage}")
                case k: com.fasterxml.jackson.core.JsonParseException =>
                  // this is a parser error thrown by spark-3.4+ which indicates the log is
                  // malformed
                  handleEventJsonParseEx(k)
                case z: ClassNotFoundException if z.getMessage != null =>
                  // Avoid reporting missing classes more than once to reduce noise in the logs
                  reportMissingEventClass(z.getMessage)
                case a: AssertionError =>
                  // This is to catch "assertion failed: expected hostname or IPv6"
                  // which may be caused by malformed eventlog.
                  handleAssertionError(a)
                case t: Throwable =>
                  // We do not want to swallow unknown exceptions so that we can handle later
                  logError(s"Unknown exception while parsing an event", t)
              }
            } else {
              // Normally it should not happen that invocation target is null.
              logError(s"Unknown exception while parsing an event", i)
            }
          case j: com.fasterxml.jackson.core.io.JsonEOFException =>
            // Note that JsonEOFException is child of JsonParseException
            // In case the eventlog is incomplete (i.e., inprogress), we show a warning message
            // because we do not want to cause the entire app to fail.
            logWarning(s"Incomplete eventlog, ${j.getMessage}")
          case k: com.fasterxml.jackson.core.JsonParseException =>
            // this is a parser error thrown by version prior to spark-3.4+ which indicates the
            // log is malformed
            handleEventJsonParseEx(k)
        }
        None
    }
  }

  /**
   * Find the accepted line prefixes for a given tool name.
   * @param toolName the simple classname of the appBase class
   * @return a list of accepted line prefixes. Throws illegalArgumentException if the tool name is
   *         unknown.
   */
  @throws[IllegalArgumentException]
  def getAcceptedLinePrefix(toolName: String): List[String] = {
    acceptedLinesToolMap.get(toolName) match {
      case Some(acceptedLines) => acceptedLines
      case None =>
        throw new IllegalArgumentException(s"Unknown tool name $toolName. " +
          s"Accepted tool names: ${acceptedLinesToolMap.keys.mkString(", ")}")
    }
  }

  // --- Generic reflective method accessors ---
  // Cached by (className, methodName) for repeated use.

  // Cache of reflective method accessors, keyed by (className, methodName).
  private val methodCache =
    new java.util.concurrent.ConcurrentHashMap[
      (String, String), Option[java.lang.reflect.Method]]()

  /** Get a String value from an object via reflective method invocation. */
  def getStringFromEvent(event: AnyRef, methodName: String): String = {
    invokeMethodOnEvent(event, methodName).asInstanceOf[String]
  }

  /** Get a Long value from an object via reflective method invocation. */
  def getLongFromEvent(event: AnyRef, methodName: String): Long = {
    invokeMethodOnEvent(event, methodName).asInstanceOf[Long]
  }

  /** Get an Option[Long] value from an object via reflective method invocation.
   *  Handles Option[Long], plain Long, and numeric types (Integer, etc.) from
   *  Jackson deserialization where JSON numbers may arrive as different types. */
  def getOptLongFromEvent(event: AnyRef, methodName: String): Option[Long] = {
    Try {
      invokeMethodOnEvent(event, methodName) match {
        case opt: Option[_] => opt.map {
          case n: Number => n.longValue()
          case other => other.asInstanceOf[java.lang.Long].toLong
        }
        case n: Number => Some(n.longValue())
        case _ => None
      }
    }.getOrElse(None)
  }

  /**
   * Invoke a no-arg method on an object, caching the Method reference.
   * Throws NoSuchMethodException if the method does not exist on the object's class.
   */
  @throws[NoSuchMethodException]
  def invokeMethodOnEvent(event: AnyRef, methodName: String): Any = {
    val clazz = event.getClass
    val key = (clazz.getName, methodName)
    val method = methodCache.computeIfAbsent(key,
      _ => Try(clazz.getMethod(methodName)).toOption)
    method match {
      case Some(m) => m.invoke(event)
      case None =>
        throw new NoSuchMethodException(s"${clazz.getName}.$methodName not found")
    }
  }
}
