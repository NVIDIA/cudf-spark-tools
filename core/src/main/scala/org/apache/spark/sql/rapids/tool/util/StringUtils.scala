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

import scala.concurrent.duration.{DurationDouble, DurationLong}

import org.apache.spark.internal.Logging
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.InvalidMemoryUnitFormatException


/**
 * Utility containing the implementation of helpers used for parsing and/or formatting
 * strings.
 */
object StringUtils extends Logging {
  // Constant used to replace the unknown values
  val UNKNOWN_EXTRACT: String = "unknown"
  // Constant used to replace values that do not apply in a specific context.
  val INAPPLICABLE_EXTRACT: String = "inapplicable"
  // Regular expression for duration-format 'H+:MM:SS.FFF'
  // Note: this is not time-of-day. Hours can be larger than 12.
  private val regExDurationFormat = "^(\\d+):([0-5]\\d):([0-5]\\d\\.\\d+)$"
  private val regExMemorySize =
    "^(?i)(\\d+(?:\\.\\d+)?)(b|k(?:ib|b)?|m(?:ib|b)?|g(?:ib|b)?|t(?:ib|b)?|p(?:ib|b)?)$".r
  // Regular expression to retrieve the memory size of GPU metrics into 3 groups.
  // The new GPU metrics introduced in https://github.com/NVIDIA/cudf-spark/pull/12517 are not
  // parsed as long. Instead, it is parsed as a composite string of the value and the unit:
  // - 111534336000 -> 0.74GB (11534336000 bytes)
  // - 1289750 -> 1.23MB (1289750 bytes)
  // - 1044585 -> 1020.10KB (1044585 bytes)
  // The three groups are:
  // - 0: the entire content
  // - 1: the numeric value preceding the unit
  // - 2: the unit
  // - 3: the number of bytes
  // scalastyle:off line.size.limit
  private val regExMemoryMetric =
    "^(?i)(\\d+(?:\\.\\d+)?)(b|k(?:ib|b)?|m(?:ib|b)?|g(?:ib|b)?|t(?:ib|b)?|p(?:ib|b)?)\\s+\\((\\d+)\\s+bytes\\)$".r
  // scalastyle:on line.size.limit
  val SUPPORTED_SIZE_UNITS: Seq[String] = Seq("b", "k", "m", "g", "t", "p")
  /**
   * Checks if the strData is of pattern 'HH:MM:SS.FFF' and return the time as long if applicable.
   * If the string is not in the expected format, the result is None.
   * Note that the hours part can be larger than 23.
   *
   * @param strData the data to be evaluated
   * @return long value of the duration
   */
  def parseFromDurationToLongOption(strData: String): Option[Long] = {
    // String operation is usually faster than regular expression match/find. That's why we check
    // using string instead of `regEx.findFirstMatchIn()`
    if (strData.matches(regExDurationFormat)) {
      val tokens = strData.split(":")
      Some(tokens(0).toLong.hours.toMillis +
        tokens(1).toLong.minutes.toMillis +
        tokens(2).toDouble.seconds.toMillis)
    } else {
      None
    }
  }

  /**
   * A utility that parses the value of RAPIDS GPU metrics that were introduced in
   * https://github.com/NVIDIA/cudf-spark/pull/12517.
   * The metrics are in human readable format (e.g., 0.74GB (11534336000 bytes)).
   * @param strData the metric value
   * @return the number of bytes if the string is in the expected format
   */
  def parseFromGPUMemoryMetricToLongOption(strData: String): Option[Long] = {
    regExMemoryMetric.findFirstMatchIn(strData) flatMap { m =>
      try {
        // return the group containing the actual values in bytes.
        // (see the comment definition of regExMemoryMetric)
        Some(m.group(3).toLong)
      } catch {
        case _: NumberFormatException =>
          None
      }
    }
  }

  /**
   * A utility function to guard a string with double quotes. This is handy to use when we know that
   * the argument string has no double_quotes. Therefore, it is faster to execute.
   * @param str the string to be processed.
   * @return return the str surrounded with double quotes.
   */
  def quoteCSVString(str: String): String = {
    "\"" + str + "\""
  }

  def reformatCSVString(str: String): String = {
    quoteCSVString(str.replace("\"", "\"\""))
  }

  def escapeMetaCharacters(str: String): String = {
    str.replaceAll("\n", "\\\\n")
      .replaceAll("\r", "\\\\r")
      .replaceAll("\t", "\\\\t")
      .replaceAll("\f", "\\\\f")
      .replaceAll("\b", "\\\\b")
      .replaceAll("\u000B", "\\\\v")
      .replaceAll("\u0007", "\\\\a")
  }

  /**
   * Process a string based on parameters:
   * - escapes special characters (i.e., \n) if doEscapeMetaCharacters is set to true.
   * - truncates the string if maxLength is a positive integer. It adds ellipses to the end of the
   *   truncated string if showEllipses is set to true.
   * - trim trailing white spaces.
   * @param str the original input string
   * @param doEscapeMetaCharacters when set to true, escapes the special characters. See the full
   *                               list of characters in #escapeMetaCharacters(java.lang.String)
   * @param maxLength maximum length of the output string. Default is 100.
   *                  A 0-value does not truncate the output string.
   * @param showEllipses When set to true and maxLength is >= 0, ellipses will be appended to the
   *                     output string to indicate the string content was truncated.
   * @return a formatted output string based on the input parameters.
   */
  def renderStr(str: String,
      doEscapeMetaCharacters: Boolean,
      maxLength: Int = 100,
      showEllipses: Boolean = false): String = {
    val truncatedStr = if (maxLength > 0 && str.length > maxLength) {
      val tmpStr = str.substring(0, Math.min(str.length, maxLength))
      if (showEllipses && tmpStr.length > 4) {
        // do not show ellipses for strings shorter than 4 characters.
        tmpStr + "..."
      } else {
        tmpStr
      }
    } else {
      str
    }
    val escapedStr = if (doEscapeMetaCharacters) {
      escapeMetaCharacters(truncatedStr)
    } else {
      truncatedStr
    }
    // Finally trim the string to remove any trailing spaces.
    escapedStr.trim
  }

  // Convert a null-able String to Option[Long]
  def stringToLong(in: String): Option[Long] = try {
    Some(in.toLong)
  } catch {
    case _: NumberFormatException => None
  }

  /**
   * Converts a memory size string to bytes. The input string can be in one of two formats:
   * 1. A number with a unit suffix (e.g., "4m", "2.5g", "1024k")
   * 2. A plain number (e.g., "1024", "2.5") - interpreted using the defaultUnit
   * Examples:
   * - convertMemorySizeToBytes("4m") => 4194304 (4 * 1024 * 1024)
   * - convertMemorySizeToBytes("2.5g") => 2684354560 (2.5 * 1024 * 1024 * 1024)
   * - convertMemorySizeToBytes("1024", Some(ByteUnit.KiB)) => 1048576 (1024 * 1024)
   * - convertMemorySizeToBytes("2.5", Some(ByteUnit.MiB)) => 2621440 (2.5 * 1024 * 1024)
   * - convertMemorySizeToBytes("256", None) =>
   *      throws IllegalArgumentException as no unit is specified and no default unit is set
   *
   * @param value The string to convert (e.g., "4m", "2.5g", "1024")
   * @param defaultUnit The unit to use when no unit suffix is provided
   * @return The size in bytes as a Long
   * @throws InvalidMemoryUnitFormatException if the input string is not in a valid format
   *                                          or if no default unit is set
   */
  def convertMemorySizeToBytes(value: String, defaultUnit: Option[ByteUnit]): Long = {
    regExMemorySize.findFirstMatchIn(value.toLowerCase) match {
      case None =>
        if (defaultUnit.isEmpty) {
          throw new InvalidMemoryUnitFormatException(
            s"Unable to convert '$value': No unit specified and no default unit set. " +
              "Please provide a value with a unit or set a default unit."
          )
        }

        // Handle plain numeric values (e.g., "1024", "2.5") using the defaultUnit
        try {
          (value.toDouble * defaultUnit.get.toBytes(1)).toLong
        } catch {
          case _: NumberFormatException =>
            throw new InvalidMemoryUnitFormatException(
              s"Cannot convert '$value' to memory unit ${defaultUnit.get.toString}.")
        }
      case Some(m) =>
        // Handle values with unit suffixes (e.g., "4m", "2.5g")
        val unitSize = m.group(2).substring(0, 1)
        val sizeNum = m.group(1).toDouble
        (sizeNum * Math.pow(1024, SUPPORTED_SIZE_UNITS.indexOf(unitSize))).toLong
    }
  }

  // Map of ByteUnit to (unit suffix, divisor)
  private lazy val byteUnitMap = Seq(
    ByteUnit.PiB -> (("p", 1024.0 * 1024.0 * 1024.0 * 1024.0 * 1024.0)),
    ByteUnit.TiB -> (("t", 1024.0 * 1024.0 * 1024.0 * 1024.0)),
    ByteUnit.GiB -> (("g", 1024.0 * 1024.0 * 1024.0)),
    ByteUnit.MiB -> (("m", 1024.0 * 1024.0)),
    ByteUnit.KiB -> (("k", 1024.0)),
    ByteUnit.BYTE -> (("b", 1.0))
  )

  /**
   * Converts a memory size string to MiB. The input string can be in one of two formats:
   * 1. A number with a unit suffix (e.g., "4m", "2.5g", "1024k")
   * 2. A plain number (e.g., "1024", "2.5") - interpreted using the defaultUnit
   */
  def convertToMB(value: String, defaultUnit: Option[ByteUnit]): Long = {
    val bytes = convertMemorySizeToBytes(value, defaultUnit)
    bytes / (1024 * 1024)
  }

  /**
   * Converts bytes to the largest possible unit suffix that avoids fractions.
   *
   * Examples:
   * - convertBytesToLargestUnit(4194304) => "4m"
   * - convertBytesToLargestUnit(2147483648L) => "2g"
   * - convertBytesToLargestUnit(1536) => "1.5k" => "1536b"
   * - convertBytesToLargestUnit(0) => "0b"
   *
   * @param bytes The number of bytes to convert
   * @return String with the largest appropriate unit suffix that avoids fractions
   */
  def convertBytesToLargestUnit(bytes: Long): String = {
    if (bytes == 0) {
      return "0b"
    }
    byteUnitMap.collectFirst {
      case (unit, (unitSuffix, divisor)) if (bytes.toDouble / divisor) % 1 == 0 =>
        // Find the first unit that results in a whole number
        // and convert the bytes to that unit
        val valueInUnit = ByteUnit.BYTE.convertTo(bytes, unit)
        s"$valueInUnit$unitSuffix"
    }.getOrElse(s"${bytes}b") // Fallback to bytes if no clean division found
  }
}
