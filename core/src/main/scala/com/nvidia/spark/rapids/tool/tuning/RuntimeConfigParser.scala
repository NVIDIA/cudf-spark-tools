/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.tuning

import scala.util.Try

import org.apache.spark.network.util.JavaUtils

/** Narrow parsers for values whose grammar must match the target Spark or plugin runtime. */
private[tuning] object RuntimeConfigParser {
  /** Validated byte count with the canonical form used in diagnostic comments. */
  case class ParsedBytes(bytes: Long) {
    def display: String = s"${bytes}b"
  }

  /** Parse a byte value with the exact integer grammar used by Spark. */
  def parseBytes(value: String): Option[ParsedBytes] = {
    Option(value).flatMap { nonNullValue =>
      Try(JavaUtils.byteStringAsBytes(nonNullValue)).toOption.map(ParsedBytes)
    }
  }

  /** Parse a positive byte value with the exact integer grammar used by Spark. */
  def parsePositiveBytes(value: String): Option[ParsedBytes] = {
    parseBytes(value).filter(_.bytes > 0L)
  }

  /** Parse a non-negative runtime byte setting, including zero-valued disable settings. */
  def parseNonNegativeBytes(value: String): Option[Long] = {
    parseBytes(value).filter(_.bytes >= 0L).map(_.bytes)
  }

  def parseFraction(value: String): Option[Double] = {
    Option(value).flatMap(v => Try(v.trim.toDouble).toOption)
      .filter(v => !v.isNaN && !v.isInfinite && v >= 0.0 && v <= 1.0)
  }

  def parseBoolean(value: String): Option[Boolean] = {
    Option(value).map(_.trim.toLowerCase).collect {
      case "true" => true
      case "false" => false
    }
  }

  def parsePositiveCount(value: String): Option[Long] = {
    Option(value).flatMap(v => Try(v.trim.toLong).toOption).filter(_ > 0L)
  }

  /** Parse a runtime value with a signed 32-bit integer's grammar and range. */
  def parseInt(value: String): Option[Int] = {
    Option(value).flatMap(v => Try(v.trim.toInt).toOption)
  }

  /** Parse a positive count whose runtime grammar and range are a signed 32-bit integer. */
  def parsePositiveInt(value: String): Option[Int] = {
    parseInt(value).filter(_ > 0)
  }

  /** Match Spark ResourceUtils, which parses executor resource amounts without trimming. */
  def parsePositiveExecutorResourceAmount(value: String): Option[Int] = {
    Option(value).flatMap(v => Try(v.toInt).toOption).filter(_ > 0)
  }

  def parsePositiveDouble(value: String): Option[Double] = {
    Option(value).flatMap(v => Try(v.trim.toDouble).toOption)
      .filter(v => !v.isNaN && !v.isInfinite && v > 0.0)
  }
}
