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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.rapids.tool.ToolUtils

/**
 * object Utils provides toolkit functions
 *
 */
object ProfileUtils {

  // Create a SparkSession
  def createSparkSession: SparkSession = {
    SparkSession
        .builder()
        .appName("Rapids Spark Qualification/Profiling Tool")
        .getOrCreate()
  }

  // Check if the job/stage is GPU mode is on
  def isPluginEnabled(properties: collection.Map[String, String]): Boolean = {
    ToolUtils.isPluginEnabled(properties.toMap)
  }

  // Return None if either of them are None
  def optionLongMinusOptionLong(a: Option[Long], b: Option[Long]): Option[Long] =
    try Some(a.get - b.get) catch {
      case _: NoSuchElementException => None
    }

  // Return None if either of them are None
  def OptionLongMinusLong(a: Option[Long], b: Long): Option[Long] =
    try Some(a.get - b) catch {
      case _: NoSuchElementException => None
    }

  // if a string contains what we are going to use for a delimiter, replace it with something else
  // unless the delimiter is a comma. In CSV, quoting the string makes the comma safe to use. This
  // entire function could be removed if the tests that relied on naive split behavior was removed
  def replaceDelimiter(str: String, delimiter: String): String = {
    if (str != null && !delimiter.equals(ProfileOutputWriter.CSVDelimiter) &&
      str.contains(delimiter)) {
      val replaceWith = if (delimiter.equals(";")) {
        ":"
      } else if (delimiter.equals("|")) {
        ":"
      } else {
        ";"
      }
      str.replace(delimiter, replaceWith)
    } else {
      str
    }
  }
}
