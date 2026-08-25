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

package org.apache.spark.sql.rapids.tool

import scala.jdk.CollectionConverters._

/**
 * cuDF plugin tools utilities.
 */
package object util {
  val RAPIDS_TOOLS_SYS_PROP_PREFIX = "rapids.tools."
  def getSystemProperties: Map[String, String] = {
    System.getProperties.stringPropertyNames().asScala
      .map(key => (key, System.getProperty(key))).toMap
  }

  def loadConfFromSystemProperties: Map[String, String] = {
    getSystemProperties.filter(_._1.startsWith(RAPIDS_TOOLS_SYS_PROP_PREFIX))
  }
}
