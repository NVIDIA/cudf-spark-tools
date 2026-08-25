/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

package org.apache.spark.scheduler

import org.apache.spark.sql.rapids.tool.annotation.ToolsReflection

/**
 * Base trait for events related to SparkRapids build info. This used to add extra APIs that are
 * not defined in the base Spark trait. This is a work around to be compatible in
 * runtime with custom Spark implementations that define abstract methods in the trait.
 * see https://github.com/NVIDIA/cudf-spark-tools/issues/1360
 */
trait ToolsListenerEventExtraAPIs {
  @ToolsReflection("BD-3.2.1",
    "Ignore the implementation: Definition for an abstract field in the SparkListenerEvent.")
  val eventTime: Long = 0
  @ToolsReflection("BD-3.2.1",
    "Ignore the implementation: Definition for an abstract field in the SparkListenerEvent.")
  val eventType: String = ""
}
