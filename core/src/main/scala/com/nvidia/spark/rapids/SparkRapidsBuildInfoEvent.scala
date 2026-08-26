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

package com.nvidia.spark.rapids

import org.apache.spark.scheduler.SparkListenerEvent
import org.apache.spark.sql.rapids.tool.annotation.ToolsReflection


/**
 * This is a copy from spark-rapids: https://github.com/NVIDIA/cudf-spark/blob/
 * branch-24.10/sql-plugin/src/main/scala/com/nvidia/spark/rapids/Plugin.scala#L416.
 *
 * TODO: set up a automated job to sync this with spark-rapids plugin.
 */
case class SparkRapidsBuildInfoEvent(
  sparkRapidsBuildInfo: Map[String, String],
  sparkRapidsJniBuildInfo: Map[String, String],
  cudfBuildInfo: Map[String, String],
  sparkRapidsPrivateBuildInfo: Map[String, String]
) extends SparkListenerEvent {
  @ToolsReflection("BD-3.2.1", "Ignore")
  override val eventTime: Long = 0
  @ToolsReflection("BD-3.2.1", "Ignore")
  override val eventType: String = ""
}
