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

import com.fasterxml.jackson.annotation.{JsonIgnoreProperties, JsonProperty}
import com.nvidia.spark.rapids.tool.planparser.db.DBLogMetadataEvent

import org.apache.spark.sql.rapids.tool.annotation.ToolsReflection

/**
 * Databricks writes this event at the start of each rolled event-log file, for example
 * {"Event":"DBCEventLoggingListenerMetadata","Spark Version":"4.0.0",...}.
 *
 * Spark has no class for it. For an event it does not know, Spark's JsonProtocol loads the
 * class named by the "Event" field and deserializes the JSON into it, as it does for
 * SparkRapidsBuildInfoEvent. That name has no package, so this class is in the empty package.
 * See DBLogMetadataEvent.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
case class DBCEventLoggingListenerMetadata(
    @JsonProperty("Spark Version") sparkVersion: String) extends DBLogMetadataEvent {
  @ToolsReflection("BD-3.2.1", "Ignore")
  override val eventTime: Long = 0
  @ToolsReflection("BD-3.2.1", "Ignore")
  override val eventType: String = ""
}
