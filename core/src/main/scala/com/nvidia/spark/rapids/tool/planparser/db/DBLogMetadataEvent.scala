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

package com.nvidia.spark.rapids.tool.planparser.db

import org.apache.spark.scheduler.SparkListenerEvent

/**
 * Databricks' DBCEventLoggingListenerMetadata event, which Databricks writes at the start of
 * each rolled event-log file. Databricks event logs have no SparkListenerLogStart, so this is
 * where they record the Spark version.
 *
 * The event's class is the top-level DBCEventLoggingListenerMetadata, in the empty package
 * because that is the name the "Event" field carries. Code in named packages cannot refer to
 * the empty package, so it matches on this trait instead.
 */
trait DBLogMetadataEvent extends SparkListenerEvent {
  def sparkVersion: String
}
