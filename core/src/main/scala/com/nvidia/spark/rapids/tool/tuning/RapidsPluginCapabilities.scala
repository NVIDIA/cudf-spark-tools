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

import org.apache.spark.sql.rapids.tool.ToolUtils

/**
 * Plugin versions served by one compatibility implementation. The lower bound is inclusive.
 * `None` keeps the latest implementation open-ended; when a replacement is added, its minimum
 * version becomes the old implementation's exclusive upper bound.
 *
 * @param minPluginVersionInclusive first plugin release served by the implementation
 * @param maxPluginVersionExclusive first plugin release not served, or `None` for the latest
 */
private[tuning] case class PluginVersionRange(
    minPluginVersionInclusive: String,
    maxPluginVersionExclusive: Option[String]) {
  private val releaseVersionPattern = "\\d{2}\\.\\d{2}\\.\\d+".r

  private def isReleaseVersion(value: String): Boolean = {
    Option(value).exists(candidate =>
      releaseVersionPattern.pattern.matcher(candidate).matches())
  }

  def contains(pluginVersion: String): Boolean = {
    isReleaseVersion(pluginVersion) && isReleaseVersion(minPluginVersionInclusive) &&
      maxPluginVersionExclusive.forall(isReleaseVersion) &&
      ToolUtils.compareVersions(pluginVersion, minPluginVersionInclusive).exists(_ >= 0) &&
      maxPluginVersionExclusive.forall { maximum =>
        ToolUtils.compareVersions(pluginVersion, maximum).exists(_ < 0)
      }
  }
}

/** Version-aware capabilities used when resolving a prospective RAPIDS configuration. */
private[tuning] object RapidsPluginCapabilities {
  val MAX_CONCURRENT_GPU_TASKS_MIN_VERSION: String = "25.10.0"

  /** First plugin version whose GpuSemaphore dynamically adjusts memory admission. */
  val AUTO_CONCURRENT_GPU_TASKS_MIN_VERSION: String = "25.06.0"

  private def versionAtLeast(pluginVersion: Option[String], minimumVersion: String): Boolean = {
    pluginVersion.exists { version =>
      ToolUtils.compareVersions(version, minimumVersion).exists(_ >= 0)
    }
  }

  /** Returns true when the resolved plugin dynamically adjusts GPU memory admission. */
  def supportsAutoConcurrentGpuTasks(pluginVersion: Option[String]): Boolean = {
    versionAtLeast(pluginVersion, AUTO_CONCURRENT_GPU_TASKS_MIN_VERSION)
  }

  /**
   * Returns true when the uniquely resolved RAPIDS plugin version supports
   * `spark.rapids.sql.maxConcurrentGpuTasks`.
   */
  def supportsMaxConcurrentGpuTasks(pluginVersion: Option[String]): Boolean = {
    versionAtLeast(pluginVersion, MAX_CONCURRENT_GPU_TASKS_MIN_VERSION)
  }

  /** Select the unique RMM compatibility implementation for a resolved plugin version. */
  def rmmPoolSizingProfile(
      pluginVersion: String): Either[String, RapidsRmmPoolModel.RmmPoolSizingProfile] = {
    RapidsRmmPoolModel.selectCompatibilityProfile(pluginVersion)
  }
}
