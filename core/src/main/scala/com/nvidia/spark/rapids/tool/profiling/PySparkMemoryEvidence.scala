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

package com.nvidia.spark.rapids.tool.profiling

import org.apache.spark.sql.rapids.tool.AppBase

/**
 * Configuration-independent evidence from one failed stage attempt that hit the Python memory
 * limit. An empty executor peak sequence is retained so callers can request telemetry on a retry.
 */
case class PySparkMemoryEvidence(
    stageId: Int,
    stageAttemptId: Int,
    executorPythonVMemoryPeaks: Seq[Long])

object PySparkMemoryEvidence {
  private val pythonExceptionPattern =
    """(?m)(?:^|[\s:])org\.apache\.spark\.api\.python\.PythonException(?=[:\s]|$)""".r
  private val allocationExceptionPattern =
    """(?m)^\s*(?:MemoryError|numpy\._?core\._exceptions\._ArrayMemoryError|""" +
      """pyarrow\.lib\.ArrowMemoryError)(?=:|\s*$)"""
  private val compiledAllocationExceptionPattern = allocationExceptionPattern.r

  /** A failure must contain both Spark's PythonException wrapper and a Python allocation error. */
  def isPythonMemoryLimitFailure(failureReason: String): Boolean = {
    pythonExceptionPattern.findFirstIn(failureReason).isDefined &&
      compiledAllocationExceptionPattern.findFirstIn(failureReason).isDefined
  }

  /**
   * Derives evidence while keeping failures and executor metrics isolated by exact stage attempt.
   */
  def fromApp(app: AppBase): Seq[PySparkMemoryEvidence] = {
    val failedAttempts = app.stageManager.getFailedStages.map { stage =>
      (stage.getId, stage.getAttemptId, stage.getFailureReason)
    }
    collect(failedAttempts, app.stageExecutorMetricsManager.getPythonVMemory)
  }

  /** Visible for focused classification and attempt-isolation tests. */
  def collect(
      failedAttempts: Iterable[(Int, Int, String)],
      getExecutorPeaks: (Int, Int) => Map[String, Long]): Seq[PySparkMemoryEvidence] = {
    failedAttempts.collect {
      case (stageId, attemptId, reason) if isPythonMemoryLimitFailure(reason) =>
        val peaks = getExecutorPeaks(stageId, attemptId).iterator.collect {
          case (executorId, peak) if executorId != "driver" && peak > 0L => peak
        }.toSeq.sorted
        PySparkMemoryEvidence(stageId, attemptId, peaks)
    }.toSeq.sortBy(evidence => (evidence.stageId, evidence.stageAttemptId))
  }
}
