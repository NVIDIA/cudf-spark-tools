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

import com.nvidia.spark.rapids.tool.AppSummaryInfoBaseProvider
import org.scalatest.funsuite.AnyFunSuite

class OomDetectionSuite extends AnyFunSuite {

  private val pythonWrapper = "org.apache.spark.api.python.PythonException: " +
    "Traceback (most recent call last):\n"

  // (description, failureReason, expectedIsGpuOom)
  private val gpuOomTestCases = Seq(
    ("GpuSplitAndRetryOOM",
      "com.nvidia.spark.rapids.jni.GpuSplitAndRetryOOM: " +
        "GPU OutOfMemory: a batch of 1 cannot be split!",
      true),
    ("GpuRetryOOM",
      "com.nvidia.spark.rapids.jni.GpuRetryOOM: GPU OutOfMemory",
      true),
    ("GpuOOM base class",
      "com.nvidia.spark.rapids.jni.GpuOOM: GPU OutOfMemory",
      true),
    ("pre-24.02 jni.SplitAndRetryOOM (no Gpu prefix)",
      "com.nvidia.spark.rapids.jni.SplitAndRetryOOM: " +
        "GPU OutOfMemory: a batch of 1 cannot be split!",
      true),
    ("pre-24.02 jni.RetryOOM (no Gpu prefix)",
      "com.nvidia.spark.rapids.jni.RetryOOM: GPU OutOfMemory",
      true),
    ("CpuSplitAndRetryOOM should not match",
      "com.nvidia.spark.rapids.jni.CpuSplitAndRetryOOM: " +
        "CPU OutOfMemory",
      false),
    ("CpuRetryOOM should not match",
      "com.nvidia.spark.rapids.jni.CpuRetryOOM: CPU OutOfMemory",
      false),
    ("NullPointerException should not match",
      "java.lang.NullPointerException: some error",
      false),
    ("ExecutorLostFailure should not match",
      "ExecutorLostFailure (executor 5 exited) Exit status: 137",
      false)
  )

  gpuOomTestCases.foreach { case (desc, reason, expected) =>
    test(s"isGpuOom: $desc") {
      assert(SparkRapidsOomExceptions.isGpuOom(reason) === expected)
    }
  }

  Seq(
    "MemoryError" -> "MemoryError: unable to allocate",
    "NumPy _ArrayMemoryError" ->
      "numpy.core._exceptions._ArrayMemoryError: Unable to allocate an array",
    "NumPy 2.x _ArrayMemoryError" ->
      "numpy._core._exceptions._ArrayMemoryError: Unable to allocate an array",
    "PyArrow ArrowMemoryError" -> "pyarrow.lib.ArrowMemoryError: malloc failed"
  ).foreach { case (description, allocationFailure) =>
    test(s"isPythonMemoryLimitFailure: $description") {
      assert(PySparkMemoryEvidence.isPythonMemoryLimitFailure(
        pythonWrapper + allocationFailure))
    }
  }

  Seq(
    "Java OutOfMemoryError" ->
      (pythonWrapper + "java.lang.OutOfMemoryError: Java heap space"),
    "generic Python exception" -> (pythonWrapper + "ValueError: invalid value"),
    "prose mentioning MemoryError" ->
      (pythonWrapper + "RuntimeError: user message mentions MemoryError but is not one"),
    "executor loss" -> "ExecutorLostFailure (executor 5 exited) Exit status: 137",
    "allocation exception without PythonException wrapper" -> "MemoryError: allocation failed",
    "truncated wrapper" -> "org.apache.spark.api.python.PythonException: Traceback truncated"
  ).foreach { case (description, reason) =>
    test(s"isPythonMemoryLimitFailure rejects $description") {
      assert(!PySparkMemoryEvidence.isPythonMemoryLimitFailure(reason))
    }
  }

  test("collect preserves exact failed attempts and empty telemetry evidence") {
    val failedAttempts = Seq(
      (3, 0, pythonWrapper + "MemoryError: failed attempt"),
      (4, 2, pythonWrapper + "pyarrow.lib.ArrowMemoryError: no samples"),
      (5, 0, pythonWrapper + "ValueError: unrelated failure"))
    val samples = Map(
      (3, 0) -> Map("1" -> 7L, "2" -> 5L, "driver" -> 101L, "3" -> 0L),
      // A successful retry for stage 3 must not contribute to failed attempt 0.
      (3, 1) -> Map("1" -> 99L))

    val evidence = PySparkMemoryEvidence.collect(failedAttempts,
      (stageId, attemptId) => samples.getOrElse((stageId, attemptId), Map.empty))

    assert(evidence === Seq(
      PySparkMemoryEvidence(3, 0, Seq(5L, 7L)),
      PySparkMemoryEvidence(4, 2, Seq.empty)))
  }

  test("base app summary provider has no PySpark memory evidence") {
    assert(new AppSummaryInfoBaseProvider().getPySparkMemoryEvidence.isEmpty)
  }
}
