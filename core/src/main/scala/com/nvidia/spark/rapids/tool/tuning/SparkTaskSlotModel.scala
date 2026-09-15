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

import org.apache.spark.launcher.SparkLauncher

/** Pure Spark task-slot calculation for the supported one-GPU-per-executor target. */
private[tuning] object SparkTaskSlotModel {
  /** Spark property identifiers needed to resolve the model's target resource inputs. */
  object Properties {
    val ExecutorCores: String = SparkLauncher.EXECUTOR_CORES
    val ExecutorGpuAmount = "spark.executor.resource.gpu.amount"
    val TaskCpus = "spark.task.cpus"
    val TaskGpuAmount = "spark.task.resource.gpu.amount"
  }

  /** Spark runtime defaults used only when the target has no effective property value. */
  object SupportedRuntimeDefaults {
    val TaskCpus = "1"
  }

  /** Properties whose prospective target values must keep their runtime parsing semantics. */
  val TargetTaskResourceKeys: Set[String] = Set(
    Properties.ExecutorCores,
    Properties.ExecutorGpuAmount,
    Properties.TaskCpus,
    Properties.TaskGpuAmount)

  /**
   * Effective target resource inputs. The caller resolves these from target configuration,
   * recommendations, or supported runtime defaults; this model does not derive them.
   */
  case class TargetTaskSlotInput(
      executorCores: Int,
      executorGpuAmount: Int,
      taskCpus: Int,
      taskGpuAmount: Double)

  /** Task slots for one target GPU. Every field except `input` is calculated by this model. */
  case class TaskSlotsPerGpu(
      input: TargetTaskSlotInput,
      effectiveSlots: Long,
      cpuSlotCap: Long,
      gpuSlotCap: Long)

  /**
   * Calculate effective Spark task slots for a one-GPU-per-executor target. Spark permits
   * fractional task GPU resources through 0.5; a request of exactly one GPU is the only supported
   * value above 0.5 in this model.
   */
  def calculateTaskSlotsPerGpu(input: TargetTaskSlotInput): Either[String, TaskSlotsPerGpu] = {
    if (input.executorCores <= 0) {
      Left("target executor cores are missing or invalid")
    } else if (input.executorGpuAmount <= 0) {
      Left(s"'${Properties.ExecutorGpuAmount}' must be a positive integer")
    } else if (input.executorGpuAmount != 1) {
      Left(s"unsupported '${Properties.ExecutorGpuAmount}=${input.executorGpuAmount}'; " +
        s"expected one GPU per executor")
    } else if (input.taskCpus <= 0) {
      Left(s"'${Properties.TaskCpus}' must be a positive integer")
    } else if (input.taskGpuAmount.isNaN || input.taskGpuAmount.isInfinite ||
        input.taskGpuAmount <= 0.0) {
      Left(s"'${Properties.TaskGpuAmount}' must be positive")
    } else if (input.taskGpuAmount > 0.5 && input.taskGpuAmount != 1.0) {
      Left(s"unsupported '${Properties.TaskGpuAmount}=${input.taskGpuAmount}' " +
        s"for one GPU per executor")
    } else {
      val cpuSlots = input.executorCores.toLong / input.taskCpus.toLong
      val gpuSlots = if (input.taskGpuAmount <= 0.5) {
        math.floor(1.0 / input.taskGpuAmount).toLong
      } else {
        1L
      }
      val taskSlots = math.min(cpuSlots, gpuSlots)
      if (taskSlots <= 0L) {
        Left("target CPU or GPU task resources provide no runnable task slot")
      } else {
        Right(TaskSlotsPerGpu(
          input = input,
          effectiveSlots = taskSlots,
          cpuSlotCap = cpuSlots,
          gpuSlotCap = gpuSlots))
      }
    }
  }
}
