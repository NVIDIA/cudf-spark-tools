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

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class SparkTaskSlotModelSuite extends AnyFunSuite {
  import SparkTaskSlotModel._

  private def taskSlotInput(
      executorCores: Int = 16,
      executorGpuAmount: Int = 1,
      taskCpus: Int = 1,
      taskGpuAmount: Double = 0.001): TargetTaskSlotInput = {
    TargetTaskSlotInput(executorCores, executorGpuAmount, taskCpus, taskGpuAmount)
  }

  private def rightValue[A](result: Either[String, A]): A = result match {
    case Right(value) => value
    case Left(error) => fail(error)
  }

  test("calculateTaskSlotsPerGpu applies CPU and fractional GPU resource limits") {
    val cpuInput = taskSlotInput(taskCpus = 2)
    val cpuLimited = rightValue(calculateTaskSlotsPerGpu(cpuInput))
    cpuLimited.input shouldBe cpuInput
    cpuLimited.effectiveSlots shouldBe 8L
    cpuLimited.cpuSlotCap shouldBe 8L
    cpuLimited.gpuSlotCap shouldBe 1000L

    val gpuInput = taskSlotInput(taskGpuAmount = 0.25)
    val gpuLimited = rightValue(calculateTaskSlotsPerGpu(gpuInput))
    gpuLimited.input shouldBe gpuInput
    gpuLimited.effectiveSlots shouldBe 4L
    gpuLimited.cpuSlotCap shouldBe 16L
    gpuLimited.gpuSlotCap shouldBe 4L

    rightValue(calculateTaskSlotsPerGpu(
      taskSlotInput(taskGpuAmount = 1.0))).effectiveSlots shouldBe 1L
  }

  test("calculateTaskSlotsPerGpu rejects invalid or unsupported resource settings") {
    Seq(
      calculateTaskSlotsPerGpu(taskSlotInput(executorCores = 0)),
      calculateTaskSlotsPerGpu(taskSlotInput(executorGpuAmount = 0)),
      calculateTaskSlotsPerGpu(taskSlotInput(executorGpuAmount = 2)),
      calculateTaskSlotsPerGpu(taskSlotInput(taskCpus = 0)),
      calculateTaskSlotsPerGpu(taskSlotInput(taskGpuAmount = 0.0)),
      calculateTaskSlotsPerGpu(taskSlotInput(taskGpuAmount = 0.75)),
      calculateTaskSlotsPerGpu(taskSlotInput(taskGpuAmount = 2.0)),
      calculateTaskSlotsPerGpu(taskSlotInput(
        executorCores = 1, taskCpus = 2))).foreach(_.isLeft shouldBe true)
  }
}
