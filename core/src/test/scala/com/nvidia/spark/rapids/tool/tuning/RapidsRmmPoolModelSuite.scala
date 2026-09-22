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

import com.nvidia.spark.rapids.tool.TargetGpuCapacitySource
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class RapidsRmmPoolModelSuite extends AnyFunSuite {
  import RapidsRmmPoolModel._

  private val MiB = 1024L * 1024L
  private val GiB = 1024L * MiB

  private def defaultPoolInput: NominalRmmPoolInput = {
    NominalRmmPoolInput(
      deviceCapacityBytes = 16L * GiB,
      capacitySource = TargetGpuCapacitySource.DeviceCatalog,
      allocFraction = 1.0,
      maxAllocFraction = 1.0,
      minAllocFraction = 0.25,
      baseReserveBytes = 640L * MiB,
      rmmAllocatorMode = RmmAllocatorModes.Async,
      shuffleMode = ShuffleModes.Multithreaded,
      ucxBounceBufferSizeBytes = 4L * MiB,
      chunkedPackPoolSizeBytes = 10L * MiB,
      exactAllocationConfigured = false,
      uvmEnabled = false)
  }

  private def rightValue[A](result: Either[String, A]): A = result match {
    case Right(value) => value
    case Left(error) => fail(error)
  }

  private def leftMessage[A](result: Either[String, A]): String = result match {
    case Left(error) => error
    case Right(value) => fail(s"expected failure but got $value")
  }

  private lazy val profile = rightValue(
    RapidsPluginCapabilities.rmmPoolSizingProfile("25.08.0"))

  test("usesUcxBounceBufferReserve identifies only the ASYNC plus UCX combination") {
    profile.usesUcxBounceBufferReserve(
      RmmAllocatorModes.Async, ShuffleModes.Ucx) shouldBe true
    profile.usesUcxBounceBufferReserve("async", ShuffleModes.Ucx) shouldBe true
    profile.usesUcxBounceBufferReserve(
      RmmAllocatorModes.Default, ShuffleModes.Ucx) shouldBe false
    profile.usesUcxBounceBufferReserve(
      RmmAllocatorModes.Async, ShuffleModes.Multithreaded) shouldBe false
    profile.usesUcxBounceBufferReserve(RmmAllocatorModes.Async, "ucx") shouldBe false
  }

  test("calculateNominalPool models default startup and UCX reserves") {
    val input = defaultPoolInput
    val defaultPool = rightValue(profile.calculateNominalPool(input))
    defaultPool.profileInfo shouldBe profile.info
    defaultPool.input shouldBe input
    defaultPool.rmmPoolBytes shouldBe (16L * GiB - 650L * MiB)
    defaultPool.input.capacitySource shouldBe TargetGpuCapacitySource.DeviceCatalog
    defaultPool.nominalFreeBytesBeforeReserve shouldBe (16L * GiB - 10L * MiB)
    defaultPool.minimumAllocationBytes shouldBe 4L * GiB
    defaultPool.maximumAllocationBytes shouldBe 16L * GiB
    defaultPool.ucxReserveAdjustmentBytes shouldBe 0L
    defaultPool.effectiveReserveBytes shouldBe 640L * MiB
    defaultPool.maximumPoolBytesAfterReserve shouldBe (16L * GiB - 640L * MiB)

    val ucxPool = rightValue(profile.calculateNominalPool(
      defaultPoolInput.copy(shuffleMode = ShuffleModes.Ucx)))
    ucxPool.rmmPoolBytes shouldBe (16L * GiB - 658L * MiB)
    ucxPool.ucxReserveAdjustmentBytes shouldBe 8L * MiB
    ucxPool.effectiveReserveBytes shouldBe 648L * MiB
  }

  test("calculateNominalPool supports DEFAULT and ARENA modes") {
    Seq(RmmAllocatorModes.Default, RmmAllocatorModes.Arena).foreach { allocatorMode =>
      val pool = rightValue(profile.calculateNominalPool(defaultPoolInput.copy(
        rmmAllocatorMode = allocatorMode, shuffleMode = ShuffleModes.Ucx)))
      pool.rmmPoolBytes shouldBe (16L * GiB - 650L * MiB)
      pool.ucxReserveAdjustmentBytes shouldBe 0L
    }
  }

  test("calculateNominalPool applies the plugin's 512-byte truncation") {
    val input = defaultPoolInput.copy(
      deviceCapacityBytes = 10003L,
      allocFraction = 0.5,
      maxAllocFraction = 1.0,
      minAllocFraction = 0.0,
      baseReserveBytes = 1L,
      ucxBounceBufferSizeBytes = 0L,
      chunkedPackPoolSizeBytes = 1L)

    rightValue(profile.calculateNominalPool(input)).rmmPoolBytes shouldBe 4608L
  }

  test("calculateNominalPool fails closed for unsupported runtime modes") {
    val invalidInputs = Seq(
      defaultPoolInput.copy(exactAllocationConfigured = true) -> "exact allocation",
      defaultPoolInput.copy(uvmEnabled = true) -> "uvm.enabled=true",
      defaultPoolInput.copy(rmmAllocatorMode = "NONE") -> "unsupported",
      defaultPoolInput.copy(rmmAllocatorMode = " ASYNC") -> "unsupported",
      defaultPoolInput.copy(shuffleMode = "CUSTOM") -> "unsupported",
      defaultPoolInput.copy(shuffleMode = "ucx") -> "unsupported",
      defaultPoolInput.copy(allocFraction = Double.NaN) -> "fractions are invalid")

    invalidInputs.foreach { case (input, expectedMessage) =>
      leftMessage(profile.calculateNominalPool(input)) should include(expectedMessage)
    }
  }

  test("calculateNominalPool rejects CACHE_ONLY mode") {
    leftMessage(profile.calculateNominalPool(
      defaultPoolInput.copy(shuffleMode = "CACHE_ONLY"))) should include("unsupported")
  }

  test("calculateNominalPool rejects impossible sizes and checked arithmetic overflow") {
    leftMessage(profile.calculateNominalPool(
      defaultPoolInput.copy(chunkedPackPoolSizeBytes = Long.MaxValue))) should
      include("startup allocation exceeds")

    leftMessage(profile.calculateNominalPool(defaultPoolInput.copy(
      shuffleMode = ShuffleModes.Ucx,
      ucxBounceBufferSizeBytes = Long.MaxValue))) should include("overflows")

    leftMessage(profile.calculateNominalPool(
      defaultPoolInput.copy(baseReserveBytes = 17L * GiB))) should include("reserve exceeds")
  }
}
