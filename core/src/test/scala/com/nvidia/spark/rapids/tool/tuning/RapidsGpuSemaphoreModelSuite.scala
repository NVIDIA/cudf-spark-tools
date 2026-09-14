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

class RapidsGpuSemaphoreModelSuite extends AnyFunSuite {
  import RapidsGpuSemaphoreModel._

  test("maxRepresentableTaskCount follows whole admission-permit boundaries") {
    maxRepresentableTaskCount(AdmissionPermitSizeBytes - 1L) shouldBe Right(1L)
    maxRepresentableTaskCount(AdmissionPermitSizeBytes) shouldBe Right(1L)
    maxRepresentableTaskCount(AdmissionPermitSizeBytes * 2L) shouldBe Right(2L)
  }

  test("maxRepresentableTaskCount rejects a non-positive RMM pool without throwing") {
    maxRepresentableTaskCount(0L) shouldBe Left("RMM pool must be positive")
    maxRepresentableTaskCount(-1L) shouldBe Left("RMM pool must be positive")
  }
}
