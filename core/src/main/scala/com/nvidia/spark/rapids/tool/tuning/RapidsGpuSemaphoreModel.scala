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

import org.apache.spark.network.util.ByteUnit

/** Compatibility calculations for the RAPIDS GpuSemaphore memory-admission representation. */
private[tuning] object RapidsGpuSemaphoreModel {
  /**
   * Mirrors the plugin's private `GpuSemaphore.PERMIT_MEMORY_SIZE`. The tools module has no
   * plugin dependency, so it cannot import that value. The runtime represents estimated task
   * memory as one permit per 32 MiB of initialized RMM pool memory; this is an admission
   * granularity, not an allocation or user-tunable AutoTuner policy.
   */
  val AdmissionPermitSizeBytes: Long = ByteUnit.MiB.toBytes(32L)

  /**
   * Maximum task count representable by whole admission permits. Invalid pool sizes return a
   * diagnostic so AutoTuner can fail closed instead of throwing an exception.
   */
  def maxRepresentableTaskCount(rmmPoolBytes: Long): Either[String, Long] = {
    if (rmmPoolBytes <= 0L) {
      Left("RMM pool must be positive")
    } else {
      Right(math.max(1L, rmmPoolBytes / AdmissionPermitSizeBytes))
    }
  }
}
