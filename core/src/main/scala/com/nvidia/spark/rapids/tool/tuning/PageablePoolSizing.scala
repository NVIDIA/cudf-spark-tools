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

/** Pool storage is part of overhead, never an additional container reservation. */
private[tuning] object PageablePoolSizing {
  def sizeMiB(overhead: Long, pinned: Long, cap: Long, hostLimit: Option[Long]): Option[Long] = {
    if (overhead <= 0 || pinned < 0 || cap <= 0) return None
    val poolBudget = math.min(overhead * 3 / 4, hostLimit.filter(_ > 0).getOrElse(overhead))
    val size = math.min(cap, math.max(0L, poolBudget - pinned) / 2)
    if (size >= 256) Some(size) else None
  }
}
