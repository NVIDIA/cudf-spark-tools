/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.analysis.util

import com.nvidia.spark.rapids.tool.profiling.StageAggTaskMetricsProfileResult

/**
 * Implementation of Accumulator object for Photon.
 * It takes the shuffleWriteValues and peakMemValues Accumulables as an argument because those
 * values are not available in the TaskModel.
 */
case class StageAggPhoton(
    shuffleWriteValues: Iterable[Long],
    peakMemValues: Iterable[Long]) extends TaskMetricsAccumRec {

  override def addRecord(rec: StageAggTaskMetricsProfileResult): Unit = {
    throw new UnsupportedOperationException("Not implemented: Cannot use cached results to" +
      "calculate stage aggregates")
  }

  override def finalizeAggregation(): Unit = {
    // Fix the shuffleWriteTimes and the peakMemoryValues to use the shuffleWriteValues and
    // the peakMemValues.
    // A stage that shuffles through a CPU Exchange instead of a Photon sink has no Photon shuffle
    // write metric, but its tasks report the shuffle write time, so keep the task value when
    // shuffleWriteValues is empty. A stage writes one shuffle, so it can't have both.
    // Likewise, a stage with no Photon peak memory metric keeps its tasks' peak execution memory.
    if (!isEmptyAggregates) {
      // Re-calculate the photon specific fields only if the accumulator has tasks.
      // Otherwise, leave it as 0.
      if (shuffleWriteValues.nonEmpty) {
        swWriteTimeSum = shuffleWriteValues.sum
      }
      if (peakMemValues.nonEmpty) {
        peakExecutionMemoryMax = peakMemValues.max
      }
    }
    super.finalizeAggregation()
  }
}
