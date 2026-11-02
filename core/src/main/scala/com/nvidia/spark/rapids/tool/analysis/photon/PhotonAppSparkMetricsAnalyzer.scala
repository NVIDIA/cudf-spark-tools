/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.analysis.photon

import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.tool.analysis.AppSparkMetricsAnalyzer
import com.nvidia.spark.rapids.tool.analysis.util.{AggAccumHelper, AggAccumPhotonHelper}
import com.nvidia.spark.rapids.tool.planparser.db.PhotonParseHelper

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.store.AccumInfo

/**
 * Photon-specific implementation of AppSparkMetricsAnalyzer.
 *
 * This class extends AppSparkMetricsAnalyzer to handle Databricks Photon applications that
 * require special metric aggregation logic. In Photon, some metrics (peak memory and shuffle
 * write time) are calculated from accumulators instead of task metrics.
 *
 * Key differences from base class:
 * - Collects Photon-specific accumulators for peak memory and shuffle write time
 * - Uses AggAccumPhotonHelper instead of standard AggAccumHelper for metric aggregation
 * - Metrics are extracted from accumulators using Photon-specific metric labels
 *
 * @param app the Photon AppBase object to analyze
 */
class PhotonAppSparkMetricsAnalyzer(app: AppBase) extends AppSparkMetricsAnalyzer(app) {

  // Photon-specific accumulators collected once during initialization
  private val photonPeakMemoryAccumInfos = ArrayBuffer[AccumInfo]()
  private val photonShuffleWriteTimeAccumInfos = ArrayBuffer[AccumInfo]()

  // Collect Photon-specific accumulators during initialization
  collectPhotonAccumulators()

  /**
   * Collects Photon-specific accumulators for peak memory and shuffle write time.
   *
   * For Photon apps, peak memory and shuffle write time need to be calculated from accumulators
   * instead of task metrics.
   * Approach:
   *   1. Collect accumulators for each metric type.
   *   2. For each stage, retrieve the relevant accumulators and calculate aggregated values.
   *
   * Note: A HashMap could be used instead of separate mutable.ArrayBuffer for each metric type,
   * but avoiding it for readability.
   */
  private def collectPhotonAccumulators(): Unit = {
    app.accumManager.applyToAccumInfoMap { accumInfo =>
      accumInfo.infoRef.name.value match {
        case name if PhotonParseHelper.isPhotonPeakMemoryMetric(name) =>
          // Collect accumulators for peak memory
          photonPeakMemoryAccumInfos += accumInfo
        case name if PhotonParseHelper.isPhotonShuffleWriteTimeMetric(name) =>
          // Collect accumulators for shuffle write time
          photonShuffleWriteTimeAccumInfos += accumInfo
        case _ => // Ignore other accumulators
      }
    }
  }

  /**
   * Creates a Photon-specific accumulator helper for aggregating metrics.
   *
   * Unlike the base implementation which uses standard task metrics, this creates an
   * AggAccumPhotonHelper that extracts peak memory and shuffle write time from accumulators.
   *
   * @param stageId The stage ID for which to create the helper
   * @return AggAccumPhotonHelper configured with Photon-specific accumulator values
   */
  override protected def createAccumHelper(stageId: Int): AggAccumHelper = {
    // For max peak memory, we need to look at the accumulators at the task level.
    // We leverage the stage level metrics and get the max task update from it
    val peakMemoryValues = photonPeakMemoryAccumInfos.flatMap { accumInfo =>
      accumInfo.getMaxForStage(stageId)
    }
    // For sum of shuffle write time, we need to look at the accumulators at the stage level.
    // We get the values associated with all tasks for a stage
    val shuffleWriteValues = photonShuffleWriteTimeAccumInfos.flatMap { accumInfo =>
      accumInfo.getTotalForStage(stageId)
    }
    new AggAccumPhotonHelper(shuffleWriteValues, peakMemoryValues)
  }
}
