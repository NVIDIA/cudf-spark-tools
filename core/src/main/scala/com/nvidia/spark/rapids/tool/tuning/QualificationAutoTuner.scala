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

package com.nvidia.spark.rapids.tool.tuning

import com.nvidia.spark.rapids.tool.{AppSummaryInfoBaseProvider, ClusterSizingStrategy, ConstantTotalCoresStrategy, Platform}
import com.nvidia.spark.rapids.tool.profiling.DriverLogInfoProvider
import com.nvidia.spark.rapids.tool.tuning.config.{QualTuningConfigProvider, TuningConfigProvider, TuningConfiguration}

/**
 * Implementation of the `AutoTuner` designed the Qualification Tool. This class can be used to
 * implement the logic to recommend AutoTuner configurations by the Qualification Tool.
 */
class QualificationAutoTuner(
    appInfoProvider: AppSummaryInfoBaseProvider,
    platform: Platform,
    driverInfoProvider: DriverLogInfoProvider,
    userProvidedTuningConfigs: Option[TuningConfiguration])
  extends AutoTuner(appInfoProvider, platform, driverInfoProvider,
    userProvidedTuningConfigs, QualificationAutoTunerHelper) {

  override type ConfigProviderType = QualTuningConfigProvider

  override protected def createConfigProvider(
      config: Option[TuningConfiguration]): QualTuningConfigProvider = {
    TuningConfigProvider
      .builder
      .withUserProvidedConfig(config)
      .build[QualTuningConfigProvider]
  }

  /**
   * Determines whether a tuning entry should be included in the final recommendations
   * for the Qualification Tool. Applies the base status filtering and additionally
   * only includes entries that are bootstrap-enabled and not marked as removed.
   */
  override def shouldIncludeInFinalRecommendations(tuningEntry: TuningEntryTrait): Boolean = {
    super.shouldIncludeInFinalRecommendations(tuningEntry) &&
      tuningEntry.isBootstrap() && !tuningEntry.isRemoved()
  }

  /**
   * Qualification Bootstrap ignores existing "spark.plugins" property and
   * cuDF plugin is added.
   * Reference: https://github.com/NVIDIA/cudf-spark-tools/issues/1825#issuecomment-3138122418
   */
  override def recommendPluginPropsInternal(): Unit = {
    appendRecommendation("spark.plugins", autoTunerHelper.rapidsPluginClassName)
    // If the user has not explicitly enforced any 'spark.plugins',
    // add a comment to inform them about how to specify additional plugins
    // using the target cluster's 'sparkProperties.enforced' section.
    if (!platform.userEnforcedRecommendations.contains("spark.plugins")) {
      appendComment(additionalSparkPluginsComment)
    }
  }

  /**
   * Override to indicate that shuffle manager cannot be recommended
   * in Qualification Tool.
   */
  override def getShuffleManagerClassName: Either[String, String] = {
    Left(shuffleManagerCommentForQualification)
  }
}

/**
 * Provides configuration settings for the Qualification Tool's AutoTuner
 */
object QualificationAutoTunerHelper extends AutoTunerHelper {
  /**
   * For the Qualification Tool's recommendation for cluster sizing, we want to keep
   * the total number of CPU cores between the source and target clusters constant.
   */
  def recommendedClusterSizingStrategy(platform: Platform): ClusterSizingStrategy =
    ConstantTotalCoresStrategy

  override def createAutoTunerInstance(
      appInfoProvider: AppSummaryInfoBaseProvider,
      platform: Platform,
      driverInfoProvider: DriverLogInfoProvider,
      userProvidedTuningConfigs: Option[TuningConfiguration]): AutoTuner = {
    // TODO: This should be refactored to ensure only instance of `QualAppSummaryInfoProvider`
    //       passed to the `QualificationAutoTuner` instance.
    new QualificationAutoTuner(
      appInfoProvider, platform, driverInfoProvider, userProvidedTuningConfigs)
  }
}
