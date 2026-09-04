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

import scala.util.{Failure, Success, Try}

import com.nvidia.spark.rapids.tool.Platform
import com.nvidia.spark.rapids.tool.analysis.{AppSQLPlanAnalyzer, QualSparkMetricsAggregator}
import com.nvidia.spark.rapids.tool.profiling.{DataSourceProfileResult, RecommendedCommentResult, RecommendedPropertyResult}
import com.nvidia.spark.rapids.tool.tuning.config.TuningConfiguration
import org.apache.hadoop.conf.Configuration

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.qualification.{QualificationAppInfo, QualificationSummaryInfo}
import org.apache.spark.sql.rapids.tool.util.{PropertiesLoader, RapidsToolsConfUtil}

case class TuningResult(
    appID: String,
    recommendations: Seq[TuningEntryTrait],
    comments: Seq[RecommendedCommentResult],
    combinedProps: Option[Seq[RecommendedPropertyResult]] = None)

/**
 * Container which holds metadata and arguments specific to the execution of the AutoTuner.
 * TODO: we need to use the same class in constructing the AutoTuner in the Profiling tools.
 * @param outputRootDir the output directory to dump the recommendation/comments.
 * @param hadoopConf optional configuration to access the remote storage.
 */
case class TunerContext (
    outputRootDir: String,
    hadoopConf: Configuration,
    tuningConfigsPath: Option[String]) extends Logging {

  def getOutputPath: String = {
    // Return the root output directory (not the tuning subdirectory)
    // The specific tuning paths (tuning/ and tuning_apps/) are computed by the report generator
    outputRootDir
  }

  /**
   * Load user provided tuning configurations from the specified path.
   * @return An optional TuningConfiguration if the path is valid and the file is
   *         successfully loaded; None otherwise.
   */
  private lazy val userProvidedTuningConfigs: Option[TuningConfiguration] = {
    tuningConfigsPath.flatMap { path =>
      Try {
        PropertiesLoader[TuningConfiguration].loadFromFile(path)
      } match {
        case Success(configOpt) =>
          configOpt match {
            case Some(config) =>
              logDebug(s"Successfully loaded user tuning configurations from: $path")
              Some(config)
            case None =>
              logWarning(s"Failed to load tuning configurations from: $path. " +
                "File may be empty or have invalid format.")
              None
          }
        case Failure(e) =>
          logError(s"Error loading tuning configurations from: $path", e)
          None
      }
    }
  }

  /**
   * Get a deep copy of the user provided tuning configurations.
   * We do not need to copy the user configs because the current implementation does not change the
   * user configs (it applies updates to the base configs only). However, to be safe in case
   * future implementations modify the user configs, we return a deep copy here.
   * @return An optional deep copy of the user provided tuning configurations.
   */
  private def getUserProvidedTuningConfigs: Option[TuningConfiguration] = {
    userProvidedTuningConfigs.map(_.copy())
  }

  def tuneApplication(
      appInfo: QualificationAppInfo,
      appAggStats: Option[QualificationSummaryInfo],
      appIndex: Int = 1,
      dsInfo: Seq[DataSourceProfileResult],
      platform: Platform): Option[TuningResult] = {
    val sqlAnalyzer = AppSQLPlanAnalyzer(appInfo)
    val rawAggMetrics =
      QualSparkMetricsAggregator.getAggRawMetrics(appInfo, appIndex, Some(sqlAnalyzer))
    // Pass the analyzer instance along so the tuning path reuses its SQL plan analysis.
    QualificationAutoTunerRunner(appInfo, appAggStats, this, rawAggMetrics, dsInfo,
      Some(sqlAnalyzer)).collect {
      case qualTuner =>
        Try {
          qualTuner.runAutoTuner(platform, getUserProvidedTuningConfigs)
        } match {
          case Success(r) => r
          case Failure(e) =>
            logError(s"Failed to generate tuning recommendations for app: ${appInfo.appId}", e)
            null
        }
    }
  }
}

object TunerContext extends Logging {
  class Builder {
    private var outputRootDir: String = _
    private var hadoopConf: Option[Configuration] = None
    private var tuningConfigsPath: Option[String] = None

    def withOutputRootDir(dir: String): Builder = {
      this.outputRootDir = dir
      this
    }

    def withHadoopConf(conf: Configuration): Builder = {
      this.hadoopConf = Some(conf)
      this
    }

    def withTuningConfigsPath(path: Option[String]): Builder = {
      this.tuningConfigsPath = path
      this
    }

    def build(): Option[TunerContext] = {
      Try {
        val hConf = hadoopConf.getOrElse(RapidsToolsConfUtil.newHadoopConf())
        TunerContext(outputRootDir, hConf, tuningConfigsPath)
      } match {
        case Success(c) => Option(c)
        case Failure(e) =>
          logError("Could not create Tuner Context", e)
          None
      }
    }
  }

  def builder(): Builder = new Builder()
}
