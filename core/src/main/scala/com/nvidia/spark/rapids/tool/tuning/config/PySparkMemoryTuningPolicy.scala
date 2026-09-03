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

package com.nvidia.spark.rapids.tool.tuning.config

import java.util.Locale

import scala.util.Try

sealed trait PySparkMemoryRebalanceSource

object PySparkMemoryRebalanceSource {
  case object Heap extends PySparkMemoryRebalanceSource
  case object Overhead extends PySparkMemoryRebalanceSource

  def parse(value: String): PySparkMemoryRebalanceSource = {
    Option(value).map(_.trim.toUpperCase(Locale.ROOT)) match {
      case Some("HEAP") => Heap
      case Some("OVERHEAD") => Overhead
      case _ =>
        throw new IllegalArgumentException(
          s"Invalid ${PySparkMemoryTuningPolicy.REBALANCE_SOURCE} value '$value': " +
            "expected HEAP or OVERHEAD.")
    }
  }
}

case class PySparkMemoryTuningPolicy(
    evidenceHeadroomMultiplier: BigDecimal,
    retryGrowthFactor: BigDecimal,
    recommendTelemetryConfigs: Boolean,
    rebalanceSource: PySparkMemoryRebalanceSource)

object PySparkMemoryTuningPolicy {
  val EVIDENCE_HEADROOM_MULTIPLIER = "PYSPARK_MEMORY_EVIDENCE_HEADROOM_MULTIPLIER"
  val RETRY_GROWTH_FACTOR = "PYSPARK_MEMORY_RETRY_GROWTH_FACTOR"
  val METRICS_POLLING_INTERVAL = "PYSPARK_MEMORY_METRICS_POLLING_INTERVAL"
  val RECOMMEND_TELEMETRY_CONFIGS = "PYSPARK_MEMORY_RECOMMEND_TELEMETRY_CONFIGS"
  val REBALANCE_SOURCE = "PYSPARK_MEMORY_REBALANCE_SOURCE"
  val PYSPARK_MEMORY_KEY = "spark.executor.pyspark.memory"
  val YARN_IS_PYTHON_KEY = "spark.yarn.isPython"
  val KUBERNETES_RESOURCE_TYPE_KEY = "spark.kubernetes.resource.type"
  val PROCESS_TREE_METRICS_KEY = "spark.executor.processTreeMetrics.enabled"
  val STAGE_EXECUTOR_METRICS_KEY = "spark.eventLog.logStageExecutorMetrics"
  val METRICS_POLLING_INTERVAL_KEY = "spark.executor.metrics.pollingInterval"

  def from(configProvider: TuningConfigProvider): PySparkMemoryTuningPolicy = {
    PySparkMemoryTuningPolicy(
      evidenceHeadroomMultiplier = parseGrowingMultiplier(
        EVIDENCE_HEADROOM_MULTIPLIER,
        configProvider.getEntry(EVIDENCE_HEADROOM_MULTIPLIER).getDefault),
      retryGrowthFactor = parseGrowingMultiplier(
        RETRY_GROWTH_FACTOR,
        configProvider.getEntry(RETRY_GROWTH_FACTOR).getDefault),
      recommendTelemetryConfigs =
        configProvider.getEntry(RECOMMEND_TELEMETRY_CONFIGS).getDefault.toBoolean,
      rebalanceSource = PySparkMemoryRebalanceSource.parse(
        configProvider.getEntry(REBALANCE_SOURCE).getDefault))
  }

  private def parseGrowingMultiplier(configName: String, value: String): BigDecimal = {
    val parsed = Option(value).flatMap(raw => Try(BigDecimal(raw.trim)).toOption)
    parsed.filter(_ > BigDecimal(1)).getOrElse {
      throw new IllegalArgumentException(
        s"Invalid $configName value '$value': expected a finite decimal value greater than 1.")
    }
  }
}
