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

import com.nvidia.spark.rapids.tool.AppSummaryInfoBaseProvider
import com.nvidia.spark.rapids.tool.analysis.AggRawMetricsResult
import com.nvidia.spark.rapids.tool.profiling.DataSourceProfileResult

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.plangraph.ToolsPlanGraph
import org.apache.spark.sql.rapids.tool.qualification.{QualificationAppInfo, QualificationSummaryInfo}

/**
 * Implementation of AppInfoPropertyGetter to wrap the output of the Qualification analysis.
 * @param appInfo the main QualificationAppInfo object representing the CPU application.
 * @param appAggStats optional stats aggregate is included here for future improvement as we may
 *                    need to feed the autotuner with values from the aggregates.
 * @param rawAggMetrics the raw profiler aggregation metrics
 * @param dsInfo Data source information
 */
class QualAppSummaryInfoProvider(
    val appInfo: QualificationAppInfo,
    val appAggStats: Option[QualificationSummaryInfo],
    val rawAggMetrics: AggRawMetricsResult,
    val dsInfo: Seq[DataSourceProfileResult]) extends AppSummaryInfoBaseProvider with Logging {
  // Group data source info by location for distinct location calculations.
  // TODO: Should We only consider data sources from the final plan?
  //       And also drop the location is unknown.
  //      dsInfo.filter(_.fromFinalPlan)
  private lazy val distinctLocations = dsInfo.groupBy(_.location)

  override def isAppInfoAvailable = true
  private def findPropertyInternal(
      key: String, props: collection.Map[String, String]): Option[String] = {
    props.get(key)
  }

  override def getAllProperties: Map[String, String] = {
    appInfo.sparkProperties
  }

  override def getSparkProperty(propKey: String): Option[String] = {
    findPropertyInternal(propKey, appInfo.sparkProperties)
  }

  override def getRapidsProperty(propKey: String): Option[String] = {
    getSparkProperty(propKey)
  }

  override def getSystemProperty(propKey: String): Option[String] = {
    findPropertyInternal(propKey, appInfo.systemProperties)
  }

  override def getSparkVersion: Option[String] = {
    Option(appInfo.sparkVersion)
  }

  def getAppID: String = appInfo.appId

  override def getJvmGCFractions: Seq[Double] = {
    rawAggMetrics.sqlAggs.map {
      taskMetrics => taskMetrics.jvmGCTimeSum * 1.0 / taskMetrics.executorCPUTimeSum
    }
  }

  override def getSpilledMetrics: Seq[Long] = {
    rawAggMetrics.sqlAggs.map { task =>
      task.diskBytesSpilledSum + task.memoryBytesSpilledSum
    }
  }

  // Return shuffle stage(Id)s which have positive spilling metrics
  // The heuristics below assume that these are CPU event logs and just look at the
  // size of memory bytes spilled.
  override def getShuffleStagesWithPosSpilling: Set[Long] = {
    // TODO:Should not this be same as SingleAppSummaryInfoProvider.getShuffleStagesWithPosSpilling?
    rawAggMetrics.stageAggs.collect { case row
      if row.srTotalBytesReadSum + row.swBytesWrittenSum > 0 &&
      row.diskBytesSpilledSum > 0 => row.id
    }.toSet
  }

  override def getShuffleSkewStages: Set[Long] = {
    rawAggMetrics.taskShuffleSkew.map { row => row.stageId }.toSet
  }

  override def getMaxInput: Double = {
    if (rawAggMetrics.sqlAggs.nonEmpty) {
      rawAggMetrics.sqlAggs.map(_.inputBytesReadMax).max.toDouble
    } else {
      0.0
    }
  }

  protected[tool] def planGraphForSqlVersion(
      sqlId: Long, version: Int): Option[ToolsPlanGraph] = {
    FileScanInputMetrics.latestPlanGraph(appInfo, sqlId, version)
  }

  override lazy val getMaxFileScanInput: Option[Double] = {
    FileScanInputMetrics.maxInputBytes(
      dsInfo, rawAggMetrics.stageAggs, planGraphForSqlVersion)
  }

  // Rapids Jar will be empty since CPU event logs are used here
  override def getRapidsJars: Seq[String] = {
    Seq.empty
  }

  override def getDistinctLocationPct: Double = {
      100.0 * distinctLocations.size / dsInfo.size
    }

  override def getRedundantReadSize: Long = {
    distinctLocations
      .filter { case (_, objects) =>
        objects.size > 1 && objects.exists(_.format.contains("Parquet"))
      }
      .iterator
      .map { case (k, v) => k -> v.map(_.data_size).sum }
      .toMap
      .values
      .sum
  }

  override def getMeanInput: Double = {
    if (rawAggMetrics.ioAggs.nonEmpty) {
      rawAggMetrics.ioAggs.map(_.inputBytesReadSum).sum * 1.0 / rawAggMetrics.ioAggs.size
    } else {
      0.0
    }
  }

  override def getMeanShuffleRead: Double = {
    if (rawAggMetrics.ioAggs.nonEmpty) {
      rawAggMetrics.ioAggs.map(_.srTotalBytesReadSum).sum * 1.0 / rawAggMetrics.ioAggs.size
    } else {
      0.0
    }
  }

  override def getClassPathEntries: Map[String, String] = {
    appInfo.classpathEntries
  }
}
