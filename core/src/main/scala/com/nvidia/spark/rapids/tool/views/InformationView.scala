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

package com.nvidia.spark.rapids.tool.views

import com.nvidia.spark.rapids.tool.analysis.{ProfAppIndexMapperTrait, QualAppIndexMapperTrait}
import com.nvidia.spark.rapids.tool.profiling.{AppInfoProfileResults, AppLogPathProfileResults, RapidsJarProfileResult}

import org.apache.spark.sql.rapids.tool.{AppBase, ToolUtils}


trait AppInformationViewTrait extends ViewableTrait[AppInfoProfileResults] {
  override def getLabel: String = "Application Information"

  def getRawView(app: AppBase, index: Int): Seq[AppInfoProfileResults] = {
    app.appMetaData.map { a =>
      AppInfoProfileResults(a.appName, a.appId, Option(a.attemptId),
        a.sparkUser, a.startTime, a.endTime, app.getAppExitCode, app.getAppDuration,
        a.getDurationString, app.getSparkRuntime, app.sparkVersion, app.gpuMode,
        app.totalCoreSeconds)
    }.toSeq
  }
  override def sortView(rows: Seq[AppInfoProfileResults]): Seq[AppInfoProfileResults] = {
    rows.sortBy(cols => (cols.appId, cols.attemptId))
  }
}


object QualInformationView extends AppInformationViewTrait with QualAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}

object ProfInformationView extends AppInformationViewTrait with ProfAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}


trait AppLogPathViewTrait extends ViewableTrait[AppLogPathProfileResults] {
  override def getLabel: String = "Application Log Path Mapping"

  def getRawView(app: AppBase, index: Int): Seq[AppLogPathProfileResults] = {
    app.appMetaData.map { a =>
      AppLogPathProfileResults(a.appName, a.appId, app.getEventLogPath)
    }.toSeq
  }

  override def sortView(rows: Seq[AppLogPathProfileResults]): Seq[AppLogPathProfileResults] = {
    rows.sortBy(cols => cols.appId)
  }
}


object QualLogPathView extends AppLogPathViewTrait with QualAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}

object ProfLogPathView extends AppLogPathViewTrait with ProfAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}

trait AppRapidsJarViewTrait extends ViewableTrait[RapidsJarProfileResult] {
  override def getLabel: String = "NVIDIA cuDF plugin for Apache Spark Jar and cuDF Jar"

  override def getDescription: String = "Rapids 4 Spark Jars"

  def getRawView(app: AppBase, index: Int): Seq[RapidsJarProfileResult] = {
    if (app.gpuMode) {
      // Look for rapids-4-spark and cuDF jar in classPathEntries
      val rapidsJars = app.classpathEntries.filterKeys(_ matches ToolUtils.RAPIDS_JAR_REGEX.regex)
      if (rapidsJars.nonEmpty) {
        val cols = rapidsJars.keys.toSeq
        cols.map(jar => RapidsJarProfileResult(jar))
      } else {
        // Look for the rapids-4-spark and cuDF jars in Spark Properties
        ToolUtils.extractRAPIDSJarsFromProps(app.sparkProperties).map {
          jar => RapidsJarProfileResult(jar)
        }.toSeq
      }
    } else {
      Seq.empty
    }
  }

  override def sortView(rows: Seq[RapidsJarProfileResult]): Seq[RapidsJarProfileResult] = {
    rows.sortBy(cols => cols.jar)
  }
}

object ProfRapidsJarView extends AppRapidsJarViewTrait with ProfAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}
