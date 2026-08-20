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

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.tool.analysis.{ProfAppIndexMapperTrait, QualAppIndexMapperTrait}
import com.nvidia.spark.rapids.tool.profiling.RapidsPropertyProfileResult

import org.apache.spark.sql.rapids.tool.{AppBase, ToolUtils}


/**
 * Print cuDF plugin-related or all Spark properties when the propSource is set to "rapids".
 * Note that cuDF plugin-related properties are not necessarily starting with prefix 'spark.rapids'.
 * This table is inverse of the other tables where the row keys are property keys and the
 columns
 * are the application values. So column1 would be all the key values for app index 1.
 * @return List of properties relevant to the source.
 */
trait AppPropertiesViewTrait extends ViewableTrait[RapidsPropertyProfileResult] {
  def addNewProps(props: mutable.HashMap[String, ArrayBuffer[String]],
                  newRapidsRelated: Map[String, String]): Unit = {
    val inter = props.keys.toSeq.intersect(newRapidsRelated.keys.toSeq)
    val existDiff = props.keys.toSeq.diff(inter)
    val newDiff = newRapidsRelated.keys.toSeq.diff(inter)

    inter.foreach { key =>
      props(key) += newRapidsRelated.getOrElse(key, "null")
    }

    existDiff.foreach { key =>
      props(key) += "null"
    }

    newDiff.foreach { key =>
      val appVals = ArrayBuffer.fill(0)("null") += newRapidsRelated.getOrElse(key, "null")
      props.put(key, appVals)
    }
  }

  /**
   * Get relevant properties for the view. This method should be implemented by the
   * sub-classes to return the relevant properties for the view.
   */
  def getRelevantProperties(app: AppBase): Map[String, String]

  def getRawView(app: AppBase, index: Int): Seq[RapidsPropertyProfileResult] = {
    val props = mutable.HashMap[String, ArrayBuffer[String]]()
    val propsToKeep = getRelevantProperties(app)
    addNewProps(props, propsToKeep)
    val allRows = props.map { case (k, v) => Seq(k) ++ v }.toSeq
    val resRows = allRows.map(r => RapidsPropertyProfileResult(r.head, r.toArray))
    resRows
  }

  override def sortView(rows: Seq[RapidsPropertyProfileResult])
  : Seq[RapidsPropertyProfileResult] = {
    rows.sortBy(cols => cols.key)
  }
}

trait QualPropertiesView extends AppPropertiesViewTrait with QualAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}

trait ProfPropertiesView extends AppPropertiesViewTrait with ProfAppIndexMapperTrait {
  // Keep for the following refactor stages to customize the view based on the app type (Qual/Prof)
}

// View for Rapids specific properties
trait RapidsPropertiesView extends AppPropertiesViewTrait {
  override def getLabel: String = "Spark Rapids parameters set explicitly"
  override def getDescription: String = "Spark Rapids parameters"
  override def getRelevantProperties(app: AppBase): Map[String, String] = {
    app.sparkProperties.iterator.filter { case (k, _) => ToolUtils.isRapidsPropKey(k) }.toMap
  }
}

// View for general Spark properties excluding Rapids-specific ones
trait SparkPropertiesView extends AppPropertiesViewTrait {
  override def getLabel: String = "Spark Properties"
  override def getDescription: String = "Spark Properties"
  override def getRelevantProperties(app: AppBase): Map[String, String] = {
    app.sparkProperties.iterator
      .filter { case (key, _) => !key.contains(ToolUtils.PROPS_RAPIDS_KEY_PREFIX) }.toMap
  }
}

// View for system properties
trait SystemPropertiesView extends AppPropertiesViewTrait {
  override def getLabel: String = "System Properties"
  override def getDescription: String = "System Properties"
  override def getRelevantProperties(app: AppBase): Map[String, String] = {
    app.systemProperties
  }
}

// Create composite objects combining specific property views with tool type views.
object RapidsQualPropertiesView extends RapidsPropertiesView with QualPropertiesView
object SparkQualPropertiesView extends SparkPropertiesView with QualPropertiesView
object SystemQualPropertiesView extends SystemPropertiesView with QualPropertiesView
object RapidsProfPropertiesView extends RapidsPropertiesView with ProfPropertiesView
object SparkProfPropertiesView extends SparkPropertiesView with ProfPropertiesView
object SystemProfPropertiesView extends SystemPropertiesView with ProfPropertiesView
