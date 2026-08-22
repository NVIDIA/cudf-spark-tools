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

package com.nvidia.spark.rapids.tool.tuning.config

import java.util

import scala.beans.BeanProperty
import scala.jdk.CollectionConverters._

import org.yaml.snakeyaml.{DumperOptions, LoaderOptions, Yaml}
import org.yaml.snakeyaml.constructor.Constructor
import org.yaml.snakeyaml.representer.Representer

import org.apache.spark.sql.rapids.tool.util.UTF8Source

// scalastyle:off line.size.limit
// This is similar to confTypes defined in
// https://github.com/apache/spark/blob/branch-3.5/core/src/main/scala/org/apache/spark/internal/config/ConfigBuilder.scala
// scalastyle:on line.size.limit
object ConfTypeEnum extends Enumeration {
  val Int, Long, Double, Boolean, String, Time, Byte = Value

  def fromString(s: String): Value = {
    values.find(_.toString.toLowerCase == s.toLowerCase).getOrElse {
      throw new IllegalArgumentException(s"Unknown conf type: $s")
    }
  }

  def default: Value = String
}

object CategoryEnum extends Enumeration {
  // Functionality:
  //   - Required for functionality of Spark RAPIDS
  // Tuning:
  //   - Required to tune the Spark RAPIDS job
  // MultiThreadReadCoreMultiplier:
  //   - Special purpose category for identifying the spark property to use as a core multiplier
  //     when tuning `spark.rapids.sql.multiThreadedRead.numThreads`.
  val Functionality, Tuning, MultiThreadReadCoreMultiplier = Value

  def fromString(s: String): Value = {
    s.toLowerCase match {
      case "functionality" => Functionality
      case "tuning" => Tuning
      case "multithreadreadcoremultiplier" => MultiThreadReadCoreMultiplier
      case _ => throw new IllegalArgumentException(s"Unknown category: $s")
    }
  }

  def default: Value = Tuning
}

object LevelEnum extends Enumeration {
  // Job:
  //   - Application-level settings (e.g. `spark.kryoserializer.buffer.max`)
  // Cluster:
  //   - Executor/resource-level settings (e.g. `spark.plugins`)
  val Job, Cluster = Value

  def fromString(s: String): Value = {
    s.toLowerCase match {
      case "job" => Job
      case "cluster" => Cluster
      case _ => throw new IllegalArgumentException(s"Unknown level: $s")
    }
  }

  def default: Value = Job
}

/**
 * Represents the type information for a tuning entry configuration.
 * @param name The type name (Byte, String, Int, Time)
 * @param defaultUnit Optional default unit (e.g., "MiB" for byte type)
 */
case class ConfType(name: ConfTypeEnum.Value, defaultUnit: Option[String] = None)

object ConfType {
  def fromMap(map: util.LinkedHashMap[String, String]): ConfType = {
    val typeName = Option(map.get("name")).getOrElse(
      throw new IllegalArgumentException("Unable to create ConfType without name. " +
        "Include name in the tuning definition."))
    val confTypeEnum = ConfTypeEnum.fromString(typeName)
    val unit = Option(map.get("defaultUnit"))
    ConfType(confTypeEnum, unit)
  }
}

/**
 * A wrapper to the hold the tuning entry information.
 * @param label the property name
 * @param description used to explain the importance of that property and how it is used
 * @param enabled global flag to enable/disable the tuning entry. This is used to turn off a
 *                tuning entry
 * @param level This is used to group the tuning entries (job/cluster)
 * @param category Indicates the purpose of that property for RAPIDS.
 *                 "functionality": required to enable RAPIDS
 *                 "tuning": required to tune the runtime.
 * @param bootstrapEntry When true, the property should be added to the bootstrap configuration.
 *                       Default is true.
 * @param defaultSpark The default value of the property in Spark. This is used to set the
 *                     originalValue of the property in case it is not set by the eventlog.
 * @param modifiedBy Comma-separated list of plugins rules that set this property. This is used to
 *                   track which plugin modified property.
 * @param confType A map containing the configuration type information with optional default unit
 *                 Example: { "name": "byte", "defaultUnit": "MiB" } or { "name": "string" }
 * @param specialValues Values that should bypass type-specific normalization and formatting.
 * @param comments The defaults comments to be loaded for the entry. It is a map to represent
 *                 three different types of comments:
 *                 1. "missing" to represent the default comment to be appended to the AutoTuner's
 *                    comment when the property is missing.
 *                 2. "persistent" to represent a comment that always shows up in the AutoTuner's
 *                    output.
 *                 3. "updated" to represent a comment that shows when a property is being set by
 *                    the Autotuner.
 */
class TuningEntryDefinition(
    @BeanProperty var label: String,
    @BeanProperty var description: String,
    @BeanProperty var enabled: Boolean,
    @BeanProperty var level: String,
    @BeanProperty var category: String,
    @BeanProperty var bootstrapEntry: Boolean,
    @BeanProperty var defaultSpark: String,
    @BeanProperty var modifiedBy: String,
    @BeanProperty var confType: util.LinkedHashMap[String, String],
    @BeanProperty var specialValues: util.List[String],
    @BeanProperty var comments: util.LinkedHashMap[String, String]) {
  private lazy val confTypeInfo: ConfType = ConfType.fromMap(confType)
  private lazy val categoryEnum: CategoryEnum.Value = CategoryEnum.fromString(category)
  private lazy val levelEnum: LevelEnum.Value = LevelEnum.fromString(level)

  def this() = {
    this(
      label = "",
      description = "",
      enabled = true,
      level = "",
      category = "",
      bootstrapEntry = true,
      defaultSpark = null,
      modifiedBy = "",
      confType = new util.LinkedHashMap[String, String](),
      specialValues = new util.ArrayList[String](),
      comments = new util.LinkedHashMap[String, String]())
  }

  def isEnabled(): Boolean = {
    enabled
  }

  def isBootstrap(): Boolean = {
    bootstrapEntry || label.startsWith("spark.rapids.")
  }

  /**
   * Indicates if the property is a memory-related property.
   */
  def isMemoryProperty: Boolean = {
    confTypeInfo.name == ConfTypeEnum.Byte
  }

  def getConfUnit: Option[String] = {
    confTypeInfo.defaultUnit
  }

  def getConfTypeAsEnum: ConfTypeEnum.Value = {
    confTypeInfo.name
  }

  def getCategoryAsEnum: CategoryEnum.Value = {
    categoryEnum
  }

  def getLevelAsEnum: LevelEnum.Value = {
    levelEnum
  }

  def isSpecialValue(value: String): Boolean = {
    Option(specialValues).exists(_.asScala.contains(value))
  }

  /**
   * Indicates if the property has a default value in Spark. This implies that the default value
   * can be used to set the original value of the property.
   * @return true if the property has a default value in Spark.
   */
  def hasDefaultSpark(): Boolean = {
    defaultSpark != null
  }

  def getMissingComment(): Option[String] = {
    Option(comments.get("missing"))
  }

  def getPersistentComment(): Option[String] = {
    Option(comments.get("persistent"))
  }

  def getUpdatedComment(): Option[String] = {
    Option(comments.get("updated"))
  }

  /**
   * Mark this entry as enabled and a bootstrap entry.
   * TODO: Identify the usage of BootstrapEntry and see if we need to keep it.
   */
  def markAsEnable(): Unit = {
    this.setEnabled(true)
    this.setBootstrapEntry(true)
  }
}

class TuningEntries(
  @BeanProperty var tuningDefinitions: java.util.List[TuningEntryDefinition]) {
  def this() = {
    this(new java.util.ArrayList[TuningEntryDefinition]())
  }
}

object TuningEntryDefinition {
  // All entries from the tuningTable.yaml
  private lazy val allEntries: Map[String, TuningEntryDefinition] = loadTable()

  // A static Map between the propertyName and the TuningEntryDefinition for enabled entries.
  lazy val TUNING_TABLE: Map[String, TuningEntryDefinition] =
    allEntries.filter(_._2.isEnabled())

  /** Look up a tuning entry definition by label, including disabled entries. */
  def getEntryDefinition(label: String): Option[TuningEntryDefinition] =
    allEntries.get(label)

  /**
   * Creates a tuning definition with commonly used defaults.
   *
   * @param label the property name
   * @param description description of the property (defaults: null)
   * @param confType the configuration type name (default: "string")
   * @param defaultUnit optional default unit for byte types
   * @param enabled whether the tuning entry is enabled (default: true)
   * @param level the level of the tuning entry (default: LevelEnum.Job)
   * @param category the category (default: CategoryEnum.Tuning)
   * @param bootstrapEntry whether this should be a bootstrap entry (default: true)
   * @param defaultSpark the default Spark value (default: null)
   * @param modifiedBy the plugin(s) that modified this entry (default: "")
   * @param specialValues values that bypass type-specific normalization and formatting
   * @return a new TuningEntryDefinition instance
   */
  def apply(
      label: String,
      description: String = null,
      confType: ConfTypeEnum.Value = ConfTypeEnum.String,
      defaultUnit: Option[String] = None,
      enabled: Boolean = true,
      level: LevelEnum.Value = LevelEnum.Job,
      category: CategoryEnum.Value = CategoryEnum.Tuning,
      bootstrapEntry: Boolean = true,
      defaultSpark: String = null,
      modifiedBy: String = "",
      specialValues: Seq[String] = Seq.empty): TuningEntryDefinition = {
    // Create a new TuningEntryDefinition with the provided parameters.
    val defn = new TuningEntryDefinition()
    defn.setLabel(label)
    defn.setDescription(description)
    defn.setEnabled(enabled)
    defn.setLevel(level.toString.toLowerCase)
    defn.setCategory(category.toString.toLowerCase)
    defn.setBootstrapEntry(bootstrapEntry)
    defn.setDefaultSpark(defaultSpark)
    defn.setModifiedBy(modifiedBy)
    defn.setSpecialValues(new util.ArrayList[String](specialValues.asJava))
    // Create the confType map with the provided confType and optional defaultUnit.
    val confTypeMap = Map("name" -> confType.toString) ++ defaultUnit.map("defaultUnit" -> _)
    defn.setConfType(new java.util.LinkedHashMap(confTypeMap.asJava))
    defn
  }

  /**
   * Load the tuning table from a specific yaml resource file.
   * @return a map between property name and the TuningEntryDefinition
   */
  def loadTable(): Map[String, TuningEntryDefinition] = {
    val yamlSource =
      UTF8Source.fromResource("bootstrap/tuningTable.yaml").mkString
    val representer = new Representer(new DumperOptions())
    representer.getPropertyUtils.setSkipMissingProperties(true)
    val constructor = new Constructor(classOf[TuningEntries], new LoaderOptions())
    val yamlObjNested = new Yaml(constructor, representer)
    val entryTable: TuningEntries = yamlObjNested.load(yamlSource).asInstanceOf[TuningEntries]
    entryTable.tuningDefinitions.asScala.iterator.map(e => (e.label, e)).toMap
  }
}
