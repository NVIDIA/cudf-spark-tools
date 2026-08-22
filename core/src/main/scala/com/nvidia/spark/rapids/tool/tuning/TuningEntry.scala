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

package com.nvidia.spark.rapids.tool.tuning

import com.nvidia.spark.rapids.tool.tuning.config.{ConfTypeEnum, TuningEntryDefinition}

import org.apache.spark.internal.Logging
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.util.StringUtils

/**
 * A wrapper to the hold the tuning entry information.
 * @param name the name of the property
 * @param originalValueRaw the value from the eventlog
 * @param tunedValueRaw the value recommended by the AutoTuner
 * @param definition the definition of the tuning entry.
 */
abstract class TuningEntryBase(
    override val name: String,
    originalValueRaw: Option[String],
    var tunedValueRaw: Option[String],
    definition: Option[TuningEntryDefinition] = None) extends TuningEntryTrait {

  /**
   * Set the original value from the default value in Spark if it exists.
   * This is needed because some properties may not be set relying on the default value defined by
   * Spark configurations.
   */
  override def setOriginalValueFromDefaultSpark(): Unit = {
    originalValue match {
      case Some(_) => // Do Nothing
      case None =>
        definition match {
          case Some(defn) =>
            if (defn.hasDefaultSpark()) {
              originalValue = Some(defn.defaultSpark)
            }
          case None =>  // Do Nothing
        }
    }
  }

  override def isBootstrap(): Boolean = {
    definition match {
      case Some(defn) => defn.isBootstrap()
      case None => name.startsWith("spark.rapids.")
    }
  }

  override def isEnabled(): Boolean = {
    val globalFlag = definition match {
      case Some(defn) => defn.isEnabled()
      case None => true
    }
    globalFlag && enabled
  }

  override def setRecommendedValue(value: String): Unit = {
    tunedValueRaw = Option(value)
    super.setRecommendedValue(value)
  }

  /////////////////////////
  // Initialization Code //
  /////////////////////////
  def init(): Unit = {
    // Set values through inherited setters which handle normalization
    originalValue = originalValueRaw
    tunedValue = tunedValueRaw
    setOriginalValueFromDefaultSpark()
  }
}

class TuningEntry(
    override val name: String,
    originalValueRaw: Option[String],
    tunedValueRaw: Option[String],
    definition: Option[TuningEntryDefinition] = None)
  extends TuningEntryBase(name, originalValueRaw, tunedValueRaw, definition) {

  /**
   * Normalize the value based on the configuration data type defined in the tuning definition.
   * This ensures consistent formatting for comparison, as users may provide values in different
   * units or formats. For example, if a property is defined as a double, a user provided value
   * of "1" will be normalized to "1.0".
   */
  override def normalizeValue(propValue: String): String = {
    definition.map { defn =>
      defn.getConfTypeAsEnum match {
        case ConfTypeEnum.Int => propValue.toInt.toString
        case ConfTypeEnum.Long => propValue.toLong.toString
        case ConfTypeEnum.Double => propValue.toDouble.toString
        case ConfTypeEnum.Boolean => propValue.toBoolean.toString
        case ConfTypeEnum.Time =>
          // TODO: Implement time normalization if needed (Ref: JavaUtils.timeStringAs())
          propValue
        case ConfTypeEnum.String => propValue
        case _ => throw new IllegalArgumentException(
          s"Unsupported configuration type: ${defn.getConfTypeAsEnum}. " +
            s"Valid types are: ${ConfTypeEnum.values.mkString(", ")}")
      }
    }.getOrElse(propValue)
  }

  init()
}

class MemoryUnitTuningEntry(
    override val name: String,
    originalValueRaw: Option[String],
    tunedValueRaw: Option[String],
    definition: Option[TuningEntryDefinition] = None)
  extends TuningEntryBase(name, originalValueRaw, tunedValueRaw, definition) {

  private def isSpecialValue(value: String): Boolean = {
    definition.exists(_.isSpecialValue(value))
  }

  /**
   * Parse the default memory unit from the tuning table and store it as a ByteUnit value.
   * E.g. "MiB" -> ByteUnit.MiB
   */
  private val defaultMemoryUnit: ByteUnit = {
    val defaultMemoryUnitStr = definition.flatMap(_.getConfUnit).orNull
    require(defaultMemoryUnitStr != null,
      "Default memory unit must be specified for memory tuning entries")

    // Map of memory unit strings to ByteUnit values.
    // This is specific to the default memory unit defined in the tuning table.
    val memoryUnitsMap = Map[String, ByteUnit](
      "Byte" -> ByteUnit.BYTE,
      "KiB" -> ByteUnit.KiB,
      "MiB" -> ByteUnit.MiB,
      "GiB" -> ByteUnit.GiB,
      "TiB" -> ByteUnit.TiB,
      "PiB" -> ByteUnit.PiB
    )

    memoryUnitsMap.getOrElse(defaultMemoryUnitStr,
      throw new IllegalArgumentException(
        s"Unknown memory unit: $defaultMemoryUnitStr. " +
          s"Valid units are: ${memoryUnitsMap.keys.mkString(", ")}"))
  }

  /**
   * Normalize a memory configuration value by converting it to bytes.
   * If no unit is provided, the defaultMemoryUnitStr is used.
   *
   * @param propValue The original property value, possibly without a unit
   * @return The normalized string with bytes unit (e.g. "1024b")
   */
  override def normalizeValue(propValue: String): String = {
    if (isSpecialValue(propValue)) {
      propValue
    } else {
      val bytes = StringUtils.convertMemorySizeToBytes(propValue, Some(defaultMemoryUnit))
      s"${bytes}b"
    }
  }

  /**
   * Format the output value by converting it to the largest appropriate unit.
   * This is used to display the value in a human-readable format.
   *
   * @param propValue The property value in bytes format (e.g. "1024b")
   * @return The formatted string with appropriate unit (e.g. "1KiB")
   */
  override def formatOutput(propValue: String): String = {
    if (isSpecialValue(propValue)) {
      propValue
    } else {
      // Remove the 'b' suffix and convert to bytes
      val bytes = propValue.dropRight(1).toLong
      StringUtils.convertBytesToLargestUnit(bytes)
    }
  }

  init()
}

object TuningEntry extends Logging {
  /**
   * Build a TuningEntry object.
   * @param name the property label
   * @param originalValue the original value from the eventlog
   * @param tunedValue the value recommended by the AutoTuner
   * @param tuningDefinition optional tuning definition
   * @return a TuningEntry object
   */
  def build(
      name: String,
      originalValue: Option[String],
      tunedValue: Option[String],
      tuningDefinition: Option[TuningEntryDefinition] = None): TuningEntryBase = {
    tuningDefinition match {
      case Some(defn) if defn.isMemoryProperty =>
        new MemoryUnitTuningEntry(name, originalValue, tunedValue, tuningDefinition)
      case _ =>
        new TuningEntry(name, originalValue, tunedValue, tuningDefinition)
    }
  }
}
