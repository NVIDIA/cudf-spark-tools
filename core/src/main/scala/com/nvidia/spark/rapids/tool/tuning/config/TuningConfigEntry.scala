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

import scala.beans.BeanProperty

import com.nvidia.spark.rapids.tool.ClusterSizingStrategy

import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.util.{StringUtils, ValidatableProperties}

/**
 * Represents a tuning configuration entry with the following properties:
 *   - name:        The configuration parameter name/key.
 *   - description: Human-readable description of the configuration parameter.
 *   - usedBy:      Comma-separated list of Spark configurations this constant affects.
 *   - pluginRules: Comma-separated list of plugin rules that use this configuration (optional).
 *   - default:     The default value to use (optional).
 *   - min:         The minimum allowed value (optional).
 *   - max:         The maximum allowed value (optional).
 */
class TuningConfigEntry(
    @BeanProperty var name: String = "",
    @BeanProperty var description: String = "",
    @BeanProperty var usedBy: String = "",
    @BeanProperty var pluginRules: String = "",
    @BeanProperty var default: String = "",
    @BeanProperty var min: String = "",
    @BeanProperty var max: String = "")
  extends ValidatableProperties
  with MergeableConfigTrait[TuningConfigEntry] {

  def this() = this("", "", "", "", "", "", "")

  override def configKey: String = name

  override def isEmpty: Boolean = {
    isEmptyValue(name) && isEmptyValue(description) && isEmptyValue(usedBy) &&
      isEmptyValue(pluginRules) && isEmptyValue(default) && isEmptyValue(min) && isEmptyValue(max)
  }

  // Helper methods for memory unit conversions
  // These methods return the memory values converted to the specified ByteUnit
  // as Long values.
  def getDefaultAsMemory(unit: ByteUnit): Long = {
    getMemoryValueInUnit(default, unit)
  }

  def getMinAsMemory(unit: ByteUnit): Long = {
    getMemoryValueInUnit(min, unit)
  }

  def getMaxAsMemory(unit: ByteUnit): Long = {
    getMemoryValueInUnit(max, unit)
  }

  /** Helper method to convert memory string to target unit */
  private def getMemoryValueInUnit(value: String, targetUnit: ByteUnit): Long = {
    val valueInBytes = StringUtils.convertMemorySizeToBytes(value, Some(ByteUnit.BYTE))
    ByteUnit.BYTE.convertTo(valueInBytes, targetUnit)
  }

  /**
   * Merge this TuningConfigEntry with another one. The other entry's values will override
   * this entry's values if they are not empty.
   * Note:
   * - Name must be the same for both entries, otherwise an exception is thrown.
   * - Description and usedBy are not merged, the original values are kept.
   *
   * @param other The other TuningConfigEntry to merge with
   * @return A new TuningConfigEntry with merged values
   */
  override protected def mergeWith(other: TuningConfigEntry): TuningConfigEntry = {
    new TuningConfigEntry(
      name = this.name,
      description = this.description,
      usedBy = this.usedBy,
      pluginRules = if (isEmptyValue(other.pluginRules)) this.pluginRules else other.pluginRules,
      default = if (isEmptyValue(other.default)) this.default else other.default,
      min = if (isEmptyValue(other.min)) this.min else other.min,
      max = if (isEmptyValue(other.max)) this.max else other.max
    )
  }

  /**
   * Creates a copy of this TuningConfigEntry with all the same field values.
   *
   * @return A new TuningConfigEntry instance with copied values
   */
  override protected def copyConfig(): TuningConfigEntry = {
    new TuningConfigEntry(
      name = this.name,
      description = this.description,
      usedBy = this.usedBy,
      pluginRules = this.pluginRules,
      default = this.default,
      min = this.min,
      max = this.max
    )
  }

  override def toString: String = {
    s"TuningConfigEntry(name='$name', description='$description', usedBy='$usedBy', " +
      s"pluginRules='$pluginRules', default='$default', min='$min', max='$max')"
  }

  override def validate(): Unit = {
    // Skip validation if the entry is completely empty (i.e. when loaded from YAML with no values)
    if (!isEmpty) {
      // Validate the name is defined
      if (isEmptyValue(name)) {
        throw new IllegalArgumentException(s"Name must be defined for config '$name'")
      }
      if (name == "CLUSTER_SIZING_STRATEGY" && !isEmptyValue(default)) {
        ClusterSizingStrategy.fromString(default)
      }
      // Validate at least one of the default, min, max is defined
      if (isEmptyValue(default) && isEmptyValue(min) && isEmptyValue(max)) {
        throw new IllegalArgumentException(s"At least one of the default, min, max " +
          s"must be defined for config '$name'")
      }
    }
  }
}

object TuningConfigEntry {
  def apply(
    name: String,
    description: String = "",
    usedBy: String = "",
    pluginRules: String = "",
    default: String = "",
    min: String = "",
    max: String = ""): TuningConfigEntry = {
    new TuningConfigEntry(name, description, usedBy, pluginRules, default, min, max)
  }
}
