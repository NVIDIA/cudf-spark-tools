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

package com.nvidia.spark.rapids.tool.analysis

import scala.beans.BeanProperty

/**
 * Declaration of a single per-task metric.
 *
 * These properties describe the accumulator that emits the metric. They are declared rather
 * than inferred from the metric's name, because the name does not carry them: for example
 * `gpuMaxTaskFootprint` is measured in bytes without containing "Bytes", and
 * `gpuOnGpuTasksWaitingGPUMaxCount` is a count whose name contains "Wait".
 *
 * Uses JavaBean properties for YAML deserialization by SnakeYAML.
 *
 * @param name             the normalized accumulable name, i.e. what `AccumMetaRef.getName()`
 *                         returns. `EventUtils.normalizeMetricName` is the identity on every name
 *                         declared today, but consumers look up by the normalized form, so that is
 *                         the form declared here
 * @param family           which family of metrics this belongs to, see
 *                         [[MetricDefinition.Families]]. This is what decides whether a metric
 *                         is reported in the GPU output files; being present in the catalog is
 *                         NOT sufficient, since the catalog is not GPU-specific
 * @param source           where the per-task values come from, see [[MetricDefinition.Sources]]
 * @param unit             reported unit, see [[MetricDefinition.Units]]. A label only; no
 *                         scaling is derived from it
 * @param valueForm        how the value is serialized, see [[MetricDefinition.ValueForms]]. Note
 *                         `decimal` means the value MAY be a Double: the accumulator's zero case
 *                         still serializes as a plain `"0"`, so ingest must accept both forms
 * @param storageScale     fixed-point multiplier applied on ingest so that sub-integer values
 *                         survive a Long store. 1 for everything except decimal metrics
 * @param aggregation      how per-task values combine into a stage value, see
 *                         [[MetricDefinition.Aggregations]]
 * @param includeInDiagnostics whether the metric appears in the per-stage distribution report
 * @param description      one-line description, used for report column documentation
 */
class MetricDefinition(
    @BeanProperty var name: String,
    @BeanProperty var family: String,
    @BeanProperty var source: String,
    @BeanProperty var unit: String,
    @BeanProperty var valueForm: String,
    @BeanProperty var storageScale: Long,
    @BeanProperty var aggregation: String,
    @BeanProperty var includeInDiagnostics: Boolean,
    @BeanProperty var description: String) {

  /**
   * No-arg constructor required by SnakeYAML for deserialization.
   *
   * The vocabulary fields default to null rather than to a legal value on purpose: SnakeYAML
   * leaves an omitted YAML key at its default, and a legal default would let an entry that forgot
   * to declare, say, its aggregation load silently as `sum` -- reintroducing exactly the silent
   * guess this table exists to remove. Null fails validation with a message naming the field.
   * `storageScale` and `includeInDiagnostics` are genuinely optional and keep real defaults.
   */
  def this() = this(null, null, null, null, null, 1L, null, false, null)

  /**
   * True when the metric belongs to the GPU family and so belongs in the `gpu_*` output files.
   * Membership of the catalog on its own does not make a metric a GPU metric.
   */
  def isGpuFamily: Boolean = family == MetricDefinition.Families.Gpu

  /** True when the metric belongs to the PerfIO family. */
  def isPerfioFamily: Boolean = family == MetricDefinition.Families.Perfio

  /** True when the metric is reported in the `gpu_*` output files. */
  def isGpuReportedMetric: Boolean = MetricDefinition.Families.gpuReported.contains(family)

  /** True when the stage value is the maximum of the per-task values rather than their sum. */
  def isAggregatedByMax: Boolean = aggregation == MetricDefinition.Aggregations.Max

  /**
   * True when the serialized value may be a decimal and so needs fixed-point storage. The zero
   * case of such a metric still arrives as a plain integer.
   */
  def isDecimalValued: Boolean = valueForm == MetricDefinition.ValueForms.Decimal

  override def toString: String = {
    s"MetricDefinition(name=$name, family=$family, source=$source, unit=$unit, " +
      s"valueForm=$valueForm, " +
      s"storageScale=$storageScale, aggregation=$aggregation, " +
      s"includeInDiagnostics=$includeInDiagnostics)"
  }
}

object MetricDefinition {
  /**
   * Which family a metric belongs to. This is the field that decides GPU reporting: the catalog
   * itself is deliberately not GPU-specific, so a declaration alone must not put a metric into
   * the `gpu_*` files.
   */
  object Families {
    /** GPU execution metrics emitted by the cuDF plugin. */
    val Gpu = "gpu"
    /** PerfIO metrics emitted by the plugin's accelerated readers. */
    val Perfio = "perfio"
    /** Emitted by Spark itself. Declared for later use; nothing is in this family yet. */
    val Spark = "spark"
    val all: Set[String] = Set(Gpu, Perfio, Spark)

    /**
     * The families carried by the `gpu_*` output files. PerfIO metrics have always been reported
     * there alongside the GPU ones, so they stay together; they are a distinct family so that
     * they can be queried and, later, reported separately without changing what a GPU metric is.
     */
    val gpuReported: Set[String] = Set(Gpu, Perfio)
  }

  /** Where the per-task values come from. */
  object Sources {
    val Accumulable = "accumulable"
    /** A field already retained on TaskModel. Declared for later use; not yet consumed. */
    val TaskModel = "taskModel"
    val all: Set[String] = Set(Accumulable, TaskModel)
  }

  /** The unit a metric is reported in. This is a label; no conversion is derived from it. */
  object Units {
    val Bytes = "bytes"
    val Millis = "ms"
    val Count = "count"
    val all: Set[String] = Set(Bytes, Millis, Count)
  }

  /** How the value is serialized in the event log. */
  object ValueForms {
    val Integer = "integer"
    /** May be a Double. The accumulator's zero case still serializes as a plain "0". */
    val Decimal = "decimal"
    val all: Set[String] = Set(Integer, Decimal)
  }

  /** How per-task values combine into a stage value. */
  object Aggregations {
    val Sum = "sum"
    val Max = "max"
    val all: Set[String] = Set(Sum, Max)
  }
}
