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

package org.apache.spark.sql.rapids.tool.store

import com.nvidia.spark.rapids.tool.analysis.{MetricCatalog, MetricDefinition}


/**
 * Accumulator Meta Reference
 * This maintains the reference to the metadata associated with an accumulable
 *
 * @param id - Accumulable id
 * @param name - Reference to the accumulator name
 */
case class AccumMetaRef(id: Long, name: AccumNameRef) {
  /**
   * The catalog declaration for this metric, resolved once per accumulator id rather than once
   * per event. `AccumMetaRef` instances are created by `AccumManager.getOrCreateAccumInfo`, so
   * this keeps the catalog off the per-task-end hot path entirely.
   */
  val definition: Option[MetricDefinition] = MetricCatalog.DEFAULT.lookup(name.value)

  // An integer bitWise operator that represents different classification of the metric.
  // For metrics that are representing max values across agregates, it will be classified as 1;
  // otherwise 0.
  // Later, we can expand this classification to other categories if needed.
  val metricCategory: Int =
    if (definition.exists(_.isAggregatedByMax)) {
      1
    } else {
      0
    }

  /**
   * Fixed-point multiplier the value is stored with. 1 for everything except a metric the
   * catalog declares as decimal, whose values would otherwise not survive an integer store.
   */
  val storageScale: Long = definition.map(_.storageScale).getOrElse(1L)

  /**
   * True when the metric belongs in the `gpu_*` output files. Resolved once per accumulator id
   * for the same reason as `metricCategory`, so callers do not repeat the catalog lookup.
   */
  val isGpuReportedMetric: Boolean = MetricCatalog.DEFAULT.isGpuReportedMetric(name.value)

  def isAggregateByMax: Boolean = metricCategory == 1
  def getName(): String = name.value
}

object AccumMetaRef {
  // Which metrics aggregate by max rather than by sum is declared in
  // `configs/metrics/metricCatalog.yaml`, not hardcoded here. The declaration is a property of
  // the accumulator that emits the metric, and deriving it from the metric's name -- which is
  // what the hardcoded set amounted to -- is what this table replaces.
  val EMPTY_ACCUM_META_REF: AccumMetaRef = new AccumMetaRef(0L, AccumNameRef.EMPTY_ACC_NAME_REF)

  def apply(id: Long, name: Option[String]): AccumMetaRef =
    new AccumMetaRef(id, AccumNameRef.getOrCreateAccumNameRef(name))
}
