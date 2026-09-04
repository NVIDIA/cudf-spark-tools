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
import scala.collection.JavaConverters._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.util.{PropertiesLoader, UTF8Source, ValidatableProperties}

/**
 * Catalog of per-task and per-stage metric declarations, loaded from
 * `configs/metrics/metricCatalog.yaml`.
 *
 * The catalog exists so that a metric's unit, its serialized form and its aggregation are
 * declared once rather than inferred from substrings of its name. Adding a metric is a YAML
 * edit; no Scala change is required.
 *
 * A metric that is not declared here falls back to the legacy name-based heuristic for its unit
 * label (see [[MetricCatalog.legacyUnitFor]]), is unscaled, and is aggregated by sum. Note that
 * discovery itself is a separate concern owned by the analyzer, and its prefix rule is narrower
 * than this table -- see the resource file's header.
 *
 * @param metrics the declared metrics
 * @see [[org.apache.spark.sql.rapids.tool.util.PropertiesLoader]]
 */
class MetricCatalog(
    @BeanProperty var metrics: java.util.List[MetricDefinition]
) extends ValidatableProperties with Logging {

  /** No-arg constructor required by SnakeYAML for deserialization. */
  def this() = this(new java.util.ArrayList[MetricDefinition]())

  /**
   * Index by metric name. Lazy because SnakeYAML populates `metrics` through the setter after
   * the no-arg constructor has run, so this must not be forced before deserialization ends.
   */
  private lazy val byName: Map[String, MetricDefinition] = {
    if (metrics == null) {
      Map.empty[String, MetricDefinition]
    } else {
      metrics.asScala.map(e => e.name -> e).toMap
    }
  }

  /**
   * Validates the catalog.
   *
   * Called from the `ValidatableProperties` constructor, which runs before SnakeYAML has
   * populated the list, so an empty or null list is not an error here. The companion's loader
   * checks for emptiness explicitly once deserialization has finished.
   */
  override def validate(): Unit = {
    if (metrics == null || metrics.isEmpty) {
      return // nothing populated yet; the loader validates after deserialization
    }
    val names = metrics.asScala.map(_.name)
    val duplicates = names.groupBy(identity).filter(_._2.size > 1).keys
    if (duplicates.nonEmpty) {
      throw new IllegalArgumentException(
        s"Duplicate metric names in the metric catalog: ${duplicates.mkString(", ")}")
    }
    metrics.asScala.foreach(validateEntry)
  }

  private def validateEntry(entry: MetricDefinition): Unit = {
    def reject(field: String, value: String, allowed: Set[String]): Unit = {
      throw new IllegalArgumentException(
        s"Invalid $field '$value' for metric '${entry.name}'. " +
          s"Expected one of: ${allowed.toSeq.sorted.mkString(", ")}")
    }
    if (entry == null) {
      throw new IllegalArgumentException("Null entry in the metric catalog")
    }
    if (entry.name == null || entry.name.isEmpty) {
      throw new IllegalArgumentException("Metric name cannot be null or empty")
    }
    if (entry.description == null || entry.description.isEmpty) {
      throw new IllegalArgumentException(
        s"Description cannot be null or empty for metric '${entry.name}'")
    }
    if (!MetricDefinition.Families.all.contains(entry.family)) {
      reject("family", entry.family, MetricDefinition.Families.all)
    }
    if (!MetricDefinition.Sources.all.contains(entry.source)) {
      reject("source", entry.source, MetricDefinition.Sources.all)
    }
    if (!MetricDefinition.Units.all.contains(entry.unit)) {
      reject("unit", entry.unit, MetricDefinition.Units.all)
    }
    if (!MetricDefinition.ValueForms.all.contains(entry.valueForm)) {
      reject("valueForm", entry.valueForm, MetricDefinition.ValueForms.all)
    }
    if (!MetricDefinition.Aggregations.all.contains(entry.aggregation)) {
      reject("aggregation", entry.aggregation, MetricDefinition.Aggregations.all)
    }
    if (entry.storageScale < 1L) {
      throw new IllegalArgumentException(
        s"storageScale must be >= 1 for metric '${entry.name}', found ${entry.storageScale}")
    }
    if (entry.isDecimalValued && entry.storageScale <= 1L) {
      throw new IllegalArgumentException(
        s"storageScale must be > 1 for the decimal-valued metric '${entry.name}'; without it " +
          "the value is routed to the integer parser and every non-integral sample is dropped")
    }
    if (entry.storageScale > 1L && !MetricCatalog.isPowerOfTen(entry.storageScale)) {
      throw new IllegalArgumentException(
        s"storageScale must be a power of ten for metric '${entry.name}', " +
          s"found ${entry.storageScale}")
    }
    if (!entry.isDecimalValued && entry.storageScale != 1L) {
      throw new IllegalArgumentException(
        s"storageScale must be 1 for the integer-valued metric '${entry.name}', " +
          s"found ${entry.storageScale}")
    }
  }

  /** The declaration for a metric, or None when it is not declared. */
  def lookup(name: String): Option[MetricDefinition] = byName.get(name)

  /** True when the metric is declared in the catalog. */
  def isDeclared(name: String): Boolean = byName.contains(name)

  /** The declared family of a metric, or None when it is not declared. */
  def familyOf(name: String): Option[String] = byName.get(name).map(_.family)

  /** True when the metric is declared in the given family. */
  def isInFamily(name: String, family: String): Boolean = {
    byName.get(name).exists(_.family == family)
  }

  /** Every declared metric name in the given family. */
  def namesInFamily(family: String): Set[String] = {
    byName.values.filter(_.family == family).map(_.name).toSet
  }

  /** True when the metric is declared in the GPU family. */
  def isGpuFamily(name: String): Boolean = isInFamily(name, MetricDefinition.Families.Gpu)

  /** True when the metric is declared in the PerfIO family. */
  def isPerfioFamily(name: String): Boolean = isInFamily(name, MetricDefinition.Families.Perfio)

  /**
   * True when the metric belongs in the `gpu_*` output files.
   *
   * This is the single place the rule lives, so no caller has to reconstruct it. A DECLARED
   * metric qualifies on its family alone -- being in the catalog is not sufficient, because the
   * catalog is not GPU-specific and a Spark-family metric declared here must not leak into the
   * GPU files. An UNDECLARED metric falls back to the legacy name prefixes, so a plugin metric
   * added after a tools release is still reported rather than silently dropped.
   */
  def isGpuReportedMetric(name: String): Boolean = {
    byName.get(name) match {
      case Some(definition) => definition.isGpuReportedMetric
      case None => MetricCatalog.matchesLegacyGpuPrefix(name)
    }
  }

  /**
   * The reported unit of a metric. An undeclared metric falls back to the legacy name-based
   * heuristic, see [[MetricCatalog.legacyUnitFor]].
   *
   * These accessors are on the hot path -- one call per accumulable per task-end event -- so each
   * is a single map lookup and allocates nothing.
   */
  def unitFor(name: String): String = {
    byName.get(name).map(_.unit).getOrElse(MetricCatalog.legacyUnitFor(name))
  }

  /**
   * True when the stage value is the maximum of the per-task values rather than their sum.
   * An undeclared metric is summed, which is what the previous hardcoded set did.
   */
  def isAggregatedByMax(name: String): Boolean = byName.get(name).exists(_.isAggregatedByMax)

  /** True when the metric appears in the per-stage distribution report. Never for undeclared. */
  def includedInDiagnostics(name: String): Boolean = byName.get(name).exists(_.includeInDiagnostics)

  /** True when the value may be a decimal and needs fixed-point storage. */
  def isDecimalValued(name: String): Boolean = byName.get(name).exists(_.isDecimalValued)

  /** Fixed-point multiplier applied to a metric on ingest. 1 unless the metric is decimal. */
  def storageScaleFor(name: String): Long = {
    byName.get(name).map(_.storageScale).getOrElse(1L)
  }

  /**
   * Renders a stored value for output, undoing the fixed-point scale of a decimal metric.
   *
   * Values are stored as integers, so a metric declared with `storageScale: 1000` is held in
   * thousandths and must be divided before it is shown. Unscaled metrics render exactly as
   * before, so this is a no-op for 30 of the 31 declared metrics.
   */
  def formatValue(name: String, value: Long): String = {
    MetricCatalog.formatStoredValue(value, storageScaleFor(name))
  }

  /** Every declared metric name. */
  lazy val declaredNames: Set[String] = byName.keySet

  /** Names of the metrics that appear in the per-stage distribution report. */
  lazy val diagnosticsMetricNames: Set[String] = {
    byName.values.filter(_.includeInDiagnostics).map(_.name).toSet
  }

  /** Names of the metrics whose stage value is a maximum rather than a sum. */
  lazy val maxAggregatedNames: Set[String] = {
    byName.values.filter(_.isAggregatedByMax).map(_.name).toSet
  }

  override def toString: String = s"MetricCatalog(${byName.size} metrics)"
}

/**
 * Companion providing the catalog loaded from the packaged resource.
 */
object MetricCatalog extends Logging {
  /** Path to the catalog inside the jar. */
  private val DEFAULT_CONFIG_PATH = "configs/metrics/metricCatalog.yaml"

  /** True when the value is a positive power of ten. `formatStoredValue` relies on this. */
  def isPowerOfTen(value: Long): Boolean = {
    var v = value
    while (v > 1L && v % 10L == 0L) {
      v /= 10L
    }
    v == 1L
  }

  /**
   * The legacy name-prefix rule for recognising a plugin metric, kept as the fallback for a
   * metric the catalog does not declare.
   */
  def matchesLegacyGpuPrefix(name: String): Boolean = {
    name.startsWith("gpu") ||
      name.startsWith("perfio.") ||
      name == "multithreadReaderMaxParallelism"
  }

  /**
   * Renders a stored value, dividing out a fixed-point scale.
   *
   * Deliberately integer arithmetic rather than `String.format`/`f"%.3f"`: those use the default
   * JVM Locale, and a comma-decimal locale such as de_DE or fr_FR would emit "0,714" -- a comma
   * inside a comma-delimited CSV field, which shifts every later column of the row. Working in
   * Longs also avoids the Double round-trip entirely.
   *
   * `storageScale` is validated to be a power of ten, so the number of fractional digits is
   * exactly its digit count minus one.
   */
  def formatStoredValue(value: Long, storageScale: Long): String = {
    if (storageScale <= 1L) {
      value.toString
    } else {
      val whole = value / storageScale
      val fraction = Math.abs(value % storageScale)
      if (fraction == 0L) {
        whole.toString
      } else {
        // Integer division truncates toward zero, so a negative value whose whole part is zero
        // would otherwise lose its sign.
        val sign = if (value < 0 && whole == 0L) "-" else ""
        val digits = storageScale.toString.length - 1
        val padded = fraction.toString.reverse.padTo(digits, '0').reverse
        val trimmed = padded.reverse.dropWhile(_ == '0').reverse
        s"$sign$whole.$trimmed"
      }
    }
  }

  /**
   * The legacy name-based unit heuristic, kept as the fallback for a metric that is not declared.
   *
   * This is the rule the catalog replaces, so using it as a fallback deserves justification. It
   * is correct for any metric whose name follows the plugin's convention -- a new
   * `NanoSecondAccumulator` named `gpuFooTime` is labelled `ms`, which is what the parser
   * produces -- and wrong only for the names that motivated the catalog in the first place
   * (`gpuMaxTaskFootprint` is bytes without saying so; `...WaitingGPUMaxCount` is a count whose
   * name contains `Wait`).
   *
   * It is now LABEL-ONLY. The value is no longer scaled by it, so a wrong guess costs a wrong
   * column header rather than a destroyed number, which is what made the old rule dangerous.
   */
  def legacyUnitFor(name: String): String = {
    if (name.contains("Time") || name.contains("Wait")) {
      MetricDefinition.Units.Millis
    } else if (name.contains("Bytes")) {
      MetricDefinition.Units.Bytes
    } else {
      MetricDefinition.Units.Count
    }
  }

  /** The catalog packaged with the tools. */
  lazy val DEFAULT: MetricCatalog = loadFromResources()

  /**
   * Loads the catalog from the packaged resource.
   *
   * Throws `IllegalStateException` if the resource is missing or cannot be parsed, and
   * `IllegalArgumentException` if the parsed catalog declares no metrics.
   */
  @throws[IllegalStateException]
  @throws[IllegalArgumentException]
  def loadFromResources(): MetricCatalog = {
    // PropertiesLoader.loadFromContent calls validate() once the bean is populated, so a bad
    // declaration surfaces from inside this call. Everything is wrapped so that a parse failure
    // reports the resource path rather than escaping as a bare SnakeYAML exception.
    val catalog = try {
      val source = UTF8Source.fromResource(DEFAULT_CONFIG_PATH)
      val content = try source.mkString finally source.close()
      PropertiesLoader[MetricCatalog].loadFromContent(content).orNull
    } catch {
      case e: IllegalArgumentException => throw e // a validation failure is already specific
      case e: Exception =>
        throw new IllegalStateException(
          s"Could not load the metric catalog: $DEFAULT_CONFIG_PATH", e)
    }
    if (catalog == null || catalog.metrics == null || catalog.metrics.isEmpty) {
      throw new IllegalArgumentException(
        s"The metric catalog declares no metrics: $DEFAULT_CONFIG_PATH")
    }
    catalog
  }
}
