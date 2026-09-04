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

import scala.collection.JavaConverters._

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.rapids.tool.util.PropertiesLoader

class MetricCatalogSuite extends AnyFunSuite {

  private val catalog = MetricCatalog.DEFAULT

  private def entry(
      name: String,
      family: String = MetricDefinition.Families.Gpu,
      unit: String = MetricDefinition.Units.Count,
      valueForm: String = MetricDefinition.ValueForms.Integer,
      storageScale: Long = 1L,
      aggregation: String = MetricDefinition.Aggregations.Sum,
      source: String = MetricDefinition.Sources.Accumulable): MetricDefinition = {
    new MetricDefinition(name, family, source, unit, valueForm, storageScale, aggregation,
      false, "d")
  }

  /**
   * Builds a catalog directly. Note this validates eagerly: ValidatableProperties calls
   * validate() during construction and the constructor parameter is already visible, so an
   * invalid table throws here rather than on a later validate() call. The SnakeYAML path is
   * different -- it uses the no-arg constructor and populates through the setter, which is why
   * the loader re-validates once deserialization has finished.
   */
  private def catalogOf(entries: MetricDefinition*): MetricCatalog = {
    new MetricCatalog(new java.util.ArrayList[MetricDefinition](entries.asJava))
  }

  test("packaged catalog loads and declares metrics") {
    assert(catalog.metrics != null, "catalog should have been populated by the loader")
    assert(catalog.declaredNames.nonEmpty, "catalog should declare at least one metric")
  }

  test("packaged catalog has no duplicate names") {
    val names = catalog.metrics.asScala.map(_.name)
    assert(names.size == names.distinct.size,
      s"duplicate metric names: ${names.diff(names.distinct).distinct.mkString(", ")}")
  }

  test("every declared entry uses a known vocabulary and a coherent storage scale") {
    catalog.metrics.asScala.foreach { e =>
      assert(e.name.nonEmpty, s"empty metric name in $e")
      assert(MetricDefinition.Families.all.contains(e.family), s"bad family in $e")
      assert(MetricDefinition.Sources.all.contains(e.source), s"bad source in $e")
      assert(MetricDefinition.Units.all.contains(e.unit), s"bad unit in $e")
      assert(MetricDefinition.ValueForms.all.contains(e.valueForm), s"bad valueForm in $e")
      assert(MetricDefinition.Aggregations.all.contains(e.aggregation), s"bad aggregation in $e")
      assert(e.storageScale >= 1L, s"non-positive storageScale in $e")
      if (!e.isDecimalValued) {
        assert(e.storageScale == 1L, s"integer-valued metric must not be scaled: $e")
      }
      assert(e.description != null && e.description.nonEmpty, s"missing description in $e")
    }
  }

  test("lookup resolves a declared metric and rejects an undeclared one") {
    val footprint = catalog.lookup("gpuMaxTaskFootprint")
    assert(footprint.isDefined, "gpuMaxTaskFootprint should be declared")
    assert(footprint.get.unit == MetricDefinition.Units.Bytes,
      "gpuMaxTaskFootprint is measured in bytes even though its name has no 'Bytes'")
    assert(footprint.get.isAggregatedByMax, "gpuMaxTaskFootprint is a per-task high-water mark")
    assert(catalog.lookup("someMetricThatDoesNotExist").isEmpty)
    assert(!catalog.isDeclared("someMetricThatDoesNotExist"))
  }

  test("count metrics whose names contain Wait are not declared as durations") {
    // These are the metrics the previous substring rule mislabelled, because "Wait" is a
    // substring of "Waiting". Their unit is a count.
    Seq("gpuOnGpuTasksWaitingGPUMaxCount",
      "gpuOnGpuTasksWaitingGPUAvgCount",
      "perfio.s3.requestLimiter.maxWaitingRequests").foreach { name =>
      val e = catalog.lookup(name)
      assert(e.isDefined, s"$name should be declared")
      assert(e.get.unit == MetricDefinition.Units.Count, s"$name is a count, not a duration")
    }
  }

  test("the decimal-valued metric is the only one carrying a fixed-point scale") {
    val scaled = catalog.metrics.asScala.filter(_.storageScale != 1L).map(_.name).toSet
    assert(scaled == Set("gpuOnGpuTasksWaitingGPUAvgCount"), s"unexpected scaled metrics: $scaled")
    val decimal = catalog.metrics.asScala.filter(_.isDecimalValued).map(_.name).toSet
    assert(decimal == Set("gpuOnGpuTasksWaitingGPUAvgCount"), s"unexpected decimals: $decimal")
    assert(catalog.lookup("gpuOnGpuTasksWaitingGPUAvgCount").get.storageScale == 1000L)
  }

  test("max-aggregated metrics are exactly the declared set") {
    // Two entries here differ from the hardcoded set this replaces, both deliberately:
    // gpuOnGpuTasksWaitingGPUAvgCount (an average, but a per-task one, so the stage value is the
    // max of per-task averages) and perfio.s3.requestLimiter.maxWaitingRequests (a genuine
    // MaxLongAccumulator that was previously summed).
    val expected = Set(
      "gpuMaxPageableMemoryBytes",
      "gpuMaxDeviceMemoryBytes",
      "gpuMaxHostMemoryBytes",
      "gpuMaxPinnedMemoryBytes",
      "gpuMaxDiskMemoryBytes",
      "gpuMaxTaskFootprint",
      "gpuOnGpuTasksWaitingGPUMaxCount",
      "gpuOnGpuTasksWaitingGPUAvgCount",
      "gpuMaxConcurrentGpuTasks",
      "multithreadReaderMaxParallelism",
      "perfio.s3.requestLimiter.maxWaitingRequests")
    assert(catalog.maxAggregatedNames == expected,
      s"unexpected: ${catalog.maxAggregatedNames.diff(expected)}, " +
        s"missing: ${expected.diff(catalog.maxAggregatedNames)}")
  }

  test("diagnostics metrics are exactly the declared set") {
    val expected = Set(
      "gpuMaxTaskFootprint",
      "gpuMaxDeviceMemoryBytes",
      "gpuMaxHostMemoryBytes",
      "gpuMaxPinnedMemoryBytes",
      "gpuMaxPageableMemoryBytes",
      "gpuMaxDiskMemoryBytes",
      "gpuSpillToHostBytes",
      "gpuSpillToDiskBytes",
      "gpuMaxConcurrentGpuTasks",
      "gpuOnGpuTasksWaitingGPUMaxCount",
      "multithreadReaderMaxParallelism",
      "gpuSemaphoreWait",
      "gpuTime")
    assert(catalog.diagnosticsMetricNames == expected,
      s"unexpected: ${catalog.diagnosticsMetricNames.diff(expected)}, " +
        s"missing: ${expected.diff(catalog.diagnosticsMetricNames)}")
    assert(catalog.diagnosticsMetricNames.subsetOf(catalog.declaredNames))
  }

  test("metrics are split between the gpu and perfio families, and all are gpu-reported") {
    val perfio = catalog.namesInFamily(MetricDefinition.Families.Perfio)
    val gpu = catalog.namesInFamily(MetricDefinition.Families.Gpu)
    assert(perfio == catalog.declaredNames.filter(_.startsWith("perfio.")),
      s"perfio family should be exactly the perfio.* names, got $perfio")
    assert(gpu ++ perfio == catalog.declaredNames, "every metric is gpu or perfio today")
    assert(catalog.namesInFamily(MetricDefinition.Families.Spark).isEmpty)
    // both families land in the gpu_* files, which is what the analyzer asks about
    catalog.declaredNames.foreach { n =>
      assert(catalog.isGpuReportedMetric(n), s"$n should be reported in the gpu files")
    }
  }

  test("isGpuReportedMetric is the one place the discovery rule lives") {
    // declared: family decides, and being declared is not enough
    val c = catalogOf(
      entry("someGpuMetric", family = MetricDefinition.Families.Gpu),
      entry("perfio.x.y", family = MetricDefinition.Families.Perfio),
      entry("someSparkMetric", family = MetricDefinition.Families.Spark))
    assert(c.isGpuReportedMetric("someGpuMetric"))
    assert(c.isGpuReportedMetric("perfio.x.y"))
    assert(c.isDeclared("someSparkMetric"), "precondition: the metric is declared")
    assert(!c.isGpuReportedMetric("someSparkMetric"),
      "a declared spark-family metric must not leak into the gpu files")
    // undeclared: the legacy prefix rule, so a newly added plugin metric is still reported
    assert(c.isGpuReportedMetric("gpuBrandNewMetric"))
    assert(c.isGpuReportedMetric("perfio.gcs.something"))
    assert(c.isGpuReportedMetric("multithreadReaderMaxParallelism"))
    assert(!c.isGpuReportedMetric("internal.metrics.memoryBytesSpilled"))
  }

  test("family queries") {
    assert(catalog.familyOf("gpuMaxTaskFootprint").contains(MetricDefinition.Families.Gpu))
    assert(catalog.familyOf("perfio.s3.netty.executors")
      .contains(MetricDefinition.Families.Perfio))
    assert(catalog.familyOf("neverHeardOfIt").isEmpty)
    assert(catalog.isGpuFamily("gpuMaxTaskFootprint"))
    assert(!catalog.isGpuFamily("perfio.s3.netty.executors"))
    assert(catalog.isPerfioFamily("perfio.s3.netty.executors"))
    assert(!catalog.isPerfioFamily("gpuMaxTaskFootprint"))
  }

  test("validate rejects an unknown family") {
    val ex = intercept[IllegalArgumentException](catalogOf(entry("m", family = "quantum")))
    assert(ex.getMessage.contains("Invalid family"))
  }

  test("validate rejects duplicate metric names") {
    val ex = intercept[IllegalArgumentException](catalogOf(entry("dup"), entry("dup")))
    assert(ex.getMessage.contains("Duplicate metric names"))
  }

  test("validate rejects an unknown unit") {
    val ex = intercept[IllegalArgumentException](catalogOf(entry("m", unit = "furlongs")))
    assert(ex.getMessage.contains("Invalid unit"))
  }

  test("validate rejects an unknown aggregation") {
    val ex = intercept[IllegalArgumentException](catalogOf(entry("m", aggregation = "median")))
    assert(ex.getMessage.contains("Invalid aggregation"))
  }

  test("validate rejects a scale on an integer-valued metric") {
    val ex = intercept[IllegalArgumentException](catalogOf(entry("m", storageScale = 1000L)))
    assert(ex.getMessage.contains("storageScale must be 1"))
  }

  test("validate rejects a non-positive scale") {
    val ex = intercept[IllegalArgumentException](catalogOf(
      entry("m", valueForm = MetricDefinition.ValueForms.Decimal, storageScale = 0L)))
    assert(ex.getMessage.contains("storageScale must be >= 1"))
  }

  test("validate rejects an empty metric name") {
    val ex = intercept[IllegalArgumentException](catalogOf(entry("")))
    assert(ex.getMessage.contains("cannot be null or empty"))
  }

  test("the declared name set is exactly the plugin's task metric map") {
    // Pinned explicitly: without this, deleting a row or typing a metric name to match the
    // plugin's private field rather than its accumulable name leaves the suite green and
    // silently downgrades the metric to the undeclared fallback.
    val expected = Set(
      "gpuTime",
      "gpuSemaphoreWait",
      "gpuRetryCount",
      "gpuSplitAndRetryCount",
      "gpuRetryBlockTime",
      "gpuRetryComputationTime",
      "gpuSpillToHostTime",
      "gpuSpillToDiskTime",
      "gpuReadSpillFromHostTime",
      "gpuReadSpillFromDiskTime",
      "gpuSpillToHostBytes",
      "gpuSpillToDiskBytes",
      "gpuMaxDeviceMemoryBytes",
      "gpuMaxHostMemoryBytes",
      "gpuMaxDiskMemoryBytes",
      "gpuMaxPageableMemoryBytes",
      "gpuMaxPinnedMemoryBytes",
      "gpuOnGpuTasksWaitingGPUAvgCount",
      "gpuOnGpuTasksWaitingGPUMaxCount",
      "gpuMaxTaskFootprint",
      "multithreadReaderMaxParallelism",
      "gpuMaxConcurrentGpuTasks",
      "gpuDiskWriteSavedBytes",
      "perfio.s3.netty.executors",
      "perfio.s3.crt.executors",
      "perfio.s3.s3a.executors",
      "perfio.s3.iceberg.fallbacks",
      "perfio.gcs.http.executors",
      "perfio.gcs.grpc.executors",
      "perfio.s3.requestLimiter.totalWaitTime",
      "perfio.s3.requestLimiter.maxWaitingRequests")
    assert(catalog.declaredNames == expected,
      s"unexpected: ${catalog.declaredNames.diff(expected)}, " +
        s"missing: ${expected.diff(catalog.declaredNames)}")
    assert(catalog.metrics.size == expected.size)
  }

  test("every metric's declared unit is pinned") {
    val millis = Set(
      "gpuTime",
      "gpuSemaphoreWait",
      "gpuRetryBlockTime",
      "gpuRetryComputationTime",
      "gpuSpillToHostTime",
      "gpuSpillToDiskTime",
      "gpuReadSpillFromHostTime",
      "gpuReadSpillFromDiskTime",
      "perfio.s3.requestLimiter.totalWaitTime")
    val bytes = Set(
      "gpuSpillToHostBytes",
      "gpuSpillToDiskBytes",
      "gpuDiskWriteSavedBytes",
      "gpuMaxTaskFootprint",
      "gpuMaxDeviceMemoryBytes",
      "gpuMaxHostMemoryBytes",
      "gpuMaxPinnedMemoryBytes",
      "gpuMaxPageableMemoryBytes",
      "gpuMaxDiskMemoryBytes")
    def named(u: String): Set[String] =
      catalog.metrics.asScala.filter(_.unit == u).map(_.name).toSet
    assert(named(MetricDefinition.Units.Millis) == millis,
      s"ms mismatch: ${named(MetricDefinition.Units.Millis).diff(millis)} / " +
        s"${millis.diff(named(MetricDefinition.Units.Millis))}")
    assert(named(MetricDefinition.Units.Bytes) == bytes,
      s"bytes mismatch: ${named(MetricDefinition.Units.Bytes).diff(bytes)} / " +
        s"${bytes.diff(named(MetricDefinition.Units.Bytes))}")
    // everything else is a count
    assert(named(MetricDefinition.Units.Count) == catalog.declaredNames.diff(millis).diff(bytes))
  }

  test("an undeclared metric falls back to the legacy name heuristic for its unit") {
    // The fallback is label-only: nothing is scaled by it, so a wrong guess costs a column
    // header rather than a value.
    assert(catalog.unitFor("gpuFooTime") == MetricDefinition.Units.Millis)
    assert(catalog.unitFor("gpuFooWait") == MetricDefinition.Units.Millis)
    assert(catalog.unitFor("gpuFooBytes") == MetricDefinition.Units.Bytes)
    assert(catalog.unitFor("gpuFooCount") == MetricDefinition.Units.Count)
    // and everything else defaults conservatively
    Seq("gpuFooTime", "gpuFooBytes", "gpuFooCount").foreach { n =>
      assert(!catalog.isDeclared(n))
      assert(!catalog.isAggregatedByMax(n), s"$n should default to sum")
      assert(!catalog.includedInDiagnostics(n))
      assert(!catalog.isDecimalValued(n))
      assert(catalog.storageScaleFor(n) == 1L)
    }
  }

  test("the legacy fallback agrees with the declaration for convention-following names") {
    // Where a declared name follows the plugin's naming convention the fallback would have got
    // the unit right anyway; the interesting rows are the ones where it would not.
    val disagreeing = catalog.metrics.asScala
      .filter(e => MetricCatalog.legacyUnitFor(e.name) != e.unit)
      .map(e => s"${e.name}: declared ${e.unit}, heuristic ${MetricCatalog.legacyUnitFor(e.name)}")
      .toSet
    assert(disagreeing == Set(
      "gpuMaxTaskFootprint: declared bytes, heuristic count",
      "gpuOnGpuTasksWaitingGPUMaxCount: declared count, heuristic ms",
      "gpuOnGpuTasksWaitingGPUAvgCount: declared count, heuristic ms",
      "perfio.s3.requestLimiter.maxWaitingRequests: declared count, heuristic ms"),
      s"unexpected disagreements: $disagreeing")
  }

  test("the single-lookup accessors agree with the declared entries") {
    catalog.metrics.asScala.foreach { e =>
      assert(catalog.unitFor(e.name) == e.unit, s"unitFor disagrees for ${e.name}")
      assert(catalog.isAggregatedByMax(e.name) == e.isAggregatedByMax,
        s"isAggregatedByMax disagrees for ${e.name}")
      assert(catalog.includedInDiagnostics(e.name) == e.includeInDiagnostics,
        s"includedInDiagnostics disagrees for ${e.name}")
      assert(catalog.storageScaleFor(e.name) == e.storageScale,
        s"storageScaleFor disagrees for ${e.name}")
      assert(catalog.isDecimalValued(e.name) == e.isDecimalValued,
        s"isDecimalValued disagrees for ${e.name}")
    }
  }

  test("a YAML table with an omitted field is rejected rather than silently defaulted") {
    // SnakeYAML leaves an omitted key at the no-arg constructor's default, so the vocabulary
    // fields default to null in order to be caught here.
    val yaml =
      """|metrics:
         |  - name: someMetric
         |    family: gpu
         |    source: accumulable
         |    unit: count
         |    valueForm: integer
         |    storageScale: 1
         |    includeInDiagnostics: false
         |    description: aggregation is missing on purpose
         |""".stripMargin
    val ex = intercept[IllegalArgumentException](
      PropertiesLoader[MetricCatalog].loadFromContent(yaml))
    assert(ex.getMessage.contains("Invalid aggregation"), ex.getMessage)
  }

  test("a YAML table with a bad vocabulary value is rejected on load") {
    val yaml =
      """|metrics:
         |  - name: someMetric
         |    family: gpu
         |    source: accumulable
         |    unit: furlongs
         |    valueForm: integer
         |    storageScale: 1
         |    aggregation: sum
         |    includeInDiagnostics: false
         |    description: bad unit
         |""".stripMargin
    val ex = intercept[IllegalArgumentException](
      PropertiesLoader[MetricCatalog].loadFromContent(yaml))
    assert(ex.getMessage.contains("Invalid unit"), ex.getMessage)
  }

  test("validate rejects an entry with no description") {
    val e = new MetricDefinition("m", MetricDefinition.Families.Gpu,
      MetricDefinition.Sources.Accumulable, MetricDefinition.Units.Count,
      MetricDefinition.ValueForms.Integer, 1L, MetricDefinition.Aggregations.Sum, false, "")
    val ex = intercept[IllegalArgumentException](catalogOf(e))
    assert(ex.getMessage.contains("Description cannot be null or empty"))
  }

  test("formatStoredValue divides out a fixed-point scale") {
    val f = MetricCatalog.formatStoredValue _
    // unscaled metrics render exactly as before
    assert(f(12345L, 1L) == "12345")
    assert(f(0L, 1L) == "0")
    assert(f(-7L, 1L) == "-7")
    // scaled: integral results carry no fractional part, fractions are trimmed not padded
    assert(f(0L, 1000L) == "0")
    assert(f(1000L, 1000L) == "1")
    assert(f(2500L, 1000L) == "2.5")
    assert(f(714L, 1000L) == "0.714")
    assert(f(1200L, 1000L) == "1.2")
    assert(f(100L, 1000L) == "0.1")
    assert(f(10L, 1000L) == "0.01")
    assert(f(1L, 1000L) == "0.001")
    assert(f(20000L, 1000L) == "20")
    // negatives, including the case where the whole part truncates to zero and would lose the sign
    assert(f(-1500L, 1000L) == "-1.5")
    assert(f(-500L, 1000L) == "-0.5")
    assert(f(-1000L, 1000L) == "-1")
  }

  test("formatStoredValue is locale independent") {
    // String.format and the f-interpolator use the default Locale: under de_DE they render
    // "0,714", and a comma inside a comma-delimited CSV shifts every later column.
    val original = java.util.Locale.getDefault
    try {
      Seq(java.util.Locale.GERMANY, java.util.Locale.FRANCE, java.util.Locale.US).foreach { loc =>
        java.util.Locale.setDefault(loc)
        assert(MetricCatalog.formatStoredValue(714L, 1000L) == "0.714", s"broken under $loc")
        assert(MetricCatalog.formatStoredValue(2500L, 1000L) == "2.5", s"broken under $loc")
        assert(!MetricCatalog.formatStoredValue(714L, 1000L).contains(","), s"comma under $loc")
      }
    } finally {
      java.util.Locale.setDefault(original)
    }
  }

  test("validate rejects a decimal metric that declares no scale") {
    // Without a scale the value is routed to the integer parser and every non-integral sample is
    // dropped -- the original defect, silently reintroduced.
    val ex = intercept[IllegalArgumentException](catalogOf(
      entry("m", valueForm = MetricDefinition.ValueForms.Decimal, storageScale = 1L)))
    assert(ex.getMessage.contains("storageScale must be > 1"))
  }

  test("validate rejects a scale that is not a power of ten") {
    // formatStoredValue derives its fractional digit count from the scale's digit count.
    val ex = intercept[IllegalArgumentException](catalogOf(
      entry("m", valueForm = MetricDefinition.ValueForms.Decimal, storageScale = 1024L)))
    assert(ex.getMessage.contains("power of ten"))
    assert(MetricCatalog.isPowerOfTen(1000L) && MetricCatalog.isPowerOfTen(1L))
    assert(!MetricCatalog.isPowerOfTen(1024L) && !MetricCatalog.isPowerOfTen(500L))
  }

  test("the legacy discovery prefix is perfio., not perfio.s3.") {
    // Widened from the deleted isGpuMetric so an undeclared perfio.gcs.* or perfio.abfs.*
    // metric is discoverable rather than silently dropped. Pinned because two shipped
    // documents describe this rule.
    assert(MetricCatalog.matchesLegacyGpuPrefix("perfio.s3.brand.new"))
    assert(MetricCatalog.matchesLegacyGpuPrefix("perfio.gcs.brand.new"))
    assert(MetricCatalog.matchesLegacyGpuPrefix("perfio.abfs.brand.new"))
    assert(MetricCatalog.matchesLegacyGpuPrefix("gpuBrandNew"))
    assert(MetricCatalog.matchesLegacyGpuPrefix("multithreadReaderMaxParallelism"))
    assert(!MetricCatalog.matchesLegacyGpuPrefix("perfioNoDot"))
    assert(!MetricCatalog.matchesLegacyGpuPrefix("internal.metrics.memoryBytesSpilled"))
    assert(!MetricCatalog.matchesLegacyGpuPrefix("GPU decode time"))
  }

  test("the packaged catalog renders its one decimal metric correctly") {
    val name = "gpuOnGpuTasksWaitingGPUAvgCount"
    assert(catalog.formatValue(name, 2500L) == "2.5")
    assert(catalog.formatValue(name, 714L) == "0.714")
    // and an unscaled metric is untouched
    assert(catalog.formatValue("gpuMaxTaskFootprint", 7123115846L) == "7123115846")
  }
}
