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

package com.nvidia.spark.rapids.tool.planparser.iceberg

import com.nvidia.spark.rapids.tool.planparser.ops.UnsupportedReasonRef

import org.apache.spark.sql.rapids.tool.util.StringUtils

/**
 * cuDF plugin Iceberg GPU support gates evaluated by Iceberg exec parsers.
 *
 * Each parser-facing entry point returns either `None` (gate passes) or
 * `Some(UnsupportedReasonRef)` so callers can plumb the reason via
 * `setUnsupportedReason`.
 *
 * Iceberg-app detection and Iceberg-specific metadata live in `IcebergHelper`.
 */
object IcebergGpuSupport {

  // Allowlist of Iceberg catalogs supported by the cuDF plugin today.
  // TODO: broaden to an Iceberg BaseCatalog class-name match (hive/glue/REST/JDBC)
  // and fix EventUtils.SPARK_CATALOG_REGEX so this gate actually fires.
  private val SUPPORTED_CATALOGS = Set("hadoop")

  // Spark profiles where the cuDF plugin Iceberg provider is available. The plugin ships
  // Iceberg runtimes for Spark 3.5.x and 4.0.x only; other shims use iceberg-stub.
  private val SUPPORTED_SPARK_VERSION_REGEX = """^(?:3\.5|4\.0)\.\d+.*""".r

  // cuDF plugin Iceberg toggles. Defaults are true on the plugin side; treated as disabling
  // only when explicitly set to "false".
  val CONF_ICEBERG_ENABLED: String = "spark.rapids.sql.format.iceberg.enabled"
  val CONF_ICEBERG_WRITE_ENABLED: String = "spark.rapids.sql.format.iceberg.write.enabled"

  // Iceberg sets this on for its own writes; treated as disabling only when the
  // eventlog records an explicit "false".
  val CONF_PARQUET_FIELD_ID_WRITE: String = "spark.sql.parquet.fieldId.write.enabled"

  /**
   * True when the Spark profile is one that ships a real Iceberg provider.
   * An empty version string is treated as "unknown" and allowed, to preserve the
   * existing optimistic behavior in older test paths that don't set sparkVersion.
   */
  def isSparkVersionSupported(sparkVersion: String): Boolean = {
    if (sparkVersion == null || sparkVersion.isEmpty) {
      true
    } else {
      SUPPORTED_SPARK_VERSION_REGEX.pattern.matcher(sparkVersion).matches()
    }
  }

  /**
   * Returns false only when the property is explicitly present and set to "false"
   * (case-insensitive). Mirrors the plugin's default-on toggles.
   */
  private def isConfDisabled(
      properties: collection.Map[String, String], key: String): Boolean = {
    properties.get(key).exists(_.equalsIgnoreCase("false"))
  }

  /** True when `spark.rapids.sql.format.iceberg.enabled` is not explicitly disabled. */
  def isIcebergFormatEnabled(properties: collection.Map[String, String]): Boolean = {
    !isConfDisabled(properties, CONF_ICEBERG_ENABLED)
  }

  /** True when `spark.rapids.sql.format.iceberg.write.enabled` is not explicitly disabled. */
  def isIcebergWriteEnabled(properties: collection.Map[String, String]): Boolean = {
    !isConfDisabled(properties, CONF_ICEBERG_WRITE_ENABLED)
  }

  /**
   * True when Parquet field-ID writes are not explicitly disabled. Iceberg sets this on
   * for its own writes, so we only fail the gate when the eventlog shows it forced to
   * false.
   */
  def isParquetFieldIdWriteEnabled(properties: collection.Map[String, String]): Boolean = {
    !isConfDisabled(properties, CONF_PARQUET_FIELD_ID_WRITE)
  }

  /**
   * True when the extracted Iceberg data-file format is GPU-supported. Accepts only
   * `IcebergParquet`; rejects `IcebergOrc` / `IcebergAvro`. Null, empty, or unknown
   * format strings default to true so callers stay optimistic when the extractor
   * cannot resolve a format.
   */
  def isSupportedDataFileFormat(fmt: String): Boolean = {
    fmt == null || fmt.isEmpty || fmt.equals(StringUtils.UNKNOWN_EXTRACT) ||
      fmt.equalsIgnoreCase("IcebergParquet")
  }

  /**
   * True when the catalog type is in `SUPPORTED_CATALOGS`. Defaults to true when the
   * catalog type cannot be extracted from the properties.
   */
  def isSparkCatalogSupported(properties: collection.Map[String, String]): Boolean = {
    IcebergHelper.getCatalogType(properties) match {
      case Some(catalog) =>
        SUPPORTED_CATALOGS.contains(catalog.toLowerCase)
      case _ =>
        // we could not extract the catalog, default to True.
        true
    }
  }

  /**
   * Returns the first failing write-prerequisite gate as an `UnsupportedReasonRef`,
   * or `None` when all gates pass.
   *
   * Gate order:
   *   1. Databricks platform short-circuit. The cuDF plugin ships
   *      `iceberg-stub` on DBR shims today, so any Databricks Iceberg write is
   *      reported unsupported regardless of other gates. When DBR Iceberg support
   *      is added, this branch should consult a DBR-version predicate instead of
   *      returning unconditionally.
   *   2. Spark version (3.5.x / 4.0.x), only consulted on non-Databricks platforms.
   *      `sparkVersion` here is whatever `AppBase.sparkVersion` resolves to (raw
   *      `SparkListenerLogStart` on OSS Spark, plugin-supplied vendor label on
   *      platforms that override it), so the predicate is only meaningful when
   *      the platform-shape is known.
   *   3. `spark.rapids.sql.format.iceberg.enabled`
   *   4. `spark.rapids.sql.format.iceberg.write.enabled`
   *   5. `spark.sql.parquet.fieldId.write.enabled`
   */
  def firstUnsupportedWritePrerequisite(
      properties: collection.Map[String, String],
      sparkVersion: String,
      isDatabricks: Boolean): Option[UnsupportedReasonRef] = {
    if (isDatabricks) {
      Some(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_DATABRICKS)
    } else if (!isSparkVersionSupported(sparkVersion)) {
      Some(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_SPARK_VERSION)
    } else if (!isIcebergFormatEnabled(properties)) {
      Some(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_DISABLED)
    } else if (!isIcebergWriteEnabled(properties)) {
      Some(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_WRITE_DISABLED)
    } else if (!isParquetFieldIdWriteEnabled(properties)) {
      Some(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_FIELD_IDS)
    } else {
      None
    }
  }

  /**
   * Returns the catalog-gate failure reason or `None` when the catalog passes.
   */
  def firstUnsupportedCatalogReason(
      properties: collection.Map[String, String]): Option[UnsupportedReasonRef] = {
    if (isSparkCatalogSupported(properties)) {
      None
    } else {
      Some(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_CATALOG)
    }
  }
}
