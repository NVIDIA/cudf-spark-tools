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

import com.nvidia.spark.rapids.BaseNoSparkSuite
import com.nvidia.spark.rapids.tool.planparser.ops.UnsupportedReasonRef

/**
 * Pure unit tests for the runtime/config gates on `IcebergGpuSupport`. These do not
 * spin up Spark and only exercise the gate logic that drives PR 2093's AppendData
 * and ReplaceData support flips.
 *
 * The behaviors locked in here mirror the assumptions documented in the PR:
 *   - Spark version regex restricted to 3.5.x and 4.0.x families.
 *   - Default-on cuDF plugin Iceberg toggles fail only when explicitly disabled
 *     (case-insensitive "false").
 *   - Parquet field-ID write toggle treated the same way.
 *   - Catalog allowlist limited to "hadoop" today; missing catalog defaults true.
 */
class IcebergGpuSupportSuite extends BaseNoSparkSuite {

  // ---------------- Spark version gate ----------------

  test("isSparkVersionSupported accepts Spark 3.5.x and 4.0.x families") {
    Seq("3.5.0", "3.5.1", "3.5.7", "3.5.10",
        "4.0.0", "4.0.2", "4.0.10").foreach { v =>
      assert(IcebergGpuSupport.isSparkVersionSupported(v),
        s"expected $v to be a supported Spark version")
    }
  }

  test("isSparkVersionSupported accepts Spark builds with vendor suffix") {
    // Real eventlogs often carry suffixes like '3.5.0-amzn-1' or '3.5.1-databricks-1'.
    // The regex tolerates any trailing characters after the patch number.
    Seq("3.5.0-amzn-1", "3.5.6-databricks-1", "4.0.0-SNAPSHOT").foreach { v =>
      assert(IcebergGpuSupport.isSparkVersionSupported(v),
        s"expected vendor-suffixed $v to be supported")
    }
  }

  test("isSparkVersionSupported rejects older and newer Spark families") {
    Seq("3.2.0", "3.3.4", "3.4.0", "3.4.5",
        "4.1.0", "4.1.1", "5.0.0").foreach { v =>
      assert(!IcebergGpuSupport.isSparkVersionSupported(v),
        s"expected $v to be rejected as unsupported")
    }
  }

  test("isSparkVersionSupported is optimistic for empty or null versions") {
    // The unit test EventLogs sometimes lack a sparkVersion. Stay optimistic so
    // we don't regress existing test paths that don't set it.
    assert(IcebergGpuSupport.isSparkVersionSupported(""),
      "empty version string should be treated as unknown / allowed")
    assert(IcebergGpuSupport.isSparkVersionSupported(null.asInstanceOf[String]),
      "null version should be treated as unknown / allowed")
  }

  // ---------------- Default-on cuDF plugin Iceberg toggles ----------------

  test("isIcebergFormatEnabled defaults to true when the key is absent") {
    assert(IcebergGpuSupport.isIcebergFormatEnabled(Map.empty[String, String]))
  }

  test("isIcebergFormatEnabled returns false only when explicitly disabled") {
    val disabled = Map(IcebergGpuSupport.CONF_ICEBERG_ENABLED -> "false")
    val disabledMixedCase = Map(IcebergGpuSupport.CONF_ICEBERG_ENABLED -> "FALSE")
    val enabled = Map(IcebergGpuSupport.CONF_ICEBERG_ENABLED -> "true")
    val unknownValue = Map(IcebergGpuSupport.CONF_ICEBERG_ENABLED -> "maybe")
    assert(!IcebergGpuSupport.isIcebergFormatEnabled(disabled))
    assert(!IcebergGpuSupport.isIcebergFormatEnabled(disabledMixedCase))
    assert(IcebergGpuSupport.isIcebergFormatEnabled(enabled))
    // Unknown / garbage values fall through to the "not explicitly false" branch.
    assert(IcebergGpuSupport.isIcebergFormatEnabled(unknownValue))
  }

  test("isIcebergWriteEnabled mirrors the format toggle behavior") {
    assert(IcebergGpuSupport.isIcebergWriteEnabled(Map.empty[String, String]))
    assert(!IcebergGpuSupport.isIcebergWriteEnabled(
      Map(IcebergGpuSupport.CONF_ICEBERG_WRITE_ENABLED -> "false")))
    assert(IcebergGpuSupport.isIcebergWriteEnabled(
      Map(IcebergGpuSupport.CONF_ICEBERG_WRITE_ENABLED -> "true")))
  }

  test("isParquetFieldIdWriteEnabled mirrors the toggle behavior") {
    // Spark's own default for this config is "false", but Iceberg forces it on
    // for its own writes. We deliberately only fail when the eventlog records
    // an explicit "false".
    assert(IcebergGpuSupport.isParquetFieldIdWriteEnabled(Map.empty[String, String]))
    assert(!IcebergGpuSupport.isParquetFieldIdWriteEnabled(
      Map(IcebergGpuSupport.CONF_PARQUET_FIELD_ID_WRITE -> "false")))
    assert(IcebergGpuSupport.isParquetFieldIdWriteEnabled(
      Map(IcebergGpuSupport.CONF_PARQUET_FIELD_ID_WRITE -> "true")))
  }

  // ---------------- Data-file format predicate ----------------

  test("isSupportedDataFileFormat accepts IcebergParquet, case-insensitive") {
    assert(IcebergGpuSupport.isSupportedDataFileFormat("IcebergParquet"))
    assert(IcebergGpuSupport.isSupportedDataFileFormat("icebergparquet"))
    assert(IcebergGpuSupport.isSupportedDataFileFormat("ICEBERGPARQUET"))
  }

  test("isSupportedDataFileFormat rejects IcebergOrc and IcebergAvro") {
    assert(!IcebergGpuSupport.isSupportedDataFileFormat("IcebergOrc"))
    assert(!IcebergGpuSupport.isSupportedDataFileFormat("IcebergAvro"))
    // Other non-Parquet labels also rejected.
    assert(!IcebergGpuSupport.isSupportedDataFileFormat("Parquet"))
    assert(!IcebergGpuSupport.isSupportedDataFileFormat("Orc"))
  }

  test("isSupportedDataFileFormat is optimistic on null / empty / unknown") {
    // Mirrors the per-parser fallback: when we cannot resolve a format from the
    // event log we do not regress the verdict.
    assert(IcebergGpuSupport.isSupportedDataFileFormat(null.asInstanceOf[String]))
    assert(IcebergGpuSupport.isSupportedDataFileFormat(""))
    assert(IcebergGpuSupport.isSupportedDataFileFormat(
      org.apache.spark.sql.rapids.tool.util.StringUtils.UNKNOWN_EXTRACT))
  }

  // ---------------- Write-prerequisite cascade ----------------

  test("firstUnsupportedWritePrerequisite short-circuits to Databricks reason on DBR") {
    // Databricks branch fires even when sparkVersion is a valid OSS string and
    // every other gate would pass. The plugin uses iceberg-stub on DBR today.
    val result = IcebergGpuSupport.firstUnsupportedWritePrerequisite(
      properties = Map.empty[String, String],
      sparkVersion = "3.5.7",
      isDatabricks = true)
    assert(result.contains(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_DATABRICKS))
  }

  test("firstUnsupportedWritePrerequisite returns Databricks reason on a DBR label") {
    // When `app.sparkVersion` resolves to a DBR label (e.g. "13.3.x-..."), the
    // Databricks branch short-circuits before the version regex is consulted.
    val result = IcebergGpuSupport.firstUnsupportedWritePrerequisite(
      properties = Map.empty[String, String],
      sparkVersion = "13.3.x-aarch64-scala2.12",
      isDatabricks = true)
    assert(result.contains(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_DATABRICKS))
  }

  test("firstUnsupportedWritePrerequisite passes on OSS Spark with all gates clear") {
    val result = IcebergGpuSupport.firstUnsupportedWritePrerequisite(
      properties = Map.empty[String, String],
      sparkVersion = "3.5.7",
      isDatabricks = false)
    assert(result.isEmpty)
  }

  test("firstUnsupportedWritePrerequisite returns Spark version reason on OSS Spark 3.4") {
    val result = IcebergGpuSupport.firstUnsupportedWritePrerequisite(
      properties = Map.empty[String, String],
      sparkVersion = "3.4.1",
      isDatabricks = false)
    assert(result.contains(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_SPARK_VERSION))
  }

  test("firstUnsupportedWritePrerequisite returns format-disabled reason when gate fails") {
    val props = Map(IcebergGpuSupport.CONF_ICEBERG_ENABLED -> "false")
    val result = IcebergGpuSupport.firstUnsupportedWritePrerequisite(
      properties = props,
      sparkVersion = "3.5.7",
      isDatabricks = false)
    assert(result.contains(UnsupportedReasonRef.UNSUPPORTED_ICEBERG_DISABLED))
  }

  // ---------------- Catalog allowlist ----------------
  //
  // `EventUtils.SPARK_CATALOG_REGEX` is a triple-quoted string containing literal
  // `\\.` sequences, which the regex engine reads as "backslash + any character".
  // Real `spark.sql.catalog.<name>` keys have no backslash, so the regex never
  // matches and `isSparkCatalogSupported` defaults to true for every catalog type
  // today. The tests below lock in that current behavior.
  //
  // TODO: fix the regex and broaden the allowlist; this suite will need per-type
  // assertions once that lands.

  test("isSparkCatalogSupported is effectively always true with the current regex bug") {
    val hadoop = Map(
      "spark.sql.catalog.local" -> "org.apache.iceberg.spark.SparkCatalog",
      "spark.sql.catalog.local.type" -> "hadoop")
    val hive = Map(
      "spark.sql.catalog.local" -> "org.apache.iceberg.spark.SparkCatalog",
      "spark.sql.catalog.local.type" -> "hive")
    val rest = Map(
      "spark.sql.catalog.rest" -> "org.apache.iceberg.spark.SparkCatalog",
      "spark.sql.catalog.rest.type" -> "rest")
    assert(IcebergGpuSupport.isSparkCatalogSupported(hadoop))
    assert(IcebergGpuSupport.isSparkCatalogSupported(hive))
    assert(IcebergGpuSupport.isSparkCatalogSupported(rest))
  }

  test("isSparkCatalogSupported defaults to true when no catalog.type is set") {
    val noTypeKey = Map("spark.sql.catalog.local" -> "org.apache.iceberg.spark.SparkCatalog")
    val emptyProps = Map.empty[String, String]
    assert(IcebergGpuSupport.isSparkCatalogSupported(noTypeKey))
    assert(IcebergGpuSupport.isSparkCatalogSupported(emptyProps))
  }
}
