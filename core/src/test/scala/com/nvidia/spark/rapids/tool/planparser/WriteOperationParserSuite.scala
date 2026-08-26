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

package com.nvidia.spark.rapids.tool.planparser

import com.nvidia.spark.rapids.tool.planparser.delta.DeltaLakeHelper
import com.nvidia.spark.rapids.tool.planparser.iceberg.IcebergOps
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.rapids.tool.plangraph.SparkPlanGraphNode
import org.apache.spark.sql.rapids.tool.store.{CompressionCodec, WriteOperationMetadataTrait}
import org.apache.spark.sql.rapids.tool.util.StringUtils


class WriteOperationParserSuite extends AnyFunSuite {

  /**
   * Helper method to test `getWriteOpMetaFromNode`.
   *
   * @param node The input `SparkPlanGraphNode`.
   * @param expectedExecName The expected execution name.
   * @param expectedDataFormat The expected data format.
   * @param expectedOutputPath The expected output path (optional).
   * @param expectedOutputColumns The expected output columns (optional).
   * @param expectedWriteMode The expected write mode.
   * @param expectedTableName The expected table name.
   * @param expectedDatabaseName The expected database name.
   */
  private def testGetWriteOpMetaFromNode(
    node: SparkPlanGraphNode,
    expectedExecName: String,
    expectedDataFormat: String,
    expectedOutputPath: String,
    expectedOutputColumns: String,
    expectedWriteMode: String,
    expectedTableName: String,
    expectedDatabaseName: String,
    expectedPartitionCols: String,
    expectedCompressionOpt: String = CompressionCodec.UNCOMPRESSED): Unit = {

    val metadata: WriteOperationMetadataTrait =
      IcebergOps.extractOpMeta(node, confProvider = None)
        .getOrElse(DataWritingCommandExecParser.getWriteOpMetaFromNode(node))

    assert(metadata.execName() == expectedExecName, "execName")
    assert(metadata.dataFormat() == expectedDataFormat, "dataFormat")
    assert(metadata.outputPath() == expectedOutputPath, "outputPath")
    assert(metadata.outputColumns() == expectedOutputColumns, "outputColumns")
    assert(metadata.writeMode() == expectedWriteMode, "writeMode")
    assert(metadata.table() == expectedTableName, "tableName")
    assert(metadata.dataBase() == expectedDatabaseName, "databaseName")
    assert(metadata.partitions() == expectedPartitionCols, "partitionCols")
    assert(metadata.compressOption() == expectedCompressionOpt, "compressOptions")
  }

  // scalastyle:off line.size.limit
  test("getWriteOpMetaFromNode  — Unknown command") {
    // unknown cmd should return unknown values
    val node = new SparkPlanGraphNode(
      id = 2,
      name = "UnknownWrite",
      desc = "Some random description",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = StringUtils.UNKNOWN_EXTRACT,
      expectedDataFormat = StringUtils.UNKNOWN_EXTRACT,
      expectedOutputPath = StringUtils.UNKNOWN_EXTRACT,
      expectedOutputColumns = StringUtils.UNKNOWN_EXTRACT,
      expectedWriteMode = StringUtils.UNKNOWN_EXTRACT,
      expectedTableName = StringUtils.UNKNOWN_EXTRACT,
      expectedDatabaseName = StringUtils.UNKNOWN_EXTRACT,
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("InsertIntoHadoopFsRelationCommand — No CatalogMeta") {
    // the writeOp writes into a table, without "catalogTable" information
    val node = new SparkPlanGraphNode(
      id = 1,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand gs://path/to/database/table1, " +
        "false, Parquet, " +
        "[serialization.format=1, mergeschema=false, __hive_compatible_bucketed_table_insertion__=true], " +
        "Append, `spark_catalog`.`database`.`table1`, org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
        "org.apache.spark.sql.execution.datasources.InMemoryFileIndex(gs://path/to/database/table1), " +
        "[col01, col02, col03]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "Parquet",
      expectedOutputPath = "gs://path/to/database/table1",
      expectedOutputColumns = "col01;col02;col03",
      expectedWriteMode = "Append",
      expectedTableName = "table1",
      expectedDatabaseName = "database",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("InsertIntoHadoopFsRelationCommand — Format is 4th element") {
    // verify that the format is extracted correctly when the format is the 4th element in the
    // arguments list.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand gs://path/to/database/table1, " +
        "false, [paths=(path)], Parquet, " +
        "[serialization.format=1, mergeschema=false, __hive_compatible_bucketed_table_insertion__=true], " +
        "Append, `spark_catalog`.`database`.`table1`, org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
        "org.apache.spark.sql.execution.datasources.InMemoryFileIndex(gs://path/to/database/table1), " +
        "[col01]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "Parquet",
      expectedOutputPath = "gs://path/to/database/table1",
      expectedOutputColumns = "col01",
      expectedWriteMode = "Append",
      expectedTableName = "table1",
      expectedDatabaseName = "database",
      expectedPartitionCols = "paths=(path)"
    )
  }

  test("InsertIntoHadoopFsRelationCommand — Empty output columns") {
    // empty output columns appear as empty string.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand gs://path/to/database/table1, " +
        "false, Parquet, " +
        "[serialization.format=1, mergeschema=false, __hive_compatible_bucketed_table_insertion__=true], " +
        "Append, `database`.`table1`, org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
        "org.apache.spark.sql.execution.datasources.InMemoryFileIndex(gs://path/to/database/table1), " +
        "[]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "Parquet",
      expectedOutputPath = "gs://path/to/database/table1",
      expectedOutputColumns = "",
      expectedWriteMode = "Append",
      expectedTableName = "table1",
      expectedDatabaseName = "database",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("InsertIntoHadoopFsRelationCommand — Long schema") {
    // Long schema shows up as ellipses in the description.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand gs://path/to/database/table1, " +
        "false, [paths=(path)], Parquet, " +
        "[serialization.format=1, mergeschema=false, __hive_compatible_bucketed_table_insertion__=true], " +
        "Append, `spark_catalog`.`database`.`table1`, org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
        "org.apache.spark.sql.execution.datasources.InMemoryFileIndex(gs://path/to/database/table1), " +
        "[col01, col02, col03, ... 4 more fields]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "Parquet",
      expectedOutputPath = "gs://path/to/database/table1",
      expectedOutputColumns = "col01;col02;col03;... 4 more fields",
      expectedWriteMode = "Append",
      expectedTableName = "table1",
      expectedDatabaseName = "database",
      expectedPartitionCols = "paths=(path)"
    )
  }

  test("InsertIntoHadoopFsRelationCommand — CatalogEntry and Partitions") {
    // Long schema shows up as ellipses in the description.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand gs://path/to/metastore/database1.db/tableName, " +
        "false, [pcol_00#298], Parquet, [serialization.format=1, mergeSchema=false, " +
        "__hive_compatible_bucketed_table_insertion__=true, \npartitionOverwriteMode=DYNAMIC], " +
        "Overwrite, CatalogTable(\n" +
        "Database: database1\nTable: tableName\nOwner: spark\n" +
        "Created Time: Wed Feb 19 00:07:14 UTC 2025\nLast Access: UNKNOWN\nCreated By: Spark 3.5.1\n" +
        "Type: MANAGED\nProvider: hive\nTable Properties: [transient_lastDdlTime=1739923634]\n" +
        "Location: gs://path/to/metastore/database1.db/tableName\n" +
        "Serde Library: org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe\n" +
        "InputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat\n" +
        "OutputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat\n" +
        "Storage Properties: [serialization.format=1]\nPartition Provider: Catalog\n" +
        "Partition Columns: [`pcol_00`]\nSchema: root\n |-- col_00: string (nullable = true)\n" +
        " |-- col_01: string (nullable = true)\n |-- col_02: integer (nullable = true)\n" +
        " |-- col_03: decimal(18,2) (nullable = true)\n |-- col_04: string (nullable = true)\n" +
        " |-- col_05: string (nullable = true)\n |-- col_06: integer (nullable = true)\n" +
        " |-- col_07: decimal(18,2) (nullable = true)\n |-- col_08: string (nullable = true)\n" +
        " |-- col_09: string (nullable = true)\n |-- col_10: integer (nullable = true)\n" +
        " |-- col_11: decimal(18,2) (nullable = true)\n |-- col_12: decimal(18,2) (nullable = true)\n" +
        " |-- col_13: long (nullable = true)\n |-- pcol_00: string (nullable = true)\n" +
        "), org.apache.spark.sql.execution.datasources.CatalogFileIndex@1eaf6b4e, " +
        "[col_00, col_01, col_02, col_03, col_04, col_05, col_06, col_07, col_08, col_09, col_10, " +
        "col_11, col_12, col_13, pcol_00]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "HiveParquet",
      expectedOutputPath = "gs://path/to/metastore/database1.db/tableName",
      expectedOutputColumns = "col_00;col_01;col_02;col_03;col_04;col_05;col_06;col_07;col_08;col_09;col_10;col_11;col_12;col_13;pcol_00",
      expectedWriteMode = "Overwrite",
      expectedTableName = "tableName",
      expectedDatabaseName = "database1",
      expectedPartitionCols = "pcol_00"
    )
  }

  test("InsertIntoHadoopFsRelationCommand — With Compression Codec") {
    // Tests CPU eventlog that has defined compression codec in the options
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand " +
        "file:/path/to/outparquet, " +
        "false, " +
        "[age#11, score#12], " +
        "Parquet, " +
        "[compression=zstd, __partition_columns=[\"age\",\"score\"], path=/path/to/outparquet], " +
        "ErrorIfExists, " +
        "[name, age, score]",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "Parquet",
      expectedOutputPath = "file:/path/to/outparquet",
      expectedOutputColumns = "name;age;score",
      expectedWriteMode = "ErrorIfExists",
      expectedTableName = StringUtils.INAPPLICABLE_EXTRACT,
      expectedDatabaseName = StringUtils.INAPPLICABLE_EXTRACT,
      expectedPartitionCols = "age#11, score#12",
      expectedCompressionOpt = "zstd"
    )
  }

  test("InsertIntoHadoopFsRelationCommand — With ORC and Compression Codec") {
    // Tests CPU eventlog that has defined compression codec in the options.
    // This tests the InsertHadoopFsRelationCommand with hive ORC serde library.
    // In addition it tests that the compression is enabled to SNAPPY
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHadoopFsRelationCommand",
      desc = "Execute InsertIntoHadoopFsRelationCommand " +
        "file:/path/to/orc_hive_table, " +
        "false, " +
        "ORC, " +
        "[orc.compress=SNAPPY, serialization.format=1, " +
        "__hive_compatible_bucketed_table_insertion__=true], " +
        "Append, " +
        "`spark_catalog`.`default`.`my_compressed_orc_table_sql`, " +
        "org.apache.hadoop.hive.ql.io.orc.OrcSerde, " +
        "org.apache.spark.sql.execution.datasources.InMemoryFileIndex(" +
        "file:/path/to/orc_hive_table), " +
        "[name, id]",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "ORC",
      expectedOutputPath = "file:/path/to/orc_hive_table",
      expectedOutputColumns = "name;id",
      expectedWriteMode = "Append",
      expectedTableName = "my_compressed_orc_table_sql",
      expectedDatabaseName = "default",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT,
      expectedCompressionOpt = "snappy"
    )
  }

  test("GpuInsertIntoHadoopFsRelationCommand — Broken RAPIDS formats") {
    // The GPU eventlog may have incorrect formats
    // https://github.com/NVIDIA/cudf-spark-tools/issues/1561
    // this test ensures that the tools can tolerate the broken formats by converting them to
    // proper formats.
    val testFileFormats = Seq(
      ("com.nvidia.spark.rapids.GpuParquetFileFormat@9f5022c", "Parquet"),
      ("com.nvidia.spark.rapids.GpuOrcFileFormat@123abc", "Orc"),
      ("com.nvidia.spark.rapids.GpuHiveTextFileFormat@123abc", "HiveText"),
      ("com.nvidia.spark.rapids.GpuHiveParquetFileFormat@123abc", "HiveParquet"),
      ("com.nvidia.spark.rapids.GpuDeltaFileFormat@123abc", "Delta")
    )
    testFileFormats.foreach { case (format, expectedDataFormat) =>
      val node = new SparkPlanGraphNode(
        id = 1,
        name = "Execute GpuInsertIntoHadoopFsRelationCommand",
        desc = s"Execute GpuInsertIntoHadoopFsRelationCommand gs://path/to/database/table1, " +
          s"false, $format, " +
          "[serialization.format=1, mergeschema=false, __hive_compatible_bucketed_table_insertion__=true], " +
          "Append, `spark_catalog`.`database`.`table1`, org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
          "org.apache.spark.sql.execution.datasources.InMemoryFileIndex(gs://path/to/database/table1), " +
          "[col01, col02, col03]",
        Seq.empty
      )
      testGetWriteOpMetaFromNode(
        node,
        expectedExecName = "GpuInsertIntoHadoopFsRelationCommand",
        expectedDataFormat = expectedDataFormat,
        expectedOutputPath = "gs://path/to/database/table1",
        expectedOutputColumns = "col01;col02;col03",
        expectedWriteMode = "Append",
        expectedTableName = "table1",
        expectedDatabaseName = "database",
        expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
      )
    }
  }

  test("GpuInsertIntoHadoopFsRelationCommand — Gpu log with CatalogTable entry") {
    // This test ensures that we can parse GpuInsert when there is a CatalogTable defined.
    // The format in this test should be HiveParquet because we extract the format from the serDe
    // library.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute GpuInsertIntoHadoopFsRelationCommand",
      desc = "Execute GpuInsertIntoHadoopFsRelationCommand gs://path/to/metastore/databaseName/table1, " +
        "false, [p0#23, p1#30], com.nvidia.spark.rapids.GpuParquetFileFormat@4be4680b, " +
        "[serialization.format=1, mergeSchema=false, __hive_compatible_bucketed_table_insertion__=true, " +
        "partitionOverwriteMode=DYNAMIC], " +
        "Overwrite, " +
        "CatalogTable(\nDatabase: databaseName\nTable: table1\nOwner: root\n" +
        "Created Time: Tue Feb 25 16:58:00 UTC 2025\\nLast Access: UNKNOWN\nCreated By: Spark 3.3.2\n" +
        "Type: EXTERNAL\nProvider: hive\nTable Properties: [transient_lastDdlTime=1740502680]\n" +
        "Location: gs://path/to/metastore/databaseName/table1\n" +
        "Serde Library: org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe\n" +
        "InputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat\n" +
        "OutputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat\n" +
        "Storage Properties: [serialization.format=1]\\nPartition Provider: Catalog\n" +
        "Partition Columns: [`p0`, `p1`]\n" +
        "Schema: root\n |-- col_00: string (nullable = true)\\n" +
        " |-- col_01: string (nullable = true)\n |-- col_02: string (nullable = true)\n" +
        " |-- col_03: double (nullable = true)\n |-- col_04: integer (nullable = true)\n" +
        " |-- col_05: double (nullable = true)\n |-- col_06: decimal (2, 4)\n" +
        " |-- p0: string (nullable = true)\n" +
        " |-- p1: string (nullable = true)\n), " +
        "org.apache.spark.sql.execution.datasources.CatalogFileIndex@8fc69d8f, " +
        "[col_00, col_01, col_02, col_03, col_04, col_05, col_06, p0, p1], false, 0",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "GpuInsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "HiveParquet",
      expectedOutputPath = "gs://path/to/metastore/databaseName/table1",
      expectedOutputColumns = "col_00;col_01;col_02;col_03;col_04;col_05;col_06;p0;p1",
      expectedWriteMode = "Overwrite",
      expectedTableName = "table1",
      expectedDatabaseName = "databaseName",
      expectedPartitionCols = "p0;p1"
    )
  }

  test("GpuInsertIntoHadoopFsRelationCommand — Insert into file (no Database/Table)") {
    // This tests when an exec does not have a table. Instead, it writes immediately to a file.
    // In that case, the write operation should not have a database/table names.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute GpuInsertIntoHadoopFsRelationCommand",
      desc = "Execute GpuInsertIntoHadoopFsRelationCommand " +
        "file:/tmp/local/file/random-uuid, false, " +
        "com.nvidia.spark.rapids.GpuParquetFileFormat@421e46ed, " +
        "[path=/tmp/local/file/random-uuid/], " +
        "Overwrite, [age, name], false, 0",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "GpuInsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "Parquet",
      expectedOutputPath = "file:/tmp/local/file/random-uuid",
      expectedOutputColumns = "age;name",
      expectedWriteMode = "Overwrite",
      expectedTableName = StringUtils.INAPPLICABLE_EXTRACT,
      expectedDatabaseName = StringUtils.INAPPLICABLE_EXTRACT,
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("GpuInsertIntoHadoopFsRelationCommand — CatalogTable entry with path enclosed in brackets") {
    // Tests GPU eventlog that has defined a path between brackets
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute GpuInsertIntoHadoopFsRelationCommand",
      desc = "Execute GpuInsertIntoHadoopFsRelationCommand " +
        "file:/home/work/spark_test/data/temporary_test.db/table1, " +
        "false, com.nvidia.spark.rapids.GpuParquetFileFormat@406479dc, " +
        "[path=file:/home/work/spark_test/data/temporary_test.db/table1], " +
        "Overwrite, CatalogTable(\n" +
        "Database: temporary_test\nTable: table1\nOwner: turing\n" +
        "Created Time: Fri Sep 01 16:31:17 UTC 2023\\nLast Access: UNKNOWN\n" +
        "Created By: Spark 3.3.1\nType: MANAGED\nProvider: parquet\n" +
        "Location: file:/home/work/spark_test/data/temporary_test.db/table1\n" +
        "Serde Library: org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe\n" +
        "InputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat\n" +
        "OutputFormat: org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat\n" +
        "Schema: root\n" +
        " |-- _c0: integer (nullable = false)\n" +
        "), org.apache.spark.sql.execution.datasources.InMemoryFileIndex@f0ed3e64, [_c0], false, 0",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "GpuInsertIntoHadoopFsRelationCommand",
      expectedDataFormat = "HiveParquet",
      expectedOutputPath = "file:/home/work/spark_test/data/temporary_test.db/table1",
      expectedOutputColumns = "_c0",
      expectedWriteMode = "Overwrite",
      expectedTableName = "table1",
      expectedDatabaseName = "temporary_test",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("AppendDataExecV1 — delta format") {
    val node = new SparkPlanGraphNode(
      id = 3,
      name = "AppendDataExecV1",
      // the description should include Delta keywords; otherwise it would be considered a spark Op.
      desc =
        s"""|AppendDataExecV1 [num_affected_rows#18560L, num_inserted_rows#18561L], DeltaTableV2(org.apache.spark.sql.SparkSession@5aa5327e,abfss://abfs_path,Some(CatalogTable(
            |Catalog: spark_catalog
            |Database: database
            |Table: tableName
            |Owner: root
            |Created Time: Wed Sep 15 16:47:47 UTC 2021
            |Last Access: UNKNOWN
            |Created By: Spark 3.1.1
            |Type: EXTERNAL
            |Provider: delta
            |Table Properties: [bucketing_version=2, delta.lastCommitTimestamp=1631724453000, delta.lastUpdateVersion=0, delta.minReaderVersion=1, delta.minWriterVersion=2]
            |Location: abfss://abfs_path
            |Serde Library: org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe
            |InputFormat: org.apache.hadoop.mapred.SequenceFileInputFormat
            |OutputFormat: org.apache.hadoop.hive.ql.io.HiveSequenceFileOutputFormat
            |Partition Provider: Catalog
            |Schema: root
            | |-- field_00: string (nullable = true)
            | |-- field_01: string (nullable = true)
            | |-- field_02: string (nullable = true)
            | |-- field_03: string (nullable = true)
            | |-- field_04: string (nullable = true)
            | |-- field_05: string (nullable = true)
            | |-- field_06: string (nullable = true)
            | |-- field_07: string (nullable = true)
            | |-- field_08: string (nullable = true)
            |)),Some(spark_catalog.adl.tableName),None,Map()), Project [from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_00#15200, 20240112 AS field_01#15201, load_func00 AS field_02#15202, completed AS field_03#15203, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_04#15204, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC)) AS field_05#15205, ddsdmsp AS field_06#15206, rename_01 AS field_07#15207, from_unixtime(unix_timestamp(current_timestamp(), yyyy-MM-dd HH:mm:ss, Some(Etc/UTC), false), yyyyMMdd, Some(Etc/UTC)) AS field_08#15208], org.apache.spark.sql.execution.datasources.v2.DataSourceV2Strategy$$$$Lambda$$12118/1719387317@5a111bfa, com.databricks.sql.transaction.tahoe.catalog.WriteIntoDeltaBuilder$$$$anon$$1@24257336
            |""".stripMargin,
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "AppendDataExecV1",
      expectedDataFormat = DeltaLakeHelper.getWriteFormat, // Special handling for DeltaLake
      expectedOutputPath = StringUtils.UNKNOWN_EXTRACT,
      expectedOutputColumns = StringUtils.UNKNOWN_EXTRACT,
      expectedWriteMode = StringUtils.UNKNOWN_EXTRACT,
      expectedTableName = StringUtils.UNKNOWN_EXTRACT,
      expectedDatabaseName = StringUtils.UNKNOWN_EXTRACT,
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("InsertIntoHiveTable cmd — No catalog prefix") {
    // The catalog piece has `database`.`table`
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHiveTable",
      desc = "Execute InsertIntoHiveTable `database1`.`table1`, " +
        "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe, [dt=Some(2025-01-01)], true, false, " +
        "[col1, col2, col3]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHiveTable",
      expectedDataFormat = "HiveText",
      expectedOutputPath = StringUtils.INAPPLICABLE_EXTRACT,
      expectedOutputColumns = "col1;col2;col3",
      expectedWriteMode = "Overwrite",
      expectedTableName = "table1",
      expectedDatabaseName = "database1",
      expectedPartitionCols = "dt=Some(2025-01-01)",
      expectedCompressionOpt = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("InsertIntoHiveTable cmd — With compression codec") {
    // The plan resulting from inserting into a Hive table with Snappy compression and Parquet
    // format
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHiveTable",
      desc = "Execute InsertIntoHiveTable `spark_catalog`.`default`.`hive_table_with_snappy`, " +
        "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
        "false, " +
        "false, " +
        "[name, id], " +
        "org.apache.spark.sql.hive.execution.HiveFileFormat@7f93fce1, " +
        "org.apache.spark.sql.hive.execution.HiveTempPath@5457123a",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHiveTable",
      expectedDataFormat = "HiveParquet",
      expectedOutputPath = StringUtils.INAPPLICABLE_EXTRACT,
      expectedOutputColumns = "name;id",
      expectedWriteMode = "Append",
      expectedTableName = "hive_table_with_snappy",
      expectedDatabaseName = "default",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT,
      expectedCompressionOpt = StringUtils.UNKNOWN_EXTRACT
    )
  }


  test("InsertIntoHiveTable cmd — Catalog prefix") {
    // The catalog piece has `spark_catalog`.`database`.`table`
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHiveTable",
      desc = "Execute InsertIntoHiveTable `spark_catalog`.`database1`.`table1`, " +
        "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe, [dt=Some(2025-01-01)], false, false, " +
        "[col1, col2, col3]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHiveTable",
      expectedDataFormat = "HiveText",
      expectedOutputPath = StringUtils.INAPPLICABLE_EXTRACT,
      expectedOutputColumns = "col1;col2;col3",
      expectedWriteMode = "Append",
      expectedTableName = "table1",
      expectedDatabaseName = "database1",
      expectedPartitionCols = "dt=Some(2025-01-01)",
      expectedCompressionOpt = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("InsertIntoHiveTable cmd — Dynamic partitions") {
    // The partitions argument is missing
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute InsertIntoHiveTable",
      desc = "Execute InsertIntoHiveTable `spark_catalog`.`database1`.`table1`, " +
        "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe, false, True, " +
        "[col1, col2, col3]",
      Seq.empty
    )
    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "InsertIntoHiveTable",
      expectedDataFormat = "HiveText",
      expectedOutputPath = StringUtils.INAPPLICABLE_EXTRACT,
      expectedOutputColumns = "col1;col2;col3",
      expectedWriteMode = "Append",
      expectedTableName = "table1",
      expectedDatabaseName = "database1",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT,
      expectedCompressionOpt = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("GpuInsertIntoHiveTable — Gpu logs profiler case") {
    // The GPU eventlog has 2 different format arguments. For now, we pick the serDe library.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "Execute GpuInsertIntoHiveTable",
      desc = "Execute GpuInsertIntoHiveTable `database1`.`table1`, " +
        "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe, " +
        "[date=Some(20240621)], " +
        "org.apache.spark.sql.hive.rapids.GpuHiveParquetFileFormat@3dcbd2c4, true, false, " +
        "[col_00, col_01, col_02, col_03, col_04, col_05, col_06, col_07, col_08, col_09, col_10, " +
        "col_11, col_12, col_13, col_14, col_15, col_16, col_17, col_18, col_19, col_20, col_21, " +
        "col_22, col_23, ... 122 more fields]",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "GpuInsertIntoHiveTable",
      expectedDataFormat = "HiveParquet",
      expectedOutputPath = StringUtils.INAPPLICABLE_EXTRACT,
      expectedOutputColumns = "col_00;col_01;col_02;col_03;col_04;col_05;col_06;col_07;col_08;col_09;" +
        "col_10;col_11;col_12;col_13;col_14;col_15;col_16;col_17;col_18;col_19;col_20;col_21;" +
        "col_22;col_23;... 122 more fields",
      expectedWriteMode = "Overwrite",
      expectedTableName = "table1",
      expectedDatabaseName = "database1",
      expectedPartitionCols = "date=Some(20240621)",
      expectedCompressionOpt = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("AppendData — Iceberg table Parquet format") {
    // Test that AppendData is parsed correctly for Iceberg tables.
    val node = new SparkPlanGraphNode(
      id = 5,
      name = "AppendData",
      desc = "AppendData org.apache.spark.sql.execution.datasources.v2.DataSourceV2Strategy$$Lambda$2293/1470843699@46290193, " +
        "IcebergWrite(table=local.db.my_iceberg_table, format=PARQUET)",
      Seq.empty
    )

    testGetWriteOpMetaFromNode(
      node,
      expectedExecName = "AppendData",
      expectedDataFormat = "IcebergParquet",
      expectedOutputPath = StringUtils.UNKNOWN_EXTRACT,
      expectedOutputColumns = StringUtils.UNKNOWN_EXTRACT,
      expectedWriteMode = StringUtils.UNKNOWN_EXTRACT,
      expectedTableName = "my_iceberg_table",
      expectedDatabaseName = "db",
      expectedPartitionCols = StringUtils.UNKNOWN_EXTRACT
    )
  }

  test("MergeRows — Iceberg MERGE INTO operation is recognized") {
    // Test that MergeRows is correctly identified for Iceberg MERGE INTO operations.
    // Note: Spark plan shows "MergeRows" (without Exec suffix) as nodeName.
    //
    // Real event log structure:
    //   nodeName: "MergeRows"
    //   simpleString: "MergeRowsExec[_c0#523, _c1#524, ...]"  <- This becomes node.desc
    //
    // The simpleString only contains output columns, NOT table metadata.
    // Table/format info is in physicalPlanDescription (Arguments section), not simpleString.
    val node = new SparkPlanGraphNode(
      id = 6,
      name = "MergeRows",
      // Real simpleString format: "MergeRowsExec[output_columns...]"
      desc = "MergeRowsExec[_c0#523, _c1#524, _c2#525, _c3#526L, _c4#527, _c5#528, " +
        "_c6#529, _c7#530, _c8#531, _c9#532, _c10#533, _c11#534, _c12#535, " +
        "_c13#536, _c14#537, _c15#538, _file#539]",
      Seq.empty
    )

    // Verify IcebergOps accepts MergeRows
    assert(IcebergOps.accepts(node.name),
      "IcebergOps should accept MergeRows")

    // MergeRows doesn't have metadata in simpleString - metadata extraction returns None.
    // The IcebergWrite(table=..., format=...) info is only in physicalPlanDescription.
    val metadata: Option[WriteOperationMetadataTrait] =
      IcebergOps.extractOpMeta(node, confProvider = None)
    assert(metadata.isEmpty,
      "MergeRows metadata should be None - no table/format info in simpleString")
  }

  test("ReplaceData — Iceberg copy-on-write MERGE INTO operation is recognized") {
    // Test that ReplaceData is correctly identified for Iceberg copy-on-write MERGE INTO operations.
    // ReplaceData is the write operator that follows MergeRows in CoW mode.
    //
    // Real event log structure:
    //   nodeName: "ReplaceData"
    //   simpleString: "ReplaceData"  <- Just the name, NO metadata!
    //
    // The IcebergWrite(table=..., format=PARQUET) appears in physicalPlanDescription:
    //   (12) ReplaceData
    //   Arguments: IcebergWrite(table=spark_catalog.default.table, format=PARQUET)
    val node = new SparkPlanGraphNode(
      id = 7,
      name = "ReplaceData",
      // Real simpleString is just the operator name - no metadata
      desc = "ReplaceData",
      Seq.empty
    )

    // Verify IcebergOps accepts ReplaceData
    assert(IcebergOps.accepts(node.name),
      "IcebergOps should accept ReplaceData")

    // ReplaceData doesn't have metadata in simpleString - it's in physicalPlanDescription
    val metadata: Option[WriteOperationMetadataTrait] =
      IcebergOps.extractOpMeta(node, confProvider = None)
    assert(metadata.isEmpty,
      "ReplaceData metadata should be None - no table/format info in simpleString")
  }

  test("WriteDelta — Iceberg merge-on-read MERGE INTO operation is recognized") {
    // Test that WriteDelta is correctly identified for Iceberg merge-on-read MERGE INTO operations.
    // WriteDelta writes "delete files" instead of rewriting data files (MoR strategy).
    //
    // Event log structure:
    //   nodeName: "WriteDelta"
    //   simpleString: "WriteDelta"
    //
    // Merge-on-Read DAG:
    //   WriteDelta (14)          <- MoR write operator
    //   +- Exchange (13)
    //      +- MergeRows (12)
    //         +- SortMergeJoin RightOuter (10)
    //
    // The SparkPositionDeltaWrite info appears in physicalPlanDescription:
    //   (14) WriteDelta
    //   Arguments: org.apache.iceberg.spark.source.SparkPositionDeltaWrite@...
    val node = new SparkPlanGraphNode(
      id = 8,
      name = "WriteDelta",
      // Real simpleString is just the operator name - no metadata
      desc = "WriteDelta",
      Seq.empty
    )

    // Verify IcebergOps accepts WriteDelta
    assert(IcebergOps.accepts(node.name),
      "IcebergOps should accept WriteDelta")

    // WriteDelta doesn't have metadata in simpleString - extraction needs physicalPlanDescription
    val metadataWithoutPhysPlan: Option[WriteOperationMetadataTrait] =
      IcebergOps.extractOpMeta(node, confProvider = None, physicalPlanDescription = None)
    assert(metadataWithoutPhysPlan.isEmpty,
      "WriteDelta metadata should be None without physicalPlanDescription")
  }

  test("ReplaceData — metadata extracted from physicalPlanDescription") {
    // Test that ReplaceData metadata (table, format) can be extracted from physicalPlanDescription.
    // Node graph id=0 deliberately differs from Spark's internal id=12 in physPlanDesc to test
    // name-based matching works correctly.
    val node = new SparkPlanGraphNode(
      id = 0,
      name = "ReplaceData",
      desc = "ReplaceData",  // simpleString has no metadata
      Seq.empty
    )

    // Real physicalPlanDescription format from Spark UI
    // Note: Spark uses its own id (12), which differs from node graph id (0)
    val physicalPlanDescription =
      """(12) ReplaceData
        |Input [16]: [_c0#523, _c1#524, _c2#525, _c3#526L, _c4#527, _c5#528]
        |Arguments: IcebergWrite(table=spark_catalog.default.my_target_table, format=PARQUET)
        |""".stripMargin

    val metadata = IcebergOps.extractOpMeta(
      node, confProvider = None, physicalPlanDescription = Some(physicalPlanDescription))

    assert(metadata.isDefined, "ReplaceData metadata should be extracted from physicalPlanDescription")
    val meta = metadata.get
    assert(meta.execName() == "ReplaceData", s"execName should be ReplaceData, got ${meta.execName()}")
    assert(meta.dataFormat() == "IcebergParquet",
      s"dataFormat should be IcebergParquet, got ${meta.dataFormat()}")
    assert(meta.table() == "my_target_table",
      s"table should be my_target_table, got ${meta.table()}")
    assert(meta.dataBase() == "default",
      s"dataBase should be default, got ${meta.dataBase()}")
  }

  test("WriteDelta — metadata extracted from physicalPlanDescription") {
    // Test that WriteDelta is recognized as Iceberg position delete format.
    // Node graph id=1 deliberately differs from Spark's internal id=16 in physPlanDesc to test
    // name-based matching works correctly.
    val node = new SparkPlanGraphNode(
      id = 1,
      name = "WriteDelta",
      desc = "WriteDelta",  // simpleString has no metadata
      Seq.empty
    )

    // Real physicalPlanDescription format from Spark UI
    // Note: Spark uses its own id (16), which differs from node graph id (1)
    val physicalPlanDescription =
      """(16) WriteDelta
        |Input [21]: [__row_operation#10023, _c0#10024, _c1#10025]
        |Arguments: org.apache.iceberg.spark.source.SparkPositionDeltaWrite@5c5feaaa
        |""".stripMargin

    val metadata = IcebergOps.extractOpMeta(
      node, confProvider = None, physicalPlanDescription = Some(physicalPlanDescription))

    assert(metadata.isDefined, "WriteDelta metadata should be extracted from physicalPlanDescription")
    val meta = metadata.get
    assert(meta.execName() == "WriteDelta", s"execName should be WriteDelta, got ${meta.execName()}")
    // WriteDelta is for merge-on-read position deletes
    assert(meta.dataFormat() == "IcebergPositionDelete",
      s"dataFormat should be IcebergPositionDelete, got ${meta.dataFormat()}")
  }

  test("MergeRows — NOT included in write operations") {
    // MergeRows is an intermediate exec (OpType: Exec), NOT a write operation
    val node = new SparkPlanGraphNode(
      id = 10,
      name = "MergeRows",
      desc = "MergeRowsExec[_c0#523, _c1#524]",
      Seq.empty
    )

    val physicalPlanDescription =
      """(10) MergeRows
        |Input [36]: [_c0#478, _c1#479]
        |Arguments: isnotnull(__row_from_source#522), [keep(true, _c0#494)]
        |""".stripMargin

    val metadata = IcebergOps.extractOpMeta(
      node, confProvider = None, physicalPlanDescription = Some(physicalPlanDescription))

    // MergeRows should NOT be included in write operations
    assert(metadata.isEmpty,
      "MergeRows should NOT have write metadata")
  }
  // scalastyle:on line.size.limit
}
