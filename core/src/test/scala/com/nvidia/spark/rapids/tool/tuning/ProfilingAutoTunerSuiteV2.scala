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

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.{DynamicAllocationInfo, GpuTypes, NodeInstanceMapKey, PlatformFactory, PlatformInstanceTypes, PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.profiling.{Profiler, PySparkMemoryEvidence,
  RecommendedCommentResult, ShuffleStageInputAnalysis}
import com.nvidia.spark.rapids.tool.tuning.config.{ConfTypeEnum, TuningConfigEntry,
  TuningConfiguration, TuningEntryDefinition}
import org.scalatest.matchers.should.Matchers._
import org.scalatest.prop.TableDrivenPropertyChecks._

import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.{SparkSession, TrampolineUtil}
import org.apache.spark.sql.rapids.tool.annotation.Since
import org.apache.spark.sql.rapids.tool.util.StringUtils

/**
 * Test suite for the Profiling AutoTuner that uses the new target cluster properties format.
 *
 * This test suite introduces a cleaner way to specify target cluster configurations by explicitly
 * separating:
 * - Target cluster shape (cores, memory, GPU count/type)
 * - Target Spark properties (enforced configurations)
 *
 * This is in contrast to the legacy format in [[ProfilingAutoTunerSuite]] which overloaded the
 * same format for both source and target cluster properties.
 */
@Since("25.04.2")
class ProfilingAutoTunerSuiteV2 extends ProfilingAutoTunerSuiteBase {

  private val cacheSerializerProperty = AutoTuner.CACHE_SERIALIZER_PROPERTY
  private val gpuCacheSerializer = "com.nvidia.spark.ParquetCachedBatchSerializer"
  private val sparkCacheSerializer =
    "org.apache.spark.sql.execution.columnar.DefaultCachedBatchSerializer"

  private def runCacheSerializerAutoTuner(
      hasSqlCache: Boolean,
      sourceSerializer: Option[String] = None,
      tuningConfigs: Option[TuningConfiguration] = None,
      targetClusterInfo: Option[TargetClusterProps] = None,
      sourceProperties: Map[String, String] = Map.empty,
      showOnlyUpdatedProps: Boolean = true,
      sparkVersion: String = testSparkVersion):
      (Seq[TuningEntryTrait], Seq[RecommendedCommentResult]) = {
    val logEventsProps = mutable.LinkedHashMap(defaultDataprocProps.toSeq: _*)
    logEventsProps ++= sourceProperties
    sourceSerializer.foreach(logEventsProps.put(cacheSerializerProperty, _))
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), logEventsProps,
      Some(sparkVersion), hasSqlCache = hasSqlCache)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, targetClusterInfo)
    configureEventLogClusterInfoForTest(platform, sparkProperties = logEventsProps.toMap)
    buildAutoTunerForTests(infoProvider, platform,
      userProvidedTuningConfigs = tuningConfigs)
      .getRecommendedProperties(showOnlyUpdatedProps = showOnlyUpdatedProps)
  }

  private def cacheSerializerValue(properties: Seq[TuningEntryTrait]): Option[String] = {
    properties.find(_.name == cacheSerializerProperty).map(_.getTuneValue())
  }

  lazy val sparkSession: SparkSession = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName("Rapids Spark Profiling Tool Unit Tests")
      .getOrCreate()
  }

  test("cache serializer is recommended when SQL cache evidence was observed") {
    val (properties, _) = runCacheSerializerAutoTuner(hasSqlCache = true)

    cacheSerializerValue(properties) shouldBe Some(gpuCacheSerializer)
  }

  test("cache serializer is not recommended without a cache plan node") {
    val (properties, _) = runCacheSerializerAutoTuner(hasSqlCache = false)

    cacheSerializerValue(properties) shouldBe None
  }

  test("Spark default cache serializer is replaced") {
    val (properties, _) = runCacheSerializerAutoTuner(
      hasSqlCache = true, Some(sparkCacheSerializer))

    cacheSerializerValue(properties) shouldBe Some(gpuCacheSerializer)
  }

  test("configured cache serializer is preserved") {
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true, Some(gpuCacheSerializer))

    cacheSerializerValue(properties) shouldBe None
    comments.map(_.comment).exists(_.contains("custom cache serializer")) shouldBe false
  }

  test("custom cache serializer is preserved with an advisory") {
    val customSerializer = "example.ExistingCachedBatchSerializer"
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true, Some(customSerializer))
    val commentText = comments.map(_.comment).mkString("\n")

    cacheSerializerValue(properties) shouldBe None
    commentText should include("custom cache serializer")
    commentText should include("preserving")
    commentText should include("GPU InMemoryTableScan")
  }

  test("cache serializer recommendation uses the tuning configuration") {
    val customSerializer = "example.CustomCachedBatchSerializer"
    val tuningConfigs = ToolTestUtils.buildTuningConfigs(default = List(
      TuningConfigEntry(
        name = AutoTuner.CACHE_SERIALIZER_CONFIG,
        default = customSerializer,
        usedBy = cacheSerializerProperty)))
    val (properties, _) = runCacheSerializerAutoTuner(
      hasSqlCache = true, tuningConfigs = Some(tuningConfigs))

    cacheSerializerValue(properties) shouldBe Some(customSerializer)
  }

  test("preserved Spark cache serializer retains the compatibility advisory") {
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      preserveSparkProperties = List(cacheSerializerProperty))
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true,
      sourceSerializer = Some(sparkCacheSerializer),
      targetClusterInfo = Some(targetClusterInfo))

    cacheSerializerValue(properties) shouldBe Some(sparkCacheSerializer)
    comments.map(_.comment).mkString("\n") should include("target cluster configuration")
  }

  test("missing preserved cache serializer is recommended") {
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      preserveSparkProperties = List(cacheSerializerProperty))
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true,
      targetClusterInfo = Some(targetClusterInfo))

    cacheSerializerValue(properties) shouldBe Some(gpuCacheSerializer)
    comments.map(_.comment).mkString("\n") should not include "no source value"
  }

  test("enforced Spark cache serializer retains the compatibility advisory") {
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = Map(cacheSerializerProperty -> sparkCacheSerializer))
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true,
      targetClusterInfo = Some(targetClusterInfo),
      showOnlyUpdatedProps = false)

    cacheSerializerValue(properties) shouldBe Some(sparkCacheSerializer)
    comments.map(_.comment).mkString("\n") should include("target cluster configuration")
  }

  test("disabled GPU cache scan suppresses serializer recommendation with an advisory") {
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true,
      sourceProperties = Map(AutoTuner.IN_MEMORY_TABLE_SCAN_PROPERTY -> "false"))
    val commentText = comments.map(_.comment).mkString("\n")

    cacheSerializerValue(properties) shouldBe None
    commentText should include(AutoTuner.IN_MEMORY_TABLE_SCAN_PROPERTY)
    commentText should include("is not recommended")
    commentText should include("GPU cache scans remain disabled")
  }

  test("Spark 3.5.1 cache serializer recommendation includes the AQE fallback advisory") {
    val (properties, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true,
      sparkVersion = "3.5.1")
    val commentText = comments.map(_.comment).mkString("\n")

    cacheSerializerValue(properties) shouldBe Some(gpuCacheSerializer)
    commentText should include("disable GPU InMemoryTableScan under AQE")
  }

  test("Spark 3.5.2 cache serializer recommendation has no AQE fallback advisory") {
    val (_, comments) = runCacheSerializerAutoTuner(
      hasSqlCache = true,
      sparkVersion = "3.5.2")

    comments.map(_.comment).mkString("\n") should not include
      "disable GPU InMemoryTableScan under AQE"
  }

  // Test that the properties from the custom target cluster props will be enforced.
  test("AutoTuner enforces properties from custom target cluster props") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        // Below properties should be overridden by the enforced properties
        "spark.sql.shuffle.partitions" -> "200",
        "spark.sql.files.maxPartitionBytes" -> "1g",
        "spark.task.resource.gpu.amount" -> "0.001",
        "spark.rapids.sql.concurrentGpuTasks" -> "4"
      )
    // 3. Define enforced properties for the target cluster
    val enforcedSparkProperties = Map(
      "spark.sql.shuffle.partitions" -> "400",
      "spark.sql.files.maxPartitionBytes" -> "101m",
      "spark.task.resource.gpu.amount" -> "0.25",
      "spark.rapids.sql.concurrentGpuTasks" -> "2"
    )
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=2
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=101m
          |--conf spark.sql.shuffle.partitions=400
          |--conf spark.task.resource.gpu.amount=0.25
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- ${getEnforcedPropertyComment("spark.rapids.sql.concurrentGpuTasks")}
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${getEnforcedPropertyComment("spark.sql.files.maxPartitionBytes")}
          |- ${getEnforcedPropertyComment("spark.sql.shuffle.partitions")}
          |- ${getEnforcedPropertyComment("spark.task.resource.gpu.amount")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test that the executor memory and memory overhead properties from the custom target cluster
  // props lead to AutoTuner warning about insufficient memory.
  test("AutoTuner warns about insufficient memory with executor heap and" +
    " memory overhead override") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Define enforced properties for the target cluster
    // Note: These values should cause insufficient memory warning
    val enforcedSparkProperties = Map(
      "spark.executor.memory" -> "40g",
      "spark.executor.memoryOverhead" -> "30g"
    )
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=[FILL_IN_VALUE]
          |--conf spark.executor.memoryOverhead=[FILL_IN_VALUE]
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- ${getEnforcedPropertyComment("spark.executor.memory")}
          |- ${getEnforcedPropertyComment("spark.executor.memoryOverhead")}
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${notEnoughMemCommentForKey("spark.executor.memory")}
          |- ${notEnoughMemCommentForKey("spark.executor.memoryOverhead")}
          |- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${notEnoughMemComment(89600)}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test that the pinned pool property from the custom target cluster
  // props lead to AutoTuner warning about insufficient memory.
  test("AutoTuner warns about insufficient memory with pinned pool override") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Define enforced properties for the target cluster
    val enforcedSparkProperties = Map(
      "spark.rapids.memory.pinnedPool.size" -> "30g", // Should cause insufficient memory warning
      "spark.sql.files.maxPartitionBytes" -> "101m"   // Should be enforced
    )
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )
    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=[FILL_IN_VALUE]
          |--conf spark.executor.memoryOverhead=[FILL_IN_VALUE]
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=101m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- ${getEnforcedPropertyComment("spark.rapids.memory.pinnedPool.size")}
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${getEnforcedPropertyComment("spark.sql.files.maxPartitionBytes")}
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${notEnoughMemCommentForKey("spark.executor.memory")}
          |- ${notEnoughMemCommentForKey("spark.executor.memoryOverhead")}
          |- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${notEnoughMemComment(126975)}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("Test Kryo Serializer does not add GPU registrator again if already present") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrator" ->
          "org.apache.SomeRegistrator,com.nvidia.spark.rapids.GpuKryoRegistrator",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.kryoserializer.buffer.max=512m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.kryoserializer.buffer.max' increasing the max buffer to prevent out-of-memory errors.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test that AutoTuner parses existing Kryo Registrator correctly
  // i.e. it removes duplicates, empty entries, and adds GpuKryoRegistrator
  test("Test AutoTuner parses existing Kryo Registrator correctly") {
    // mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrator" ->
          "org.apache.SomeRegistrator,, org.apache.OtherRegistrator,org.apache.SomeRegistrator",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )
    val autoTuner = buildDefaultDataprocAutoTuner(logEventsProps)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.kryo.registrator=org.apache.SomeRegistrator,org.apache.OtherRegistrator,com.nvidia.spark.rapids.GpuKryoRegistrator
          |--conf spark.kryoserializer.buffer.max=512m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.kryo.registrator' GpuKryoRegistrator must be appended to the existing value when using Kryo serialization.
          |- 'spark.kryoserializer.buffer.max' increasing the max buffer to prevent out-of-memory errors.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test uses target cluster properties with user-enforced Spark properties.
  // The platform is mocked as Kubernetes on OnPrem
  // to enable memory overhead calculation.
  // AutoTuner is expected to:
  // - Include the enforced Spark properties in the final configuration.
  test("Target cluster properties for OnPrem with enforced spark properties") {
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "1",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Define enforced properties for the target cluster
    val enforcedSparkProperties = Map(
      "spark.sql.shuffle.partitions" -> "101",
      "spark.sql.files.maxPartitionBytes" -> "101m",
      "spark.task.resource.gpu.amount" -> "0.25"
    )
    // sparkProperties:
    //   enforced:
    //    spark.sql.shuffle.partitions: 101
    //    spark.sql.files.maxPartitionBytes: 101m
    //    spark.task.resource.gpu.amount: 0.25
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = enforcedSparkProperties
    )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "50g")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 1,
      gpuCount = 2, // default for OnPrem
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.memoryOverhead=34918m
          |--conf spark.executor.resource.gpu.vendor=nvidia.com
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=101
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=101m
          |--conf spark.sql.shuffle.partitions=101
          |--conf spark.task.resource.gpu.amount=0.25
          |
          |Comments:
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.executor.resource.gpu.vendor' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${getEnforcedPropertyComment("spark.sql.files.maxPartitionBytes")}
          |- ${getEnforcedPropertyComment("spark.sql.shuffle.partitions")}
          |- ${getEnforcedPropertyComment("spark.task.resource.gpu.amount")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $missingGpuDiscoveryScriptComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test uses target cluster properties with a worker node having 16 cores, 64g memory,
  // 1 L4 GPU, and user-enforced Spark properties. The platform is mocked as Kubernetes on OnPrem
  // to enable memory overhead calculation.
  // AutoTuner is expected to:
  // - Recommend 32g executor memory,
  // - Calculate overhead using the max pinned pool size (4g),
  // - Include the enforced Spark properties in the final configuration.
  test("Target cluster properties for OnPrem with workerInfo and enforced spark properties") {
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Define enforced properties for the target cluster
    val enforcedSparkProperties = Map(
      "spark.sql.shuffle.partitions" -> "400",
      "spark.sql.files.maxPartitionBytes" -> "101m",
      "spark.task.resource.gpu.amount" -> "0.25",
      "spark.rapids.sql.concurrentGpuTasks" -> "1"  // For L4, default recommendation would be 3
    )
    // workerInfo:
    //   cpuCores: 16
    //   memoryGB: 64
    //   gpu:
    //     count: 1
    //     name: l4
    // sparkProperties:
    //   enforced:
    //    spark.sql.shuffle.partitions: 400
    //    spark.sql.files.maxPartitionBytes: 101m
    //    spark.task.resource.gpu.amount: 0.25
    //    spark.rapids.sql.concurrentGpuTasks: 2
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16), memoryGB = Some(64),
      gpuCount = Some(1), gpuDevice = Some(GpuTypes.L4),
      enforcedSparkProperties = enforcedSparkProperties
    )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "14000MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8, // from eventLog
      numWorkers = 2,
      gpuCount = 1, // target cluster has 1 L4
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=32g
          |--conf spark.executor.resource.gpu.vendor=nvidia.com
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=1
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=101m
          |--conf spark.sql.shuffle.partitions=400
          |--conf spark.task.resource.gpu.amount=0.25
          |
          |Comments:
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.executor.resource.gpu.vendor' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- ${getEnforcedPropertyComment("spark.rapids.sql.concurrentGpuTasks")}
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- ${getEnforcedPropertyComment("spark.sql.files.maxPartitionBytes")}
          |- ${getEnforcedPropertyComment("spark.sql.shuffle.partitions")}
          |- ${getEnforcedPropertyComment("spark.task.resource.gpu.amount")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $missingGpuDiscoveryScriptComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test uses custom target cluster properties with 40g worker memory and 2 GPUs.
  // Now, each executor can use up to 20g (including memory and overhead).
  // The user enforces spark.executor.memory to 18g. This leaves insufficient room for overhead.
  // AutoTuner is expected to warn about the insufficient executor memory configuration.
  test("Target cluster properties for OnPrem with total executor memory " +
    "exceeding available worker memory") {
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Define enforced properties for the target cluster
    val enforcedSparkProperties = Map(
      "spark.executor.cores" -> "8",
      "spark.executor.memory" -> "18g",   // Requesting more memory than available in the node
      "spark.sql.shuffle.partitions" -> "400"
    )
    // workerInfo:
    //   cpuCores: 16
    //   memoryGB: 40
    //   gpu:
    //     count: 2
    //     name: l4
    // sparkProperties:
    //   enforced:
    //    spark.executor.cores: 8
    //    spark.executor.memory: 18g
    //    spark.sql.shuffle.partitions: 400
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16), memoryGB = Some(40),
      gpuCount = Some(2), gpuDevice = Some(GpuTypes.L4),
      enforcedSparkProperties = enforcedSparkProperties
    )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "14000MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 16, // from eventLog
      numWorkers = 2,
      gpuCount = 2, // target cluster has 2 L4s
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=8
          |--conf spark.executor.memory=[FILL_IN_VALUE]
          |--conf spark.executor.memoryOverhead=[FILL_IN_VALUE]
          |--conf spark.executor.resource.gpu.vendor=nvidia.com
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=[FILL_IN_VALUE]
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=400
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.sql.shuffle.partitions=400
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- ${getEnforcedPropertyComment("spark.executor.cores")}
          |- ${getEnforcedPropertyComment("spark.executor.memory")}
          |- 'spark.executor.resource.gpu.vendor' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- ${getEnforcedPropertyComment("spark.sql.shuffle.partitions")}
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${notEnoughMemCommentForKey("spark.executor.memory")}
          |- ${notEnoughMemCommentForKey("spark.executor.memoryOverhead")}
          |- ${notEnoughMemCommentForKey("spark.rapids.memory.pinnedPool.size")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- ${notEnoughMemComment(24371)}
          |- $missingGpuDiscoveryScriptComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test verifies that an error is thrown when the target cluster YAML file
  // contains both instance type (for CSP) and resource properties (for OnPrem).
  test("Should fail when target cluster contains both CSP instanceType and OnPrem resources") {
    TrampolineUtil.withTempDir { tempDir =>
      // workerInfo:
      //   instanceType: g2-standard-8
      //   cpuCores: 16
      //   memoryGB: 64
      //   gpu:
      //     count: 1
      //     name: l4
      assertThrows[IllegalArgumentException] {
        ToolTestUtils.createTargetClusterInfoFile(
          tempDir.getAbsolutePath,
          workerNodeInstanceType = Some("g2-standard-8"),
          cpuCores = Some(16), memoryGB = Some(64),
          gpuCount = Some(1), gpuDevice = Some(GpuTypes.L4))
      }
    }
  }

  // This test verifies that an error is thrown when the target cluster YAML file
  // contains resource properties (for OnPrem) except GPU.
  test("Should fail when target cluster contains OnPrem resources except GPU") {
    TrampolineUtil.withTempDir { tempDir =>
      // workerInfo:
      //   cpuCores: 16
      //   memoryGB: 64
      assertThrows[IllegalArgumentException] {
        ToolTestUtils.createTargetClusterInfoFile(
          tempDir.getAbsolutePath,
          cpuCores = Some(16), memoryGB = Some(64))
      }
    }
  }

  // This test verifies that an error is thrown when the target cluster YAML file
  // contains both worker info and driver info for OnPrem
  test("Should fail when target cluster contains both worker info and driver info for OnPrem") {
    TrampolineUtil.withTempDir { tempDir =>
      // driverInfo:
      //   instanceType: foobar
      // workerInfo:
      //   cpuCores: 16
      //   memoryGB: 64
      //   gpu:
      //     count: 1
      //     name: l4
      assertThrows[IllegalArgumentException] {
        ToolTestUtils.createTargetClusterInfoFile(
          tempDir.getAbsolutePath,
          driverNodeInstanceType = Some("foobar"),
          cpuCores = Some(16), memoryGB = Some(64),
          gpuCount = Some(1), gpuDevice = Some(GpuTypes.L4))
      }
    }
  }

  // This test uses target cluster properties with a worker node having 16 cores, 64g memory,
  // and 1 L20 GPU. The platform is mocked as Kubernetes on OnPrem
  // to enable memory overhead calculation.
  // AutoTuner is expected to:
  // - Recommend `spark.rapids.sql.concurrentGpuTasks` to a value of 4 since L20
  //   has high memory.
  test("Target cluster properties for OnPrem with worker having L20 GPU") {
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // workerInfo:
    //   cpuCores: 16
    //   memoryGB: 64
    //   gpu:
    //     count: 1
    //     name: l20
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16), memoryGB = Some(64),
      gpuCount = Some(1), gpuDevice = Some(GpuTypes.L20)
    )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    val sparkPropsWithMemory = logEventsProps + ("spark.executor.memory" -> "14000MiB")
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8, // from eventLog
      numWorkers = 2,
      gpuCount = 1, // target cluster has 1 L20
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=32g
          |--conf spark.executor.resource.gpu.vendor=nvidia.com
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=4
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.executor.resource.gpu.vendor' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |- $missingGpuDiscoveryScriptComment
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test that the alias feature works correctly for mapping non-standard Spark properties
  // to standard ones, using the specific example from the issue:
  // spark.sql.adaptive.shuffle.maxNumPostShufflePartitions ->
  //   spark.sql.adaptive.coalescePartitions.initialPartitionNum
  test("AutoTuner should handle aliased properties from tuningDefinitions") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("n1-standard-16", Option(1))
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)

    // 2. Mock the properties loaded from eventLog with the aliased property
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        // alias property
        "spark.sql.adaptive.maxNumPostShufflePartitions" -> "100"
      )

    // 3. Create user-defined tuningDefinitions for the target cluster
    val userTuningDefinitions = createMaxNumPostShufflePartitionsTuningDefinition()

    // 4. Define enforced properties for the target cluster (no alias needed in target-cluster yaml)
    val enforcedSparkProperties = Map(
      "spark.task.resource.gpu.amount" -> "0.25",
      "spark.rapids.sql.concurrentGpuTasks" -> "2"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("n1-standard-16"),
      gpuCount = Some(1),
      enforcedSparkProperties = enforcedSparkProperties,
      tuningDefinitions = userTuningDefinitions
    )

    val infoProvider = getMockInfoProvider(
      maxInput = 100000.0,
      spilledMetrics = Seq(100000),
      jvmGCFractions = Seq(0.004),
      propsFromLog = logEventsProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = 60000.0,  // > 35000 (AQE_INPUT_SIZE_BYTES_THRESHOLD)
      meanShuffleRead = 70000.0  // > 50000 (AQE_SHUFFLE_READ_BYTES_THRESHOLD)
    )
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = 1,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=16g
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=4506m
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=2
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=32m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
          |--conf spark.sql.adaptive.maxNumPostShufflePartitions=800
          |--conf spark.sql.files.maxPartitionBytes=4g
          |--conf spark.sql.shuffle.partitions=800
          |--conf spark.task.resource.gpu.amount=0.25
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- ${getEnforcedPropertyComment("spark.rapids.sql.concurrentGpuTasks")}
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- ${getEnforcedPropertyComment("spark.task.resource.gpu.amount")}
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  /**
   * Helper method to create tuning definition for testing
   * spark.sql.adaptive.maxNumPostShufflePartitions alias
   */
  private def createMaxNumPostShufflePartitionsTuningDefinition():
    java.util.List[TuningEntryDefinition] = {
    import scala.jdk.CollectionConverters._
    List(TuningEntryDefinition(
      label = "spark.sql.adaptive.maxNumPostShufflePartitions",
      description = "Custom tuning definition for testing alias feature",
      confType = ConfTypeEnum.Int)).asJava
  }

  // This test validates that user-provided tuning configurations are honored by AutoTuner.
  // AutoTuner is expected to:
  // - Recommend `spark.executor.memory` to a value of 0.9g/core * 16cores ~ 14736m
  // - Recommend `spark.rapids.sql.concurrentGpuTasks` to a value of 1
  test("AutoTuner honours user provided tuning configurations") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Mock the user-provided tuning configurations. Equivalent YAML snippet:
    // tuningConfigs:
    //   default:
    //   - name: HEAP_PER_CORE
    //     default: 0.9g
    //   - name: CONC_GPU_TASKS
    //     max: 1
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "HEAP_PER_CORE", default = "0.9g"),
      TuningConfigEntry(name = "CONC_GPU_TASKS", max = "1")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner =
      buildAutoTunerForTests(
        infoProvider,
        platform,
        Some(Kubernetes),
        Some(userProvidedTuningConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=14736m
          |--conf spark.executor.memoryOverhead=37692m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=1
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // This test validates that AutoTuner throws IllegalArgumentException when user provides
  // tuning configurations with typos in name.
  test("AutoTuner should throw IllegalArgumentException for typo in tuning config name") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1"
      )
    // 3. Mock the user-provided tuning configurations with typo. Equivalent YAML snippet:
    // tuningConfigs:
    //   default:
    //   - name: BATCH_SIZE_BTYES  # Typo: should be BATCH_SIZE_BYTES
    //     default: 512m
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "BATCH_SIZE_BTYES", default = "512m")  // Typo in name
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )

    // AutoTuner should throw IllegalArgumentException due to invalid tuning config name
    assertThrows[IllegalArgumentException] {
      val autoTuner =
        buildAutoTunerForTests(infoProvider, platform, None, Some(userProvidedTuningConfigs))
      autoTuner.getRecommendedProperties()
    }
  }

  // This test validates that AutoTuner honours existing values of
  // `spark.executor.resource.gpu.amount` when it is already set.
  test("AutoTuner honours existing values of spark.executor.resource.gpu.amount") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("g2-standard-16")
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)
    // 2. Mock the properties loaded from eventLog
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "2"
      )
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = gpuInstance.numGpus,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=512m
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("AutoTuner does not reduce initialPartitionNum when ColumnarExchange ratio is smaller") {
    // Test case: max columnar exchange data size = 1000GB, batch size is default value 2gb
    // original initialPartitionNum = 2560
    // ratio = ceil(1000GB / 2GB) = 501, which is < 2560
    // Expected: no adjustment since ColumnarExchange ratio should only increase partitions

    val maxColumnarExchangeDataSizeBytes = Some(1000L * 1024 * 1024 * 1024) // 1000GB in bytes
    val originalInitialPartitionNum = "2560"

    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum" -> originalInitialPartitionNum,
        "spark.sql.adaptive.enabled" -> "true",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes" -> "64m",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1"
      )

    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9, // Large input to trigger AQE recommendations
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = logEventsProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = 1.0E9, // Large mean input
      meanShuffleRead = 1.0E9, // Large mean shuffle read
      maxColumnarExchangeDataSizeBytes = maxColumnarExchangeDataSizeBytes
    )

    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    // Configure cluster info
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 16,
      numWorkers = 1,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
          |--conf spark.sql.files.maxPartitionBytes=137m
          |--conf spark.sql.shuffle.partitions=2560
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- RAPIDS Accelerator for Apache Spark plugin jar is missing
          |  from the classpath entries.
          |  If the Spark RAPIDS jar is being bundled with your
          |  Spark distribution, this step is not needed.
          |- The RAPIDS Shuffle Manager requires spark.driver.extraClassPath
          |  and spark.executor.extraClassPath settings to include the
          |  path to the Spark RAPIDS plugin jar.
          |  If the Spark RAPIDS jar is being bundled with your Spark
          |  distribution, this step is not needed.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("AutoTuner increases initialPartitionNum when ColumnarExchange data size ratio " +
    "is larger") {
    // Test case: max columnar exchange data size = 6000GB,
    // original initialPartitionNum = 2560
    // Expected: ratio = ceil(6000GB / ~2GB) = 3001, which is > 2560
    // So recommended initialPartitionNum = max(2560, 3001) = 3001 (increased)

    val maxColumnarExchangeDataSizeBytes = Some(6000L * 1024 * 1024 * 1024) // 6000GB in bytes
    val originalInitialPartitionNum = "2560"

    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum" -> originalInitialPartitionNum,
        "spark.sql.adaptive.enabled" -> "true",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes" -> "64m",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1"
      )

    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9,
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = logEventsProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = 1.0E9,
      meanShuffleRead = 1.0E9,
      maxColumnarExchangeDataSizeBytes = maxColumnarExchangeDataSizeBytes
    )

    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 16,
      numWorkers = 1,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=3001
          |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
          |--conf spark.sql.files.maxPartitionBytes=137m
          |--conf spark.sql.shuffle.partitions=3001
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' adjusted from 2560 to 3001 based on ColumnarExchange data size (6442450944000 bytes) and GPU batch size (2147483647 bytes)
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- RAPIDS Accelerator for Apache Spark plugin jar is missing
          |  from the classpath entries.
          |  If the Spark RAPIDS jar is being bundled with your
          |  Spark distribution, this step is not needed.
          |- The RAPIDS Shuffle Manager requires spark.driver.extraClassPath
          |  and spark.executor.extraClassPath settings to include the
          |  path to the Spark RAPIDS plugin jar.
          |  If the Spark RAPIDS jar is being bundled with your Spark
          |  distribution, this step is not needed.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  test("AutoTuner handles case when no ColumnarExchange data size is available") {
    // Test case: no ColumnarExchange data size available,
    // original initialPartitionNum = 2560
    // Expected: no adjustment should be made, original value should be preserved

    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "1",
        "spark.executor.memory" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.sql.adaptive.coalescePartitions.initialPartitionNum" -> "2560",
        "spark.sql.adaptive.enabled" -> "true",
        "spark.sql.adaptive.coalescePartitions.minPartitionSize" -> "4m",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes" -> "64m",
        "spark.sql.shuffle.partitions" -> "200",
        "spark.task.resource.gpu.amount" -> "0.25",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.concurrentGpuTasks" -> "1"
      )

    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9,
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = logEventsProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = 1.0E9,
      meanShuffleRead = 1.0E9,
      maxColumnarExchangeDataSizeBytes = None // No ColumnarExchange data size
    )

    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 16,
      numWorkers = 1,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=19660m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=6g
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
          |--conf spark.sql.files.maxPartitionBytes=137m
          |--conf spark.sql.shuffle.partitions=2560
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- RAPIDS Accelerator for Apache Spark plugin jar is missing
          |  from the classpath entries.
          |  If the Spark RAPIDS jar is being bundled with your
          |  Spark distribution, this step is not needed.
          |- The RAPIDS Shuffle Manager requires spark.driver.extraClassPath
          |  and spark.executor.extraClassPath settings to include the
          |  path to the Spark RAPIDS plugin jar.
          |  If the Spark RAPIDS jar is being bundled with your Spark
          |  distribution, this step is not needed.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Test that the alias feature works correctly for initialPartitionNum properties
  // using spark.sql.adaptive.maxNumPostShufflePartitions as alias
  test("AutoTuner should handle aliased initialPartitionNum properties from tuningDefinitions") {
    // 1. Mock source cluster info for dataproc
    val instanceMapKey = NodeInstanceMapKey("n1-standard-16", Option(1))
    val gpuInstance = PlatformInstanceTypes.DATAPROC_BY_INSTANCE_NAME(instanceMapKey)

    // 2. Mock the properties loaded from eventLog with the aliased property
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.rapids.memory.pinnedPool.size" -> "5g",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.executor.resource.gpu.amount" -> "1",
        // alias property for initialPartitionNum
        "spark.sql.adaptive.maxNumPostShufflePartitions" -> "2000"
      )

    // 3. Create user-defined tuningDefinitions for the target cluster
    val userTuningDefinitions = createMaxNumPostShufflePartitionsTuningDefinition()

    // 4. Define enforced properties for the target cluster
    val enforcedSparkProperties = Map(
      "spark.task.resource.gpu.amount" -> "0.25",
      "spark.rapids.sql.concurrentGpuTasks" -> "2"
    )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("n1-standard-16"),
      gpuCount = Some(1),
      enforcedSparkProperties = enforcedSparkProperties,
      tuningDefinitions = userTuningDefinitions
    )

    val infoProvider = getMockInfoProvider(
      maxInput = 100000.0,
      spilledMetrics = Seq(100000),
      jvmGCFractions = Seq(0.004),
      propsFromLog = logEventsProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = 60000.0,  // > 35000 (AQE_INPUT_SIZE_BYTES_THRESHOLD)
      meanShuffleRead = 70000.0,  // > 50000 (AQE_SHUFFLE_READ_BYTES_THRESHOLD)
      rapidsJars = Seq.empty,
      distinctLocationPct = 0.0,
      redundantReadSize = 0L,
      shuffleStagesWithPosSpilling = Set.empty,
      shuffleSkewStages = Set.empty,
      scanStagesWithGpuOom = Set.empty,
      gpuShuffleStagesWithContainerOom = Set.empty,
      maxColumnarExchangeDataSizeBytes = Some(1000L * 1024 * 1024 * 1024) // 1000GB
    )
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))

    val sparkPropsWithMemory =
      logEventsProps + ("spark.executor.memory" -> (gpuInstance.memoryMB.toString + "MiB"))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = gpuInstance.cores,
      numWorkers = 4,
      gpuCount = 1,
      sparkProperties = sparkPropsWithMemory.toMap
    )
    val autoTuner = buildAutoTunerForTests(infoProvider, platform)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)

    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dataproc.enhanced.execution.enabled=false
          |--conf spark.dataproc.enhanced.optimizer.enabled=false
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=16g
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=4506m
          |--conf spark.rapids.shuffle.multiThreaded.maxBytesInFlight=4g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=28
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=28
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=2
          |--conf spark.rapids.sql.format.parquet.multithreaded.combine.waitTime=1000
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=80
          |--conf spark.rapids.sql.reader.multithreaded.combine.sizeBytes=10m
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=32m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.adaptive.coalescePartitions.parallelismFirst=false
          |--conf spark.sql.files.maxPartitionBytes=4g
          |--conf spark.sql.shuffle.partitions=2000
          |--conf spark.task.resource.gpu.amount=0.25
          |
          |Comments:
          |- 'spark.dataproc.enhanced.execution.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.execution.enabled' was not set.
          |- 'spark.dataproc.enhanced.optimizer.enabled' should be disabled. WARN: Turning this property on might case the GPU accelerated Dataproc cluster to hang.
          |- 'spark.dataproc.enhanced.optimizer.enabled' was not set.
          |- 'spark.executor.memory' was not set.
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.maxBytesInFlight' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was user-enforced in the target cluster properties.
          |- 'spark.rapids.sql.format.parquet.multithreaded.combine.waitTime' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.rapids.sql.reader.multithreaded.combine.sizeBytes' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was user-enforced in the target cluster properties.
          |- RAPIDS Accelerator for Apache Spark plugin jar is missing
          |  from the classpath entries.
          |  If the Spark RAPIDS jar is being bundled with your
          |  Spark distribution, this step is not needed.
          |- The RAPIDS Shuffle Manager requires spark.driver.extraClassPath
          |  and spark.executor.extraClassPath settings to include the
          |  path to the Spark RAPIDS plugin jar.
          |  If the Spark RAPIDS jar is being bundled with your Spark
          |  distribution, this step is not needed.
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // Tests for NON_EXECUTOR_MEM_FRACTION tuning config

  // This test verifies that AutoTuner uses platform default when
  // NON_EXECUTOR_MEM_FRACTION is not specified (default is -1, meaning use platform default)
  test("AutoTuner uses platform default nonExecutorMemoryFraction") {
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("g2-standard-8")
    )
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))

    // Verify platform default is 0.2 (20% not available for executors) for Dataproc
    assert(platform.nonExecutorMemoryFraction == 0.2,
      s"Expected Dataproc default to be 0.2, got ${platform.nonExecutorMemoryFraction}")
  }

  // This test verifies that different platforms have different defaults
  test("Different platforms have different nonExecutorMemoryFraction defaults") {
    // Create platforms without target cluster info to test default values
    val dataprocPlatform = PlatformFactory.createInstance(PlatformNames.DATAPROC, None)
    val emrPlatform = PlatformFactory.createInstance(PlatformNames.EMR, None)

    assert(dataprocPlatform.nonExecutorMemoryFraction == 0.2,
      "Dataproc default should be 0.2 (20% not available for executors)")
    assert(emrPlatform.nonExecutorMemoryFraction == 0.3,
      "EMR default should be 0.3 (30% not available for executors)")
  }

  // Test custom NON_EXECUTOR_MEM_FRACTION value (0.0 = 100% memory utilization)
  test("AutoTuner accepts NON_EXECUTOR_MEM_FRACTION = 0.0 for full memory utilization") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "26742m",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    // Set NON_EXECUTOR_MEM_FRACTION to 0.0 (100% available for executors)
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "NON_EXECUTOR_MEM_FRACTION", default = "0.0")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    // Should use 0.0 (100% available) instead of Dataproc default 0.2
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn),
      Some(userProvidedTuningConfigs))
    val (_, _) = autoTuner.getRecommendedProperties()
  }

  // Test custom NON_EXECUTOR_MEM_FRACTION value (0.5 = 50% reserved)
  test("AutoTuner accepts custom NON_EXECUTOR_MEM_FRACTION = 0.5") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "26742m",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    // Set NON_EXECUTOR_MEM_FRACTION to 0.5 (50% reserved, 50% available)
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "NON_EXECUTOR_MEM_FRACTION", default = "0.5")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn),
      Some(userProvidedTuningConfigs))
    val (_, _) = autoTuner.getRecommendedProperties()
  }

  // Test invalid NON_EXECUTOR_MEM_FRACTION (>= 1.0) falls back to platform default
  test("AutoTuner falls back to platform default when NON_EXECUTOR_MEM_FRACTION >= 1.0") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "26742m",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    // Set invalid NON_EXECUTOR_MEM_FRACTION >= 1.0 (should fall back to platform default)
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "NON_EXECUTOR_MEM_FRACTION", default = "1.5")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    // Should fall back to Dataproc default (0.2) with a warning
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn),
      Some(userProvidedTuningConfigs))
    val (_, _) = autoTuner.getRecommendedProperties()
  }

  // Test that NON_EXECUTOR_MEM takes precedence over NON_EXECUTOR_MEM_FRACTION
  test("NON_EXECUTOR_MEM takes precedence over NON_EXECUTOR_MEM_FRACTION") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "16",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "26742m",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    // Set both absolute and fraction - absolute should take precedence
    val defaultTuningConfigsEntries = List(
      TuningConfigEntry(name = "NON_EXECUTOR_MEM", default = "5g"),
      TuningConfigEntry(name = "NON_EXECUTOR_MEM_FRACTION", default = "0.5")
    )
    val userProvidedTuningConfigs = ToolTestUtils.buildTuningConfigs(
      default = defaultTuningConfigsEntries)

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0), logEventsProps,
      Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 32,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    // NON_EXECUTOR_MEM (5g) should be used, not NON_EXECUTOR_MEM_FRACTION (0.5)
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn),
      Some(userProvidedTuningConfigs))
    val (_, _) = autoTuner.getRecommendedProperties()
  }

  // Test that user-enforced spark.executor.cores in target cluster is respected
  // when calculating recommended number of executors.
  test("AutoTuner respects user-enforced executor cores in cluster sizing") {
    // Source app with 11 cores, 148 executors (via maxExecutors)
    val logEventsProps: mutable.Map[String, String] = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "11",
      "spark.executor.memory" -> "21g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.dynamicAllocation.enabled" -> "true",
      "spark.dynamicAllocation.minExecutors" -> "1",
      "spark.dynamicAllocation.maxExecutors" -> "148",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
      "spark.rapids.sql.enabled" -> "true"
    )

    // Target cluster: 48 cores, 4 GPUs (default would be 12 cores/executor)
    // But user enforces 11 cores per executor
    val enforcedSparkProperties = Map(
      "spark.executor.cores" -> "11"
    )
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(48),  // 48 cores, 4 GPUs = default 12 cores per executor
      memoryGB = Some(192L),
      gpuCount = Some(4),
      gpuDevice = Some(GpuTypes.L4),
      enforcedSparkProperties = enforcedSparkProperties
    )

    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 11,
      numWorkers = 148,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
    val (properties, _) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, Seq.empty)

    // With user-enforced cores=11 and source total cores = 11 * 148 = 1628
    // Recommended executors = 1628 / 11 = 148
    // Without the fix (using default 12 cores), it would be 1628 / 12 = 135
    // This is the key assertion - executor.instances=148 proves cores=11 was used
    assert(autoTunerOutput.contains("spark.executor.instances=148"),
      s"Expected spark.executor.instances=148 with enforced cores=11, but got:\n$autoTunerOutput")
    // Note: spark.executor.cores may not appear in output if source already has same value
  }

  // On-prem profiling without target cluster uses SourceCoresPreservingStrategy,
  // which keeps source cores. Dynamic allocation runs with 1:1 ratio so values are unchanged.
  test("On-prem profiling without target cluster uses source cores for all recommendations") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "4",
        "spark.executor.memory" -> "47222m",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.dynamicAllocation.enabled" -> "true",
        "spark.dynamicAllocation.initialExecutors" -> "4",
        "spark.dynamicAllocation.minExecutors" -> "1",
        "spark.dynamicAllocation.maxExecutors" -> "10",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 4,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.memory=16g
          |--conf spark.executor.memoryOverhead=35560m
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // With a target cluster that has workerInfo, ConstantGpuCountStrategy re-slices cores.
  test("On-prem profiling with target cluster produces full recommendations") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "47222m",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.rapids.sql.enabled" -> "true",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16),
      memoryGB = Some(64),
      gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString)
    )

    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.executor.cores=16
          |--conf spark.executor.memory=32g
          |--conf spark.executor.memoryOverhead=32g
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=24
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=24
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=32
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.memoryOverhead' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  // With a target cluster, ConstantGpuCountStrategy re-slices cores (8 -> 16).
  // Core ratio = 8/16 = 0.5, so DA values are halved.
  // initialExecutors is boosted to max(floor(8*0.5), executor.instances=18) = 18.
  // maxExecutors = floor(18*0.5) = 9. Violation: initial(18) > max(9), capped to 9.
  test("dynamic allocation enforces invariant with target cluster") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.memory" -> "16g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.dynamicAllocation.enabled" -> "true",
        "spark.dynamicAllocation.initialExecutors" -> "8",
        "spark.dynamicAllocation.minExecutors" -> "4",
        "spark.dynamicAllocation.maxExecutors" -> "18",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.enabled" -> "true"
      )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16),
      memoryGB = Some(64),
      gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString)
    )

    val infoProvider = getMockInfoProvider(0, Seq(0),
      Seq(0.0), logEventsProps, Some(testSparkVersion))
    val platform =
      PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 18,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(
      infoProvider, platform, Some(Yarn))
    val (properties, comments) =
      autoTuner.getRecommendedProperties()
    assertDynamicAllocationRecommendations(properties, comments,
      DynamicAllocationInfo(
        enabled = true, max = "9", min = "2",
        initial = "9"))
  }

  // Issue #2053: when spark.executor.cores is in the target cluster `preserve` list,
  // the preserved source value (8) must be the sizing baseline, NOT re-sliced to the
  // platform default (16). Same scenario as "dynamic allocation enforces invariant with
  // target cluster" above, but with preserve. Core ratio = 8/8 = 1.0, so source DA is not
  // scaled; initialExecutors is still boosted to match executor.instances=18.
  // Expected: cores=8, instances=18, DA min=4 / initial=18 / max=18.
  test("preserved spark.executor.cores is respected by cluster sizing and dynamic allocation") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.memory" -> "16g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.dynamicAllocation.enabled" -> "true",
        "spark.dynamicAllocation.initialExecutors" -> "8",
        "spark.dynamicAllocation.minExecutors" -> "4",
        "spark.dynamicAllocation.maxExecutors" -> "18",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.enabled" -> "true"
      )

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16),
      memoryGB = Some(64),
      gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString),
      preserveSparkProperties = List("spark.executor.cores")
    )

    val infoProvider = getMockInfoProvider(0, Seq(0),
      Seq(0.0), logEventsProps, Some(testSparkVersion))
    val platform =
      PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 18,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
    // Use showOnlyUpdatedProps = false so that DA values left unchanged from source
    // (min/max, since the core ratio is now 1.0) are still present to assert on.
    val (properties, _) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val recommendedProps = properties.map(p => p.name -> p.getTuneValue()).toMap

    def assertProp(name: String, expected: String): Unit =
      assert(recommendedProps.get(name).contains(expected),
        s"$name: expected $expected, got ${recommendedProps.get(name)}")

    assertProp("spark.executor.cores", "8")
    assertProp("spark.executor.instances", "18")
    assertProp("spark.dynamicAllocation.minExecutors", "4")
    assertProp("spark.dynamicAllocation.initialExecutors", "18")
    assertProp("spark.dynamicAllocation.maxExecutors", "18")
  }

  // Issue #2053: when spark.executor.memory is preserved, the source heap value (8g)
  // must be used as the memory sizing BASELINE that drives dependent memory sizing
  // (e.g. executor.memoryOverhead), not merely echoed to the output. Asserting only the
  // final executor.memory value is insufficient: the final preserve override already
  // forces that value regardless of whether the baseline reads preserve. We therefore
  // assert that a dependent value (memoryOverhead) actually changes when the heap
  // baseline is the preserved 8g versus the larger computed default.
  test("preserved spark.executor.memory is used as the memory sizing baseline") {
    def runWith(preserve: List[String]): Map[String, String] = {
      val logEventsProps: mutable.Map[String, String] =
        mutable.LinkedHashMap[String, String](
          "spark.executor.cores" -> "8",
          "spark.executor.instances" -> "2",
          "spark.executor.memory" -> "8g",
          "spark.executor.resource.gpu.amount" -> "1",
          "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
          "spark.rapids.sql.enabled" -> "true"
        )
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(16),
        memoryGB = Some(64),
        gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4.toString),
        preserveSparkProperties = preserve
      )
      val infoProvider = getMockInfoProvider(0, Seq(0),
        Seq(0.0), logEventsProps, Some(testSparkVersion))
      val platform =
        PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
      configureEventLogClusterInfoForTest(
        platform,
        numCores = 8,
        numWorkers = 2,
        gpuCount = 1,
        sparkProperties = logEventsProps.toMap
      )
      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
      val (properties, _) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
      properties.map(p => p.name -> p.getTuneValue()).toMap
    }

    val withPreserve = runWith(List("spark.executor.memory"))
    val withoutPreserve = runWith(List.empty)

    // Sanity: the preserved heap is echoed to the output.
    val heapMB = withPreserve.get("spark.executor.memory")
      .map(v => StringUtils.convertToMB(v, Some(ByteUnit.BYTE)))
      .getOrElse(fail("spark.executor.memory not found in recommendations"))
    assert(heapMB == StringUtils.convertToMB("8g", Some(ByteUnit.BYTE)),
      s"Expected preserved heap 8g, got ${heapMB}MB")

    // The real proof: preserving the 8g heap as the baseline changes dependent memory
    // sizing. Without preserve, the baseline is the larger computed default, so the
    // recommended executor.memoryOverhead differs.
    val overheadWith = withPreserve.get("spark.executor.memoryOverhead")
    val overheadWithout = withoutPreserve.get("spark.executor.memoryOverhead")
    assert(overheadWith.isDefined && overheadWithout.isDefined,
      s"Expected executor.memoryOverhead in both runs, " +
        s"got with=$overheadWith without=$overheadWithout")
    assert(overheadWith != overheadWithout,
      s"Preserving spark.executor.memory should change dependent memory sizing; " +
        s"memoryOverhead with preserve=$overheadWith vs without=$overheadWithout")
  }

  // Issue #2053: host off-heap limit settings affect pinned-memory sizing on on-prem
  // clusters, so preserved source values must be used as calculation baselines.
  test("preserved host off-heap limit settings are used as memory sizing baselines") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "20",
        "spark.executor.memory" -> "6144M",
        "spark.executor.instances" -> "20",
        "spark.rapids.memory.host.offHeapLimit.enabled" -> "true",
        "spark.rapids.memory.host.offHeapLimit.size" -> "80g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
        "spark.rapids.sql.enabled" -> "true"
      )
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(20),
      memoryGB = Some(120L),
      gpuCount = Some(1),
      gpuMemory = Some("48g"),
      gpuDevice = Some("l20"),
      preserveSparkProperties = List(
        "spark.rapids.memory.host.offHeapLimit.enabled",
        "spark.rapids.memory.host.offHeapLimit.size")
    )
    val infoProvider = getMockInfoProvider(0, Seq(0),
      Seq(0.0), logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 20,
      numWorkers = 4,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, _) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val recommendedProps = properties.map(p => p.name -> p.getTuneValue()).toMap

    assert(recommendedProps.get("spark.rapids.memory.host.offHeapLimit.enabled").contains("true"))
    assert(recommendedProps.get("spark.rapids.memory.host.offHeapLimit.size").contains("80g"))
    assert(recommendedProps.get("spark.rapids.memory.pinnedPool.size").contains("40g"),
      s"Expected pinned memory to use preserved host off-heap limit size, " +
        s"got ${recommendedProps.get("spark.rapids.memory.pinnedPool.size")}")
  }

  // Test for https://github.com/NVIDIA/spark-rapids-tools/issues/2074
  // When HEAP_PER_CORE caps executor heap (48g -> 16g with 8 cores), the freed memory
  // should be redistributed into overhead to preserve the total memory budget (~66g).
  test("On-prem K8s with large heap reduced by HEAP_PER_CORE redistributes freed memory") {
    val logEventsProps: mutable.Map[String, String] =
      mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "18",
        "spark.executor.memory" -> "48g",
        "spark.executor.memoryOverhead" -> "18g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.dynamicAllocation.enabled" -> "true",
        "spark.dynamicAllocation.initialExecutors" -> "8",
        "spark.dynamicAllocation.minExecutors" -> "4",
        "spark.dynamicAllocation.maxExecutors" -> "18",
        "spark.executor.resource.gpu.discoveryScript" ->
          "${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin"
      )

    val infoProvider = getMockInfoProvider(8126464.0, Seq(0), Seq(0.004),
      logEventsProps, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

    configureEventLogClusterInfoForTest(
      platform,
      numCores = 8,
      numWorkers = 18,
      gpuCount = 1,
      sparkProperties = logEventsProps.toMap
    )

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    val autoTunerOutput = Profiler.getAutoTunerResultsAsString(properties, comments)
    // scalastyle:off line.size.limit
    // Total memory budget = 48g + 18g = 67584 MB
    // HEAP_PER_CORE (2g) * 8 cores = 16g -> heap capped from 48g to 16g
    // execMemLeft = 67584 - 16384 = 51200 MB
    // pinnedMem = min(8g, (51200 - 1638) / 2) = 8g
    // finalExecutorMemOverhead = max(1638 + 8192 + 8192, 51200) = 51200 = 50g
    val expectedResults =
      s"""|
          |Spark Properties:
          |--conf spark.dynamicAllocation.initialExecutors=18
          |--conf spark.executor.memory=16g
          |--conf spark.executor.memoryOverhead=50g
          |--conf spark.executor.resource.gpu.vendor=nvidia.com
          |--conf spark.locality.wait=0
          |--conf spark.rapids.memory.pinnedPool.size=8g
          |--conf spark.rapids.shuffle.multiThreaded.reader.threads=20
          |--conf spark.rapids.shuffle.multiThreaded.writer.threads=20
          |--conf spark.rapids.sql.batchSizeBytes=2147483647b
          |--conf spark.rapids.sql.concurrentGpuTasks=3
          |--conf spark.rapids.sql.enabled=true
          |--conf spark.rapids.sql.multiThreadedRead.numThreads=20
          |--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$testSmVersion.RapidsShuffleManager
          |--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m
          |--conf spark.sql.adaptive.autoBroadcastJoinThreshold=[FILL_IN_VALUE]
          |--conf spark.sql.adaptive.coalescePartitions.initialPartitionNum=200
          |--conf spark.sql.adaptive.coalescePartitions.minPartitionSize=4m
          |--conf spark.sql.files.maxPartitionBytes=4g
          |--conf spark.task.resource.gpu.amount=0.001
          |
          |Comments:
          |- 'spark.executor.resource.gpu.vendor' was not set.
          |- 'spark.rapids.memory.pinnedPool.size' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.reader.threads' was not set.
          |- 'spark.rapids.shuffle.multiThreaded.writer.threads' was not set.
          |- 'spark.rapids.sql.batchSizeBytes' was not set.
          |- 'spark.rapids.sql.concurrentGpuTasks' was not set.
          |- 'spark.rapids.sql.enabled' was not set.
          |- 'spark.rapids.sql.multiThreadedRead.numThreads' was not set.
          |- 'spark.shuffle.manager' was not set.
          |- 'spark.sql.adaptive.advisoryPartitionSizeInBytes' was not set.
          |- 'spark.sql.adaptive.autoBroadcastJoinThreshold' was not set.
          |- 'spark.sql.adaptive.coalescePartitions.initialPartitionNum' was not set.
          |- 'spark.sql.files.maxPartitionBytes' was not set.
          |- 'spark.task.resource.gpu.amount' was not set.
          |- ${classPathComments("rapids.jars.missing")}
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    // scalastyle:on line.size.limit
    compareOutput(expectedResults, autoTunerOutput)
  }

  //
  // Downward shuffle partition pass
  //

  private val GiB = 1024L * 1024L * 1024L
  private val SHUFFLE_PARTITIONS_KEY = "spark.sql.shuffle.partitions"
  private val AQE_INITIAL_PARTITION_NUM_KEY =
    "spark.sql.adaptive.coalescePartitions.initialPartitionNum"
  private val ADVISORY_PARTITION_SIZE_KEY = "spark.sql.adaptive.advisoryPartitionSizeInBytes"

  /**
   * Recommended cluster of the downward-pass fixture: 25 executors of 16 cores, so one execution
   * wave is 400 task slots on the default cores basis.
   */
  private val DOWNWARD_PASS_WORKERS = 25
  private val DOWNWARD_PASS_SLOTS = 400

  /**
   * Source properties of a GPU application whose normal recommendation lands on 8000 shuffle
   * partitions, which is comfortably above the downward candidate the tests drive.
   */
  private def downwardPassSourceProps(
      extra: Map[String, String] = Map.empty): mutable.Map[String, String] = {
    val base = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "16",
      "spark.executor.instances" -> DOWNWARD_PASS_WORKERS.toString,
      "spark.executor.memory" -> "80g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.sql.adaptive.enabled" -> "true",
      SHUFFLE_PARTITIONS_KEY -> "8000",
      "spark.task.resource.gpu.amount" -> "0.25",
      "spark.rapids.sql.enabled" -> "true",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
      "spark.rapids.sql.concurrentGpuTasks" -> "1")
    extra.foreach { case (k, v) => base.put(k, v) }
    base
  }

  private def recommendedValue(
      properties: Seq[TuningEntryTrait], key: String): Option[String] = {
    properties.find(_.name == key).map(_.getTuneValue())
  }

  /**
   * The downward pass ships disabled, so every test in this section opts in explicitly. Extra
   * entries the caller supplies are merged on top of that opt-in.
   */
  private def downwardPassConfigs(
      enabled: Boolean = true,
      extra: List[TuningConfigEntry] = List.empty): TuningConfiguration = {
    ToolTestUtils.buildTuningConfigs(default =
      TuningConfigEntry(name = "DOWNWARD_SHUFFLE_ENABLED", default = enabled.toString) :: extra)
  }

  private def downwardPassEnabledConfigs: TuningConfiguration = downwardPassConfigs()

  /**
   * Runs the AutoTuner over a GPU application with the given shuffle-stage evidence and returns
   * its recommendations and comments.
   */
  private def runDownwardPass(
      shuffleStageInputAnalysis: ShuffleStageInputAnalysis,
      sourceProps: mutable.Map[String, String] = downwardPassSourceProps(),
      shuffleStagesWithPosSpilling: Set[Long] = Set(),
      gpuShuffleStagesWithContainerOom: Set[Long] = Set(),
      scanStagesWithGpuOom: Set[Long] = Set(),
      meanInputOverride: Option[Double] = None,
      maxColumnarExchangeDataSizeBytes: Option[Long] = None,
      extraTuningConfigs: List[TuningConfigEntry] = List.empty,
      platformName: String = PlatformNames.DATAPROC,
      configureClusterInfo: Boolean = true,
      enableDownwardPass: Boolean = true
  ): (Seq[TuningEntryTrait], Seq[String]) = {
    val userProvidedTuningConfigs =
      Some(downwardPassConfigs(enableDownwardPass, extraTuningConfigs))
    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9,
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = sourceProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = meanInputOverride.getOrElse(1.0E9),
      meanShuffleRead = 1.0E9,
      shuffleStagesWithPosSpilling = shuffleStagesWithPosSpilling,
      gpuShuffleStagesWithContainerOom = gpuShuffleStagesWithContainerOom,
      scanStagesWithGpuOom = scanStagesWithGpuOom,
      maxColumnarExchangeDataSizeBytes = maxColumnarExchangeDataSizeBytes,
      shuffleStageInputAnalysis = shuffleStageInputAnalysis)
    val platform = PlatformFactory.createInstance(platformName)
    if (configureClusterInfo) {
      configureEventLogClusterInfoForTest(platform, numCores = 16,
        numWorkers = DOWNWARD_PASS_WORKERS, gpuCount = 1, sparkProperties = sourceProps.toMap)
    }
    val autoTuner =
      buildAutoTunerForTests(infoProvider, platform, None, userProvidedTuningConfigs)
    val (properties, comments) = autoTuner.getRecommendedProperties()
    (properties, comments.map(_.comment))
  }

  /** A single consumer stage that needs 900 partitions at the default 1 GiB target. */
  private def worstStageNeeding900Partitions: ShuffleStageInputAnalysis = {
    completeShuffleStageInputs(Seq(4 -> 900L * GiB))
  }

  test("Downward pass lowers both partition properties atomically when AQE coalescing is on") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions)
    // 900 partitions rounds up to 3 whole waves of 400 slots.
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).contains("1200"))
    assert(recommendedValue(properties, AQE_INITIAL_PARTITION_NUM_KEY).contains("1200"))
    val appliedComment = comments.filter(_.contains("lowered from 8000 to 1200"))
    assert(appliedComment.size == 1, s"expected exactly one applied comment in: $comments")
    // The comment must identify every input of the decision so it can be audited.
    Seq(SHUFFLE_PARTITIONS_KEY, AQE_INITIAL_PARTITION_NUM_KEY, "measured", "stage 4",
      "900", "1.0", "3 execution wave(s)", s"$DOWNWARD_PASS_SLOTS cluster task slots")
      .foreach { fragment =>
        assert(appliedComment.head.contains(fragment),
          s"'$fragment' missing from: ${appliedComment.head}")
      }
  }

  test("Downward pass keeps the normal recommendation and stays quiet with no recommended cluster") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      configureClusterInfo = false)
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    // An unknown cluster shape is the common qualification case, so it must add no comment.
    assert(!comments.exists(c => c.contains("lowered from") || c.contains("slot")))
  }

  test("Downward pass leaves the AQE advisory partition size untouched") {
    val advisorySize = "64m"
    val (properties, _) = runDownwardPass(worstStageNeeding900Partitions,
      sourceProps = downwardPassSourceProps(Map(ADVISORY_PARTITION_SIZE_KEY -> advisorySize)))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).contains("1200"))
    assert(recommendedValue(properties, ADVISORY_PARTITION_SIZE_KEY).forall(_ == advisorySize))
  }

  test("Downward pass updates only the shuffle property when AQE coalescing is disabled") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      sourceProps = downwardPassSourceProps(
        Map("spark.sql.adaptive.coalescePartitions.enabled" -> "false")))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).contains("1200"))
    assert(recommendedValue(properties, AQE_INITIAL_PARTITION_NUM_KEY).isEmpty)
    assert(comments.exists(_.contains("lowered from 8000 to 1200")))
  }

  test("Downward pass preserves an increase made for shuffle spilling") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      shuffleStagesWithPosSpilling = Set(1L))
    // The spill-driven doubling must survive; nothing may be lowered.
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).contains("16000"))
    assert(comments.exists(_.contains("Shuffle partitions should be increased since spilling")))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass preserves an increase made for shuffle task OOM") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      gpuShuffleStagesWithContainerOom = Set(1L))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).exists(_.toInt >= 8000))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass preserves the existing GPU ColumnarExchange lower bound") {
    // A 60000 GiB exchange against the ~2g batch size forces partitions above 8000.
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      maxColumnarExchangeDataSizeBytes = Some(60000L * GiB))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).exists(_.toInt > 8000))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass is blocked by spill in an affected consumer stage") {
    val withSpill = completeShuffleStageInputs(Seq(4 -> 900L * GiB), hasPositiveSpill = true)
    val (properties, comments) = runDownwardPass(withSpill)
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass is blocked by skew in an affected consumer stage") {
    val withSkew = completeShuffleStageInputs(Seq(4 -> 900L * GiB), hasSkew = true)
    val (properties, comments) = runDownwardPass(withSkew)
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass warns once and changes nothing when evidence is incomplete") {
    val (properties, comments) = runDownwardPass(incompleteShuffleStageInputs())
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    val warnings = comments.filter(_.contains("could not be measured"))
    assert(warnings.size == 1, s"expected exactly one incomplete-evidence comment in: $comments")
  }

  test("Downward pass warns once and changes nothing when its configuration is invalid") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      extraTuningConfigs = List(
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS", default = "0")))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    val warnings = comments.filter(_.contains("tuning configuration is invalid"))
    assert(warnings.size == 1, s"expected exactly one invalid-config comment in: $comments")
  }

  test("Downward pass does nothing when disabled") {
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      enableDownwardPass = false)
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    // A disabled feature must stay completely silent in the user-facing output.
    assert(!comments.exists(c => c.contains("lowered from") || c.contains("downward")))
  }

  test("Downward pass respects a configured minimum reduction factor") {
    // 1500 partitions rounds up to 4 waves (1600), which is not a 2x reduction from 3000.
    val (properties, comments) = runDownwardPass(
      completeShuffleStageInputs(Seq(4 -> 1500L * GiB)),
      sourceProps = downwardPassSourceProps(Map(SHUFFLE_PARTITIONS_KEY -> "3000")),
      extraTuningConfigs = List(
        TuningConfigEntry(name = "DOWNWARD_SHUFFLE_MIN_REDUCTION_FACTOR", default = "2.0")))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "3000"))
    assert(!comments.exists(_.contains("lowered from")))
    // The default of 1.0 imposes no threshold, so the same reduction applies without the override.
    val (defaultProps, defaultComments) = runDownwardPass(
      completeShuffleStageInputs(Seq(4 -> 1500L * GiB)),
      sourceProps = downwardPassSourceProps(Map(SHUFFLE_PARTITIONS_KEY -> "3000")))
    assert(recommendedValue(defaultProps, SHUFFLE_PARTITIONS_KEY).contains("1600"))
    assert(defaultComments.exists(_.contains("lowered from 3000 to 1600")))
  }

  test("Downward pass never lowers below a single execution wave") {
    val (properties, comments) = runDownwardPass(completeShuffleStageInputs(Seq(4 -> 1L)))
    // Even a one-byte stage still gets one full wave of the recommended cluster.
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY)
      .contains(DOWNWARD_PASS_SLOTS.toString))
    assert(comments.exists(_.contains(s"lowered from 8000 to $DOWNWARD_PASS_SLOTS")))
    assert(comments.exists(_.contains("1 execution wave(s)")))
  }

  test("Downward pass slot count follows the configured slot basis") {
    // The GPU-concurrency basis uses the recommended concurrentGpuTasks instead of the cores.
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      extraTuningConfigs = List(TuningConfigEntry(name = "DOWNWARD_SHUFFLE_SLOT_BASIS",
        default = "concurrentGpuTasks")))
    val concurrentGpuTasks =
      recommendedValue(properties, "spark.rapids.sql.concurrentGpuTasks").map(_.toInt).getOrElse(
        fail("expected a concurrentGpuTasks recommendation"))
    val gpuSlots = DOWNWARD_PASS_WORKERS * concurrentGpuTasks
    assert(gpuSlots != DOWNWARD_PASS_SLOTS,
      "the two bases must differ for this test to prove anything")
    val expected = math.ceil(900.0 / gpuSlots).toInt * gpuSlots
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).contains(expected.toString))
    assert(comments.exists(_.contains(s"$gpuSlots cluster task slots")))
  }

  test("Downward pass changes neither property when one of them is enforced") {
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = Map(AQE_INITIAL_PARTITION_NUM_KEY -> "4096"))
    val sourceProps = downwardPassSourceProps()
    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9,
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = sourceProps,
      sparkVersion = Some(testSparkVersion),
      meanInput = 1.0E9,
      meanShuffleRead = 1.0E9,
      shuffleStageInputAnalysis = worstStageNeeding900Partitions)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 16,
      numWorkers = DOWNWARD_PASS_WORKERS, gpuCount = 1, sparkProperties = sourceProps.toMap)
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, None,
      Some(downwardPassEnabledConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    // The enforced AQE property blocks the whole update, so the shuffle property is untouched.
    assert(recommendedValue(properties, AQE_INITIAL_PARTITION_NUM_KEY).contains("4096"))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    assert(!comments.map(_.comment).exists(_.contains("lowered from")))
  }

  test("Downward pass is blocked while Databricks automatic shuffle optimization is active") {
    val autoOptimizeKey = "spark.databricks.adaptive.autoOptimizeShuffle.enabled"
    // The user enforces the Databricks setting, so normal tuning cannot turn it off and the
    // runtime, not this recommendation, still governs partitioning.
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      enforcedSparkProperties = Map(autoOptimizeKey -> "true"))
    val sourceProps = downwardPassSourceProps(Map(autoOptimizeKey -> "true"))
    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9,
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = sourceProps,
      sparkVersion = Some(testDatabricksVersion),
      meanInput = 1.0E9,
      meanShuffleRead = 1.0E9,
      shuffleStageInputAnalysis = worstStageNeeding900Partitions)
    val platform =
      PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS, Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 16,
      numWorkers = DOWNWARD_PASS_WORKERS, gpuCount = 1, sparkProperties = sourceProps.toMap)
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, None,
      Some(downwardPassEnabledConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    assert(!comments.map(_.comment).exists(_.contains("lowered from")))
  }

  test("Downward pass falls back to the cluster record when the platform excludes instances") {
    // Databricks excludes 'spark.executor.instances', so no recommendation exists for it and the
    // slot count must come from the recommended cluster rather than the source CPU value.
    val sourceProps = downwardPassSourceProps()
    val infoProvider = getMockInfoProvider(
      maxInput = 1.0E9,
      spilledMetrics = Seq(0, 0),
      jvmGCFractions = Seq(0.1, 0.1),
      propsFromLog = sourceProps,
      sparkVersion = Some(testDatabricksVersion),
      meanInput = 1.0E9,
      meanShuffleRead = 1.0E9,
      shuffleStageInputAnalysis = worstStageNeeding900Partitions)
    val platform = PlatformFactory.createInstance(PlatformNames.DATABRICKS_AWS)
    configureEventLogClusterInfoForTest(platform, numCores = 16,
      numWorkers = DOWNWARD_PASS_WORKERS, gpuCount = 1, sparkProperties = sourceProps.toMap)
    val autoTuner = buildAutoTunerForTests(infoProvider, platform, None,
      Some(downwardPassEnabledConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties()
    assert(recommendedValue(properties, "spark.executor.instances").isEmpty,
      "Databricks should not recommend an executor count")
    val clusterInfo = platform.recommendedClusterInfo.getOrElse(
      fail("expected a recommended cluster"))
    val expectedSlots = clusterInfo.numExecutors * clusterInfo.coresPerExecutor
    // Reading the source CPU executor cores instead would have produced a different wave.
    assert(expectedSlots != DOWNWARD_PASS_SLOTS,
      "the recommended cluster must differ from the source one for this test to prove anything")
    val expected = math.ceil(900.0 / expectedSlots).toInt * expectedSlots
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).contains(expected.toString))
    assert(comments.map(_.comment).exists(_.contains(s"$expectedSlots cluster task slots")))
  }

  test("Downward pass is blocked when the application had a failed stage") {
    val withFailure = completeShuffleStageInputs(Seq(4 -> 900L * GiB), appHasFailedStage = true)
    val (properties, comments) = runDownwardPass(withFailure)
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass is blocked when the application had an OOM") {
    // A GPU OOM anywhere in the application disqualifies the run as sizing evidence, even
    // though the consumer stages themselves look clean.
    val (properties, comments) = runDownwardPass(worstStageNeeding900Partitions,
      scanStagesWithGpuOom = Set(9L))
    assert(recommendedValue(properties, SHUFFLE_PARTITIONS_KEY).forall(_ == "8000"))
    assert(!comments.exists(_.contains("lowered from")))
  }

  test("Downward pass leaves the two partition properties in agreement") {
    // The pass updates the AQE partition property only when a value for it already exists. In a
    // normal AQE run recommendAQEProperties has already appended one, so the guard is inert and
    // both properties move together; what users depend on is that they never disagree.
    val sourceProps = downwardPassSourceProps()
    sourceProps.remove(AQE_INITIAL_PARTITION_NUM_KEY)
    val (properties, _) = runDownwardPass(worstStageNeeding900Partitions,
      sourceProps = sourceProps)
    val shuffle = recommendedValue(properties, SHUFFLE_PARTITIONS_KEY)
    val aqe = recommendedValue(properties, AQE_INITIAL_PARTITION_NUM_KEY)
    assert(shuffle.contains("1200"))
    assert(aqe.forall(_ == shuffle.get),
      s"partition properties disagree: shuffle=$shuffle aqe=$aqe")
  }
  test("PySpark memory evidence without a positive source limit does not enable telemetry " +
      "by default") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)

    assert(!properties.exists(_.name == "spark.executor.pyspark.memory"))
    val values = properties.map(property => property.name -> property.getTuneValue()).toMap
    assert(!values.contains("spark.executor.processTreeMetrics.enabled"))
    assert(!values.contains("spark.eventLog.logStageExecutorMetrics"))
    assert(!values.contains("spark.executor.metrics.pollingInterval"))
    val guidance = comments.mkString("\n")
    assert(!guidance.contains("PySpark memory autotuning needs a telemetry-enabled retry"))
  }

  test("OVERHEAD PySpark rebalance on standalone emits no partial recommendation") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
    val tuningConfigs = ToolTestUtils.buildTuningConfigs(profiling = List(
      TuningConfigEntry(name = "PYSPARK_MEMORY_REBALANCE_SOURCE", default = "OVERHEAD")))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Standalone),
      Some(tuningConfigs))
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)

    assert(!properties.exists(p => p.name == "spark.executor.pyspark.memory" && p.isTuned()))
    assert(comments.exists(_.comment.contains(
      "executor overhead is supported only for YARN and Kubernetes")))
  }

  test("invalid PySpark memory policy fails eagerly without failure evidence") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(testSparkVersion))
    val tuningConfigs = ToolTestUtils.buildTuningConfigs(profiling = List(
      TuningConfigEntry(
        name = "PYSPARK_MEMORY_EVIDENCE_HEADROOM_MULTIPLIER", default = "1.0")))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val error = intercept[IllegalArgumentException] {
      buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes), Some(tuningConfigs))
    }
    assert(error.getMessage.contains("PYSPARK_MEMORY_EVIDENCE_HEADROOM_MULTIPLIER"))
  }

  test("OVERHEAD PySpark rebalances preserve budget for evidence and telemetry retry") {
    val overheadConfigs = ToolTestUtils.buildTuningConfigs(profiling = List(
      TuningConfigEntry(name = "PYSPARK_MEMORY_REBALANCE_SOURCE", default = "OVERHEAD"),
      TuningConfigEntry(
        name = "PYSPARK_MEMORY_RECOMMEND_TELEMETRY_CONFIGS", default = "true")))

    def run(evidence: Seq[PySparkMemoryEvidence]):
        (Map[String, Long], Map[String, String], Seq[String]) = {
      val sourceProps = mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.executor.pyspark.memory" -> "4g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
        Some(reliableProcessTreeMetricsSparkVersion), pySparkMemoryEvidence = evidence)
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4.toString))
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
        Some(targetClusterInfo))
      configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
        sparkProperties = sourceProps.toMap)
      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes),
        Some(overheadConfigs))
      val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
      val memoryKeys = Set("spark.executor.memory", "spark.executor.memoryOverhead",
        "spark.executor.pyspark.memory", "spark.memory.offHeap.size")
      val memory = properties.iterator.filter { property =>
        memoryKeys.contains(property.name) && property.getTuneValue() != "[FILL_IN_VALUE]"
      }.map { property =>
        property.name -> StringUtils.convertToMB(property.getTuneValue(), Some(ByteUnit.BYTE))
      }.toMap
      (memory, properties.map(property => property.name -> property.getTuneValue()).toMap,
        comments.map(_.comment))
    }

    val (base, _, _) = run(Seq.empty)
    val peak55GiB = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
    val (evidenceAdjusted, evidenceProperties, evidenceComments) =
      run(Seq(PySparkMemoryEvidence(1, 0, Seq(peak55GiB))))
    assert(evidenceAdjusted("spark.executor.pyspark.memory") == 7L * 1024L,
      evidenceComments.mkString("\n"))
    assert(evidenceAdjusted("spark.executor.memory") == base("spark.executor.memory"))
    assert(evidenceAdjusted("spark.executor.memoryOverhead") ==
      base("spark.executor.memoryOverhead") - 3L * 1024L)
    assert(evidenceAdjusted.values.sum == base.values.sum)
    assert(!evidenceProperties.contains("spark.executor.processTreeMetrics.enabled"))
    assert(!evidenceProperties.contains("spark.eventLog.logStageExecutorMetrics"))
    assert(!evidenceProperties.contains("spark.executor.metrics.pollingInterval"))

    val (retryAdjusted, retryProperties, retryComments) =
      run(Seq(PySparkMemoryEvidence(2, 0, Seq.empty)))
    assert(retryAdjusted("spark.executor.pyspark.memory") == 6L * 1024L)
    assert(retryAdjusted("spark.executor.memoryOverhead") ==
      base("spark.executor.memoryOverhead") - 2L * 1024L)
    assert(retryAdjusted.values.sum == base.values.sum)
    assert(retryProperties("spark.executor.processTreeMetrics.enabled") == "true")
    assert(retryProperties("spark.eventLog.logStageExecutorMetrics") == "true")
    assert(retryProperties("spark.executor.metrics.pollingInterval") == "5000")
    assert(retryComments.exists(_.contains("telemetry-enabled retry")))

    val peak12GiB = 12L * 1024L * 1024L * 1024L
    val (uncappedAdjusted, _, _) = run(Seq(PySparkMemoryEvidence(3, 0, Seq(peak12GiB))))
    assert(uncappedAdjusted("spark.executor.pyspark.memory") == 15L * 1024L)
    assert(uncappedAdjusted.values.sum == base.values.sum)
  }

  test("PySpark HEAP rebalance narrowly supersedes preserved coordinated values") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString),
      preserveSparkProperties = List(
        "spark.executor.memory", "spark.executor.pyspark.memory", "spark.executor.cores"))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
      Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val values = properties.map(property => property.name -> property.getTuneValue()).toMap

    assert(values("spark.executor.memory") == "29g")
    assert(values("spark.executor.pyspark.memory") == "7g")
    val coordinatedTotalMB = Seq("spark.executor.memory", "spark.executor.pyspark.memory")
      .map(key => StringUtils.convertToMB(values(key), Some(ByteUnit.BYTE))).sum
    assert(coordinatedTotalMB == 36L * 1024L)
    assert(values("spark.executor.cores") == "8")
    assert(!comments.exists(_.comment == getPreservedPropertyComment("spark.executor.memory")))
    assert(!comments.exists(
      _.comment == getPreservedPropertyComment("spark.executor.pyspark.memory")))
    assert(comments.exists(_.comment == getPreservedPropertyComment("spark.executor.cores")))
    assert(!comments.exists(_.comment.contains("constraint=")), comments.mkString("\n"))
  }

  test("PySpark HEAP rebalance can transfer from normally calculated heap") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "16",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(16), memoryGB = Some(64), gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString))
    val platform = PlatformFactory.createInstance(PlatformNames.EMR,
      Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 16, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Yarn))
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val values = properties.map(property => property.name -> property.getTuneValue()).toMap

    assert(values("spark.executor.memory") == "29g")
    assert(values("spark.executor.pyspark.memory") == "7g")
    val coordinatedTotalMB = Seq("spark.executor.memory", "spark.executor.pyspark.memory")
      .map(key => StringUtils.convertToMB(values(key), Some(ByteUnit.BYTE))).sum
    assert(coordinatedTotalMB == 36L * 1024L)
    assert(!comments.exists(_.comment.contains("constraint=")), comments.mkString("\n"))
  }

  test("PySpark coordinated rebalance reports enforced conflicts without a partial pair") {
    val telemetryConfigs = ToolTestUtils.buildTuningConfigs(profiling = List(
      TuningConfigEntry(
        name = "PYSPARK_MEMORY_RECOMMEND_TELEMETRY_CONFIGS", default = "true")))

    def run(
        enforced: Map[String, String],
        withMetrics: Boolean = true): (Map[String, String], Seq[String]) = {
      val sourceProps = mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "32g",
        "spark.executor.pyspark.memory" -> "4g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
      val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
        Some(reliableProcessTreeMetricsSparkVersion),
        pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0,
          if (withMetrics) Seq(peakBytes) else Seq.empty)))
      val preserved = if (enforced.contains("spark.executor.memory")) List.empty
        else List("spark.executor.memory")
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4.toString), enforcedSparkProperties = enforced,
        preserveSparkProperties = preserved)
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
        Some(targetClusterInfo))
      configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
        sparkProperties = sourceProps.toMap)
      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes),
        Some(telemetryConfigs))
      val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
      (properties.map(property => property.name -> property.getTuneValue()).toMap,
        comments.map(_.comment))
    }

    Seq(
      Map("spark.executor.pyspark.memory" -> "5g"),
      Map("spark.executor.memory" -> "32g")
    ).foreach { enforced =>
      val (values, comments) = run(enforced)
      assert(values("spark.executor.memory") == "32g")
      assert(values.get("spark.executor.pyspark.memory").forall(_ != "7g"))
      val conflict = comments.filter(_.contains("constraint=enforced"))
      assert(conflict.size == 1)
      Seq("observedCurrentMB=4096", "layoutCurrentMB=", "candidateMB=7168",
        "requiredDeltaMB=", "selectedSource=spark.executor.memory", "availableDeltaMB=")
        .foreach(field => assert(conflict.head.contains(field), conflict.head))
    }

    Seq("7g", "8g").foreach { enforcedTarget =>
      val (values, comments) = run(Map("spark.executor.pyspark.memory" -> enforcedTarget))
      assert(values("spark.executor.memory") == "32g")
      assert(values("spark.executor.pyspark.memory") == enforcedTarget)
      assert(!comments.exists(_.contains("constraint=enforced")))
    }

    val (retryValues, retryComments) =
      run(Map("spark.executor.memory" -> "32g"), withMetrics = false)
    assert(retryValues.get("spark.executor.pyspark.memory").forall(_ != "6g"))
    assert(retryValues("spark.executor.processTreeMetrics.enabled") == "true")
    assert(retryValues("spark.eventLog.logStageExecutorMetrics") == "true")
    assert(retryValues("spark.executor.metrics.pollingInterval") == "5000")
    assert(retryComments.count(_.contains("constraint=enforced")) == 1)
    assert(retryComments.exists(_.contains("telemetry-enabled retry")))
  }

  test("PySpark conflict comments identify capacity and source capability") {
    def run(heap: String, master: SparkMaster, overhead: Boolean):
        (Map[String, String], Seq[String]) = {
      val sourceProps = mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> heap,
        "spark.executor.pyspark.memory" -> "4g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
      val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
        Some(reliableProcessTreeMetricsSparkVersion),
        pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4.toString),
        preserveSparkProperties = List("spark.executor.memory"))
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
        Some(targetClusterInfo))
      configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
        sparkProperties = sourceProps.toMap)
      val configs = if (overhead) Some(ToolTestUtils.buildTuningConfigs(profiling = List(
        TuningConfigEntry(name = "PYSPARK_MEMORY_REBALANCE_SOURCE", default = "OVERHEAD"))))
      else None
      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(master), configs)
      val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
      (properties.map(property => property.name -> property.getTuneValue()).toMap,
        comments.map(_.comment))
    }

    val (capacityValues, capacityComments) = run("8g", Kubernetes, overhead = false)
    assert(capacityValues("spark.executor.memory") == "8g")
    assert(capacityValues.get("spark.executor.pyspark.memory").forall(_ != "7g"))
    assert(capacityComments.count(_.contains("constraint=capacity")) == 1)

    val (masterValues, masterComments) = run("32g", Standalone, overhead = true)
    assert(masterValues("spark.executor.memory") == "32g")
    assert(masterValues.get("spark.executor.pyspark.memory").forall(_ != "7g"))
    assert(masterComments.count(_.contains("constraint=source-capability")) == 1)
  }

  test("PySpark coordinated rebalance rejects excluded pair members") {
    def run(
        overhead: Boolean,
        excluded: Option[String],
        withEvidence: Boolean): (Map[String, String], Seq[String]) = {
      val sourceProps = mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "32g",
        "spark.executor.pyspark.memory" -> "4g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
      val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
        Some(reliableProcessTreeMetricsSparkVersion),
        pySparkMemoryEvidence = if (withEvidence) {
          Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes)))
        } else {
          Seq.empty
        })
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4.toString), excludeSparkProperties = excluded.toList)
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
        Some(targetClusterInfo))
      configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
        sparkProperties = sourceProps.toMap)
      val configs = if (overhead) Some(ToolTestUtils.buildTuningConfigs(profiling = List(
        TuningConfigEntry(name = "PYSPARK_MEMORY_REBALANCE_SOURCE", default = "OVERHEAD"))))
      else None
      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes), configs)
      val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
      (properties.map(property => property.name -> property.getTuneValue()).toMap,
        comments.map(_.comment))
    }

    Seq(false, true).foreach { overhead =>
      val sourceKey = if (overhead) "spark.executor.memoryOverhead" else "spark.executor.memory"
      val (baseValues, _) = run(overhead, None, withEvidence = false)
      Seq(sourceKey, "spark.executor.pyspark.memory").foreach { excluded =>
        val (values, comments) = run(overhead, Some(excluded), withEvidence = true)
        assert(values.get("spark.executor.pyspark.memory").forall(_ != "7g"))
        if (excluded == sourceKey) {
          assert(!values.contains(sourceKey), values)
        } else {
          assert(values.get(sourceKey) == baseValues.get(sourceKey), values)
        }
        assert(comments.count(_.contains("constraint=output-eligibility")) == 1,
          comments.mkString("\n"))
      }
    }
  }

  test("PySpark arithmetic overflow emits one structured conflict") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(6L * 1024L * 1024L * 1024L))))
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString),
      preserveSparkProperties = List("spark.executor.memory"))
    val configs = ToolTestUtils.buildTuningConfigs(profiling = List(TuningConfigEntry(
      name = "PYSPARK_MEMORY_EVIDENCE_HEADROOM_MULTIPLIER", default = "1e100")))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
      Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes), Some(configs))
    val (properties, comments) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    assert(properties.find(_.name == "spark.executor.memory").get.getTuneValue() == "32g")
    assert(properties.find(_.name == "spark.executor.pyspark.memory")
      .forall(_.getTuneValue() != "7g"))
    val conflicts = comments.map(_.comment).filter(_.contains("constraint=arithmetic"))
    assert(conflicts.size == 1)
    Seq("observedCurrentMB=4096", "layoutCurrentMB=4096", "candidateMB=overflow",
      "requiredDeltaMB=overflow", "selectedSource=spark.executor.memory",
      "availableDeltaMB=unknown").foreach(field => assert(conflicts.head.contains(field)))
  }

  test("PySpark evidence uses the nearest-rank p95 boundary") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val lowPeak = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
    val highPeak = 20L * 1024L * 1024L * 1024L
    // ceil(20 * 0.95) is rank 19, so the single rank-20 outlier must not affect sizing.
    val peaks = Seq.fill(19)(lowPeak) :+ highPeak
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, peaks)))
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString),
      preserveSparkProperties = List(
        "spark.executor.memory", "spark.executor.pyspark.memory"))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
      Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, _) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val values = properties.map(property => property.name -> property.getTuneValue()).toMap

    assert(values("spark.executor.memory") == "29g")
    assert(values("spark.executor.pyspark.memory") == "7g")
  }

  test("PySpark evidence target retains the retry growth floor") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val peakBytes = 2L * 1024L * 1024L * 1024L
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
      gpuDevice = Some(GpuTypes.L4.toString),
      preserveSparkProperties = List(
        "spark.executor.memory", "spark.executor.pyspark.memory"))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
      Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
      sparkProperties = sourceProps.toMap)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
    val (properties, _) = autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val values = properties.map(property => property.name -> property.getTuneValue()).toMap

    assert(values("spark.executor.memory") == "30g")
    assert(values("spark.executor.pyspark.memory") == "6g")
  }

  forAll(Table(
    ("sparkVersion", "expectedPySparkMemory", "expectsVersionWarning"),
    ("3.5.6", "6g", true),
    ("3.5.7", "7g", false),
    ("4.0.0", "6g", true),
    ("4.0.1", "7g", false)
  )) { (sparkVersion, expectedPySparkMemory, expectsVersionWarning) =>
    test(s"PySpark memory evidence reliability for Spark $sparkVersion") {
      val sourceProps = mutable.LinkedHashMap[String, String](
        "spark.executor.cores" -> "8",
        "spark.executor.instances" -> "2",
        "spark.executor.memory" -> "32g",
        "spark.executor.pyspark.memory" -> "4g",
        "spark.executor.resource.gpu.amount" -> "1",
        "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
      val peakBytes = (BigDecimal("5.5") * BigDecimal(1024L * 1024L * 1024L)).toLong
      val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
        Some(sparkVersion),
        pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq(peakBytes))))
      val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(8), memoryGB = Some(128), gpuCount = Some(1),
        gpuDevice = Some(GpuTypes.L4.toString))
      val platform = PlatformFactory.createInstance(PlatformNames.ONPREM,
        Some(targetClusterInfo))
      configureEventLogClusterInfoForTest(platform, numCores = 8, numWorkers = 2,
        sparkProperties = sourceProps.toMap)

      val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes))
      val (properties, comments) =
        autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
      val values = properties.map(property => property.name -> property.getTuneValue()).toMap

      assert(values("spark.executor.pyspark.memory") == expectedPySparkMemory)
      assert(comments.exists(_.comment.contains("unreliable procfs metrics")) ==
        expectsVersionWarning)
    }
  }

  test("opted-in PySpark telemetry guidance does not require cluster sizing") {
    val sourceProps = mutable.LinkedHashMap[String, String](
      "spark.executor.cores" -> "8",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "32g",
      "spark.executor.pyspark.memory" -> "4g",
      "spark.executor.resource.gpu.amount" -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin")
    val infoProvider = getMockInfoProvider(0, Seq(0), Seq(0.0), sourceProps,
      Some(reliableProcessTreeMetricsSparkVersion),
      pySparkMemoryEvidence = Seq(PySparkMemoryEvidence(1, 0, Seq.empty)))
    val tuningConfigs = ToolTestUtils.buildTuningConfigs(profiling = List(
      TuningConfigEntry(
        name = "PYSPARK_MEMORY_RECOMMEND_TELEMETRY_CONFIGS", default = "true")))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)

    val autoTuner = buildAutoTunerForTests(infoProvider, platform, Some(Kubernetes),
      Some(tuningConfigs))
    val (properties, comments) =
      autoTuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val values = properties.map(property => property.name -> property.getTuneValue()).toMap

    assert(values("spark.executor.processTreeMetrics.enabled") == "true")
    assert(values("spark.eventLog.logStageExecutorMetrics") == "true")
    assert(values("spark.executor.metrics.pollingInterval") == "5000")
    assert(comments.exists(_.comment.contains("telemetry-enabled retry")))
    assert(values("spark.executor.pyspark.memory") == "4g")
  }

}
