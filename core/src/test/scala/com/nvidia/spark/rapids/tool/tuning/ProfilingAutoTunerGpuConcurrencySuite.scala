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

package com.nvidia.spark.rapids.tool.tuning

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.{PlatformFactory, PlatformNames, ToolTestUtils}
import com.nvidia.spark.rapids.tool.profiling.{Profiler, RecommendedCommentResult,
  StageAggGpuMetricsProfileResult}
import com.nvidia.spark.rapids.tool.tuning.config.{TuningConfigEntry, TuningConfiguration,
  TuningEntryDefinition}
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.annotation.Since

/**
 * Profiling AutoTuner integration coverage for footprint-based concurrent GPU task tuning.
 *
 * The scenarios use the explicit target-cluster format exercised by
 * [[ProfilingAutoTunerSuiteV2]], while keeping this feature's decision matrix isolated.
 */
@Since("26.08.2")
class ProfilingAutoTunerGpuConcurrencySuite extends ProfilingAutoTunerSuiteBase {
  private val concurrentGpuTasksKey = GpuBatchAndConcurrencyKeys.ConcurrentGpuTasks
  private val batchSizeKey = GpuBatchAndConcurrencyKeys.BatchSizeBytes
  private lazy val footprintOutputSkipList = {
    val retainedKeys = SparkTaskSlotModel.TargetTaskResourceKeys ++
      Set(batchSizeKey, concurrentGpuTasksKey)
    (TuningEntryDefinition.TUNING_TABLE.keySet -- retainedKeys).toSeq.sorted
  }

  private case class FootprintScenarioResult(
      properties: Seq[TuningEntryTrait],
      comments: Seq[RecommendedCommentResult],
      combinedProperties: Map[String, String]) {
    private val duplicatePropertyKeys = properties.groupBy(_.name).collect {
      case (key, entries) if entries.size > 1 => key
    }.toSeq.sorted
    assert(duplicatePropertyKeys.isEmpty,
      s"duplicate recommendations found for: ${duplicatePropertyKeys.mkString(", ")}")

    def values(key: String): Seq[String] = {
      properties.filter(_.name == key).map(_.getTuneValue())
    }

    def value(key: String): Option[String] = {
      val matchingValues = values(key)
      assert(matchingValues.size <= 1,
        s"expected at most one '$key' recommendation, found $matchingValues")
      matchingValues.headOption
    }

    def combinedValue(key: String): Option[String] = combinedProperties.get(key)

    def commentText: String = comments.map(_.comment).mkString("\n")

    def footprintComments: Seq[String] = {
      comments.map(_.comment).filter(_.contains(GpuConcurrencySeed.FootprintMetricName))
    }

    def renderedOutput: String = Profiler.getAutoTunerResultsAsString(properties, comments)
  }

  private def footprintMetric(
      stageId: Int = 7,
      maxBytes: Long = ByteUnit.GiB.toBytes(4L)): StageAggGpuMetricsProfileResult = {
    val minBytes = maxBytes / 2L
    StageAggGpuMetricsProfileResult(
      stageId = stageId,
      numTasks = 4,
      metricName = GpuConcurrencySeed.FootprintMetricName,
      unit = "bytes",
      total = None,
      max = Some(maxBytes),
      count = 2L,
      min = Some(minBytes),
      welfordSumSqDev = math.pow(maxBytes.toDouble, 2.0) / 8.0,
      sampleTotal = Some(maxBytes + minBytes))
  }

  private def runFootprintScenario(
      sourceBatch: Option[String] = Some("2147483647"),
      targetEnforced: Map[String, String] = Map.empty,
      targetPreserve: List[String] = List.empty,
      targetExclude: List[String] = List.empty,
      sourceExtra: Map[String, String] = Map.empty,
      evidence: Seq[StageAggGpuMetricsProfileResult] = Seq(footprintMetric()),
      pluginVersion: String = "25.08.0",
      rapidsJars: Option[Seq[String]] = None,
      targetGpuMemory: Option[String] = Some("16g"),
      tuningConfigs: Option[TuningConfiguration] = None,
      skipList: Seq[String] = Seq.empty,
      limitedLogicList: Seq[String] = Seq.empty,
      shuffleStagesWithPosSpilling: Set[Long] = Set.empty,
      maxColumnarExchangeDataSizeBytes: Option[Long] = None,
      showOnlyUpdatedProps: Boolean = true): FootprintScenarioResult = {
    val sourceProperties = mutable.LinkedHashMap[String, String](
      SparkTaskSlotModel.Properties.ExecutorCores -> "16",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "64g",
      SparkTaskSlotModel.Properties.ExecutorGpuAmount -> "1",
      "spark.plugins" -> "com.nvidia.spark.SQLPlugin",
      "spark.rapids.sql.enabled" -> "true",
      SparkTaskSlotModel.Properties.TaskGpuAmount -> "0.25")
    sourceBatch.foreach(sourceProperties.put(batchSizeKey, _))
    sourceExtra.foreach { case (key, value) => sourceProperties.put(key, value) }

    val targetClusterInfo = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("g2-standard-16"),
      gpuMemory = targetGpuMemory,
      enforcedSparkProperties = targetEnforced,
      preserveSparkProperties = targetPreserve,
      excludeSparkProperties = targetExclude)
    val platform = PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(targetClusterInfo))
    configureEventLogClusterInfoForTest(
      platform,
      numCores = 16,
      numWorkers = 2,
      gpuCount = 1,
      sparkProperties = sourceProperties.toMap)
    val infoProvider = getMockInfoProvider(
      maxInput = 0,
      spilledMetrics = Seq(0),
      jvmGCFractions = Seq(0.0),
      propsFromLog = sourceProperties,
      sparkVersion = Some(testSparkVersion),
      rapidsJars = rapidsJars.getOrElse(
        Seq(s"rapids-4-spark_2.12-$pluginVersion.jar")),
      shuffleStagesWithPosSpilling = shuffleStagesWithPosSpilling,
      maxColumnarExchangeDataSizeBytes = maxColumnarExchangeDataSizeBytes,
      gpuStageAggMetrics = evidence)
    val autoTuner = buildAutoTunerForTests(
      infoProvider, platform, userProvidedTuningConfigs = tuningConfigs)
    val (properties, comments) = autoTuner.getRecommendedProperties(
        skipList = Some(skipList),
        limitedLogicList = Some(limitedLogicList),
        showOnlyUpdatedProps = showOnlyUpdatedProps)
    val combinedProperties = autoTuner.combineSparkProperties(properties)
      .map(result => result.property -> result.value).toMap
    FootprintScenarioResult(properties, comments, combinedProperties)
  }

  test("modern plugin uses equal-batch footprint evidence without the legacy max-four cap") {
    val pluginVersion = "25.08.0"
    val result = runFootprintScenario(
      sourceBatch = Some(" 1G "),
      sourceExtra = Map(SparkTaskSlotModel.Properties.TaskGpuAmount -> "0.001"),
      pluginVersion = pluginVersion,
      tuningConfigs = Some(ToolTestUtils.buildTuningConfigs(profiling = List(
        TuningConfigEntry(name = "BATCH_SIZE_BYTES", default = "1024m")))),
      evidence = Seq(footprintMetric(maxBytes = ByteUnit.GiB.toBytes(1L))),
      skipList = footprintOutputSkipList)

    result.value(concurrentGpuTasksKey) shouldBe Some("15")
    result.value(batchSizeKey) shouldBe None
    result.combinedValue(concurrentGpuTasksKey) shouldBe Some("15")
    val combinedBatch = result.combinedValue(batchSizeKey)
      .flatMap(RuntimeConfigParser.parsePositiveBytes)
    combinedBatch shouldBe Some(RuntimeConfigParser.ParsedBytes(ByteUnit.GiB.toBytes(1L)))

    val expectedFootprintComment =
      s"'$concurrentGpuTasksKey' was seeded at 15 from historical " +
        s"'${GpuConcurrencySeed.FootprintMetricName}'. Stage 7 reported a maximum of " +
        "1073741824b across 2 task-attempt update(s), with CV 0.47 and max/mean 1.33. " +
        "The unpadded nominal target RMM pool is 16498294784b. The model treats " +
        "17179869184b from explicit target configuration as both CUDA total and pre-startup " +
        "free memory, subtracts the 10485760b chunked-pack allocation and leaves " +
        "17169383424b nominal free before the 671088640b base reserve. The effective reserve " +
        "is 671088640b. The ASYNC allocator with MULTITHREADED shuffle applies alloc/min/max " +
        "fractions 1.0/0.25/1.0, and truncates RMM values to 512-byte alignment under the " +
        "discrete-gpu-rmm-v1 compatibility profile. The effective reserve includes 0b from " +
        "the plugin pool-sizing UCX adjustment, not the full UCX allocation. CUDA context, " +
        "driver, and other future startup use are not modeled. The effective ASYNC allocator " +
        "is assumed to remain effective on the target CUDA runtime and driver. Its floor " +
        "quotient is 15, with caps for target task slots 16 (CPU cap 16 from 16 executor " +
        "cores and 1 CPU(s) per task; GPU cap 1000 from 0.001 GPU per task), 32 MiB permit " +
        "representation 491. This is a memory-admission seed, not a guarantee of observed " +
        "task concurrency or spill avoidance."
    result.footprintComments shouldBe Seq(expectedFootprintComment)

    val expectedComments = Seq(expectedFootprintComment) ++
      (if (autoTunerHelper.isPluginJarProbablyOutdated(pluginVersion)) {
        Seq(classPathComments("rapids.jars.outdated"))
      } else {
        Seq.empty
      }) ++ Seq(classPathComments("rapids.shuffle.jars"))
    val expectedOutput =
      s"""|
          |Spark Properties:
          |--conf $concurrentGpuTasksKey=15
          |
          |Comments:
          |${expectedComments.map(comment => s"- $comment").mkString("\n")}
          |""".stripMargin
    compareOutput(expectedOutput, result.renderedOutput)
  }

  test("current RMM profile is wired at the dynamic-admission minimum") {
    val result = runFootprintScenario(pluginVersion = "25.06.0")

    result.value(concurrentGpuTasksKey) shouldBe Some("3")
    result.commentText should include("discrete-gpu-rmm-v1 compatibility profile")
    result.commentText should include(GpuConcurrencySeed.FootprintMetricName)
  }

  test("ordinary batch change keeps batch recommendation and suppresses footprint seed") {
    val result = runFootprintScenario(sourceBatch = Some("1g"))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe Some("2147483647b")
    result.combinedValue(concurrentGpuTasksKey) shouldBe None
    result.combinedValue(batchSizeKey) shouldBe Some("2147483647b")
    result.commentText should include("recommended target configuration changes")
    result.commentText should include("from 1073741824b to 2147483647b")
    result.commentText should include("cross-batch footprint scaling is not modeled")
    result.commentText should include("Re-profile with 2147483647b")
  }

  test("one-byte ordinary batch change is not rounded into compatibility") {
    val result = runFootprintScenario(sourceBatch = Some("2g"))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.commentText should include("from 2147483648b to 2147483647b")
  }

  test("enforced batch change remains in output and reports its origin") {
    val result = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetEnforced = Map(batchSizeKey -> "2g"))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe Some("2g")
    result.combinedValue(concurrentGpuTasksKey) shouldBe None
    result.combinedValue(batchSizeKey) shouldBe Some("2g")
    result.commentText should include("enforced target configuration changes")
    result.commentText should include("from 1073741824b to 2147483648b")
  }

  test("preserved equal batch enables footprint recommendation") {
    val result = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetPreserve = List(batchSizeKey))

    result.value(concurrentGpuTasksKey) shouldBe Some("3")
    result.value(batchSizeKey) shouldBe Some("1g")
    result.combinedValue(concurrentGpuTasksKey) shouldBe Some("3")
    result.combinedValue(batchSizeKey) shouldBe Some("1g")
    result.commentText should not include "cross-batch footprint scaling"
  }

  test("unknown and malformed source batches suppress only footprint concurrency") {
    Seq(None, Some("garbage"), Some("1.5g"), Some("1Gi")).foreach { sourceBatch =>
      val result = runFootprintScenario(sourceBatch = sourceBatch)

      withClue(s"source batch $sourceBatch: ${result.commentText}") {
        result.value(concurrentGpuTasksKey) shouldBe None
        result.value(batchSizeKey) shouldBe Some("2147483647b")
        result.commentText should include("source or target")
        result.commentText should include("could not be resolved")
      }
    }
  }

  test("malformed source batch remains nonfatal with ColumnarExchange evidence") {
    val result = runFootprintScenario(
      sourceBatch = Some("garbage"),
      maxColumnarExchangeDataSizeBytes = Some(ByteUnit.GiB.toBytes(16L)))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe Some("2147483647b")
    result.commentText should include("source or target")
    result.commentText should include("could not be resolved")
  }

  test("ColumnarExchange bound uses the prospective target batch") {
    val result = runFootprintScenario(
      sourceBatch = Some("1g"),
      sourceExtra = Map("spark.sql.shuffle.partitions" -> "200"),
      maxColumnarExchangeDataSizeBytes = Some(ByteUnit.GiB.toBytes(600L)))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe Some("2147483647b")
    result.value("spark.sql.shuffle.partitions") shouldBe Some("301")
    result.commentText should include("GPU batch size (2147483647 bytes)")
  }

  test("ColumnarExchange bound uses the emitted batch under conflicting controls") {
    val preservedAndSkipped = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetPreserve = List(batchSizeKey),
      sourceExtra = Map("spark.sql.shuffle.partitions" -> "200"),
      skipList = Seq(batchSizeKey),
      maxColumnarExchangeDataSizeBytes = Some(ByteUnit.GiB.toBytes(600L)))
    preservedAndSkipped.value(concurrentGpuTasksKey) shouldBe None
    preservedAndSkipped.combinedValue(batchSizeKey) shouldBe Some("1g")
    preservedAndSkipped.value("spark.sql.shuffle.partitions") shouldBe Some("600")
    preservedAndSkipped.commentText should include("GPU batch size (1073741824 bytes)")

    val enforcedAndLimited = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetEnforced = Map(batchSizeKey -> "2g"),
      sourceExtra = Map("spark.sql.shuffle.partitions" -> "200"),
      limitedLogicList = Seq(batchSizeKey),
      maxColumnarExchangeDataSizeBytes = Some(ByteUnit.GiB.toBytes(600L)))
    enforcedAndLimited.value(concurrentGpuTasksKey) shouldBe None
    enforcedAndLimited.combinedValue(batchSizeKey) shouldBe Some("2g")
    enforcedAndLimited.value("spark.sql.shuffle.partitions") shouldBe Some("300")
    enforcedAndLimited.commentText should include("GPU batch size (2147483648 bytes)")
  }

  test("excluded target batch reports unknown compatibility") {
    val result = runFootprintScenario(
      sourceBatch = Some("2147483647"),
      targetExclude = List(batchSizeKey))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe None
    result.combinedValue(concurrentGpuTasksKey) shouldBe None
    result.combinedValue(batchSizeKey) shouldBe None
    result.commentText should include("source or target")
    result.commentText should include("could not be resolved")
  }

  test("no footprint evidence preserves silent modern-plugin suppression") {
    val result = runFootprintScenario(
      sourceExtra = Map(SparkTaskSlotModel.Properties.TaskGpuAmount -> "0.001"),
      evidence = Seq.empty,
      pluginVersion = "99.12.0",
      skipList = footprintOutputSkipList)

    result.value(concurrentGpuTasksKey) shouldBe None
    result.combinedValue(concurrentGpuTasksKey) shouldBe None
    result.footprintComments shouldBe empty
    result.commentText should not include concurrentGpuTasksKey
    result.commentText should not include GpuConcurrencySeed.FootprintMetricName

    val expectedOutput =
      s"""|Cannot recommend properties. See Comments.
          |
          |Comments:
          |- ${classPathComments("rapids.shuffle.jars")}
          |""".stripMargin
    compareOutput(expectedOutput, result.renderedOutput)
  }

  test("late modern suppression removes initialized concurrency from full output") {
    val scenarios = Seq(
      runFootprintScenario(
        sourceExtra = Map(concurrentGpuTasksKey -> "6"),
        evidence = Seq.empty,
        showOnlyUpdatedProps = false),
      runFootprintScenario(
        sourceBatch = Some("1g"),
        sourceExtra = Map(concurrentGpuTasksKey -> "6"),
        showOnlyUpdatedProps = false))

    scenarios.foreach { result =>
      result.value(concurrentGpuTasksKey) shouldBe None
      result.combinedValue(concurrentGpuTasksKey) shouldBe None
    }
  }

  test("unsupported nominal pool is actionable and never restores legacy formula") {
    val result = runFootprintScenario(
      sourceExtra = Map(RapidsRmmPoolModel.Properties.UvmEnabled -> "true"))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.commentText should include("nominal target RMM pool could not be modeled")
    result.commentText should include(RapidsRmmPoolModel.Properties.UvmEnabled)
    result.commentText should include("plugin runtime auto-tuning")
  }

  test("older plugin retains legacy recommendation even with footprint evidence") {
    val result = runFootprintScenario(
      pluginVersion = "25.04.0",
      evidence = Seq(footprintMetric(maxBytes = ByteUnit.GiB.toBytes(8L))))

    result.value(concurrentGpuTasksKey) shouldBe Some("3")
    result.commentText should not include GpuConcurrencySeed.FootprintMetricName
    result.commentText should not include s"'$concurrentGpuTasksKey' was not set."
  }

  test("internal runtime cap applies only when the plugin implements it") {
    val sourceExtra = Map(GpuBatchAndConcurrencyKeys.MaxConcurrentGpuTasks -> "2")
    val beforeCapability = runFootprintScenario(
      pluginVersion = "25.08.0", sourceExtra = sourceExtra)
    val withCapability = runFootprintScenario(
      pluginVersion = "25.10.0", sourceExtra = sourceExtra)

    beforeCapability.value(concurrentGpuTasksKey) shouldBe Some("3")
    withCapability.value(concurrentGpuTasksKey) shouldBe Some("2")
    withCapability.commentText should include("plugin maximum 2")
  }

  test("internal runtime cap distinguishes no limit from an invalid integer") {
    val key = GpuBatchAndConcurrencyKeys.MaxConcurrentGpuTasks
    Seq("0", "-1").foreach { rawValue =>
      val result = runFootprintScenario(
        pluginVersion = "25.10.0", sourceExtra = Map(key -> rawValue))
      withClue(s"no-limit value $rawValue: ${result.renderedOutput}") {
        result.value(concurrentGpuTasksKey) shouldBe Some("3")
        result.commentText should not include "task-count ceiling could not be resolved"
      }
    }

    Seq("garbage", "2147483648").foreach { rawValue =>
      val result = runFootprintScenario(
        pluginVersion = "25.10.0", sourceExtra = Map(key -> rawValue))
      withClue(s"invalid value $rawValue: ${result.renderedOutput}") {
        result.value(concurrentGpuTasksKey) shouldBe None
        result.commentText should include("plugin task-count ceiling could not be resolved")
        result.commentText should include("valid integer")
      }
    }
  }

  test("effective target CPU and GPU resources cap footprint concurrency") {
    val gpuLimited = runFootprintScenario(
      targetEnforced = Map(SparkTaskSlotModel.Properties.TaskGpuAmount -> "0.25"),
      evidence = Seq(footprintMetric(maxBytes = ByteUnit.GiB.toBytes(1L))))
    gpuLimited.value(concurrentGpuTasksKey) shouldBe Some("4")
    gpuLimited.commentText should include("GPU cap 4 from 0.25 GPU per task")

    val cpuLimited = runFootprintScenario(
      targetEnforced = Map(
        SparkTaskSlotModel.Properties.ExecutorCores -> "8",
        SparkTaskSlotModel.Properties.TaskCpus -> "2"),
      evidence = Seq(footprintMetric(maxBytes = ByteUnit.GiB.toBytes(1L))))
    cpuLimited.value(concurrentGpuTasksKey) shouldBe Some("4")
    cpuLimited.commentText should include("CPU cap 4 from 8 executor cores")
  }

  test("unsupported target task resources fail closed with an actionable comment") {
    val scenarios = Seq(
      ("unsupported fractional task GPU", runFootprintScenario(
        targetEnforced = Map(SparkTaskSlotModel.Properties.TaskGpuAmount -> "0.75")),
        "spark.task.resource.gpu.amount=0.75"),
      ("multiple executor GPUs", runFootprintScenario(
        targetEnforced = Map(SparkTaskSlotModel.Properties.ExecutorGpuAmount -> "2")),
        "spark.executor.resource.gpu.amount=2"),
      ("decimal executor GPU", runFootprintScenario(
        targetEnforced = Map(SparkTaskSlotModel.Properties.ExecutorGpuAmount -> "1.0")),
        "spark.executor.resource.gpu.amount' could not be resolved to a positive integer"),
      ("space-padded executor GPU", runFootprintScenario(
        targetEnforced = Map(SparkTaskSlotModel.Properties.ExecutorGpuAmount -> " 1 ")),
        "spark.executor.resource.gpu.amount' could not be resolved to a positive integer"),
      ("missing executor cores", runFootprintScenario(
        skipList = Seq(SparkTaskSlotModel.Properties.ExecutorCores)),
        "spark.executor.cores' could not be resolved to a positive integer"))

    scenarios.foreach { case (label, result, expectedReason) =>
      withClue(s"$label: ${result.renderedOutput}") {
        result.value(concurrentGpuTasksKey) shouldBe None
        result.combinedValue(concurrentGpuTasksKey) shouldBe None
        result.commentText should include(
          "effective target task slots per GPU could not be modeled")
        result.commentText should include(expectedReason)
      }
    }
  }

  test("explicit operator tuning maximum caps footprint evidence") {
    val tuningConfigs = ToolTestUtils.buildTuningConfigs(default = List(
      TuningConfigEntry(name = "CONC_GPU_TASKS", max = "2")))
    val result = runFootprintScenario(tuningConfigs = Some(tuningConfigs))

    result.value(concurrentGpuTasksKey) shouldBe Some("2")
    result.commentText should include("operator tuning maximum 2")
  }

  test("invalid explicit operator tuning maximum suppresses footprint evidence") {
    Seq("0", "-1", "malformed", "9223372036854775808").foreach { rawMax =>
      val tuningConfigs = ToolTestUtils.buildTuningConfigs(default = List(
        TuningConfigEntry(name = "CONC_GPU_TASKS", max = rawMax)))
      val result = runFootprintScenario(tuningConfigs = Some(tuningConfigs))

      withClue(s"operator maximum $rawMax: ${result.commentText}") {
        result.value(concurrentGpuTasksKey) shouldBe None
        result.combinedValue(concurrentGpuTasksKey) shouldBe None
        result.commentText should include(s"CONC_GPU_TASKS.max=$rawMax")
        result.commentText should include("not a positive integer")
      }
    }
  }

  test("footprint larger than nominal pool emits seed one with warning") {
    val result = runFootprintScenario(
      evidence = Seq(footprintMetric(maxBytes = ByteUnit.GiB.toBytes(20L))))

    result.value(concurrentGpuTasksKey) shouldBe Some("1")
    result.commentText should include("WARN")
    result.commentText should include("smaller than the historical single-task footprint")
  }

  test("target concurrency override bypasses footprint compatibility checks") {
    val result = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetEnforced = Map(concurrentGpuTasksKey -> "5"))

    result.value(concurrentGpuTasksKey) shouldBe Some("5")
    result.commentText should not include GpuConcurrencySeed.FootprintMetricName
  }

  test("byte-equal enforced batch enables footprint recommendation") {
    val result = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetEnforced = Map(batchSizeKey -> "1024m"))

    result.value(concurrentGpuTasksKey) shouldBe Some("3")
    result.combinedValue(concurrentGpuTasksKey) shouldBe Some("3")
    result.commentText should not include "cross-batch footprint scaling"
  }

  test("preserved batch without a source retains ordinary batch and reports unknown") {
    val result = runFootprintScenario(
      sourceBatch = None,
      targetPreserve = List(batchSizeKey))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe Some("2147483647b")
    result.combinedValue(batchSizeKey) shouldBe Some("2147483647b")
    result.commentText should include("specified in preserve list but not found")
    result.commentText should include("source or target")
    result.commentText should include("could not be resolved")
  }

  test("caller-limited batch requires a source value") {
    val withSource = runFootprintScenario(
      limitedLogicList = Seq(batchSizeKey))

    withSource.value(concurrentGpuTasksKey) shouldBe Some("3")
    withSource.combinedValue(concurrentGpuTasksKey) shouldBe Some("3")
    withSource.combinedValue(batchSizeKey) shouldBe Some("2147483647")
    withSource.commentText should not include "could not be resolved"

    val withoutSource = runFootprintScenario(
      sourceBatch = None,
      limitedLogicList = Seq(batchSizeKey))

    withoutSource.value(concurrentGpuTasksKey) shouldBe None
    withoutSource.value(batchSizeKey) shouldBe None
    withoutSource.combinedValue(batchSizeKey) shouldBe None
    withoutSource.commentText should include("could not be resolved")
  }

  test("caller-skipped batch is unknown and removed from combined output") {
    val result = runFootprintScenario(skipList = Seq(batchSizeKey))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe None
    result.combinedValue(batchSizeKey) shouldBe None
    result.commentText should include("source or target")
    result.commentText should include("could not be resolved")
  }

  test("concurrency output controls bypass evidence and preserve final-output semantics") {
    val preserved = runFootprintScenario(
      sourceExtra = Map(concurrentGpuTasksKey -> "6"),
      targetPreserve = List(concurrentGpuTasksKey))
    preserved.value(concurrentGpuTasksKey) shouldBe Some("6")
    preserved.combinedValue(concurrentGpuTasksKey) shouldBe Some("6")
    preserved.commentText should not include GpuConcurrencySeed.FootprintMetricName

    val preservedWithoutSource = runFootprintScenario(
      targetPreserve = List(concurrentGpuTasksKey))
    preservedWithoutSource.value(concurrentGpuTasksKey) shouldBe Some("3")
    preservedWithoutSource.combinedValue(concurrentGpuTasksKey) shouldBe Some("3")
    preservedWithoutSource.commentText should include(
      s"'$concurrentGpuTasksKey' was specified in preserve list but not found in source " +
        "properties.\n  AutoTuner will continue with its recommendation for this property.")
    preservedWithoutSource.commentText should not include GpuConcurrencySeed.FootprintMetricName

    val removedScenarios = Seq(
      "target exclusion" -> runFootprintScenario(
        sourceExtra = Map(concurrentGpuTasksKey -> "6"),
        targetExclude = List(concurrentGpuTasksKey)),
      "caller skip" -> runFootprintScenario(
        sourceExtra = Map(concurrentGpuTasksKey -> "6"),
        skipList = Seq(concurrentGpuTasksKey)),
      "caller limit" -> runFootprintScenario(
        sourceExtra = Map(concurrentGpuTasksKey -> "6"),
        limitedLogicList = Seq(concurrentGpuTasksKey)))

    removedScenarios.foreach { case (label, result) =>
      withClue(s"$label: ${result.commentText}") {
        result.value(concurrentGpuTasksKey) shouldBe None
        result.combinedValue(concurrentGpuTasksKey) shouldBe None
        result.commentText should not include GpuConcurrencySeed.FootprintMetricName
      }
    }
  }

  test("equal enforced batch stays compatible under conflicting caller controls") {
    val scenarios = Seq(
      ("caller skip", runFootprintScenario(
          sourceBatch = Some("1g"),
          targetEnforced = Map(batchSizeKey -> "1024m"),
          skipList = Seq(batchSizeKey)), "1024m"),
      ("caller limit", runFootprintScenario(
          sourceBatch = Some("1g"),
          targetEnforced = Map(batchSizeKey -> "1024m"),
          limitedLogicList = Seq(batchSizeKey)), "1g"))

    scenarios.foreach { case (label, result, renderedBatch) =>
      withClue(s"$label: ${result.renderedOutput}") {
        result.value(concurrentGpuTasksKey) shouldBe Some("3")
        result.combinedValue(concurrentGpuTasksKey) shouldBe Some("3")
        result.combinedValue(batchSizeKey) shouldBe Some(renderedBatch)
        result.combinedValue(batchSizeKey)
          .flatMap(RuntimeConfigParser.parsePositiveBytes) shouldBe
          Some(RuntimeConfigParser.ParsedBytes(ByteUnit.GiB.toBytes(1L)))
        result.footprintComments.count(_.contains("was seeded at 3")) shouldBe 1
      }
    }
  }

  test("caller-skipped preserved batch suppresses footprint seeding conservatively") {
    val result = runFootprintScenario(
      sourceBatch = Some("1g"),
      targetPreserve = List(batchSizeKey),
      skipList = Seq(batchSizeKey))

    result.value(concurrentGpuTasksKey) shouldBe None
    result.combinedValue(concurrentGpuTasksKey) shouldBe None
    result.value(batchSizeKey) shouldBe Some("1g")
    result.combinedValue(batchSizeKey) shouldBe Some("1g")
    result.footprintComments should have size 1
    result.footprintComments.head should include("source or target")
    result.footprintComments.head should include("could not be resolved")
    result.footprintComments.head should not include "was seeded at"
  }

  test("target-enforced concurrency wins conflicting caller controls without evidence reuse") {
    val scenarios = Seq(
      "caller skip" -> runFootprintScenario(
        targetEnforced = Map(concurrentGpuTasksKey -> "5"),
        skipList = Seq(concurrentGpuTasksKey)),
      "caller limit" -> runFootprintScenario(
        targetEnforced = Map(concurrentGpuTasksKey -> "5"),
        limitedLogicList = Seq(concurrentGpuTasksKey)))

    scenarios.foreach { case (label, result) =>
      withClue(s"$label: ${result.renderedOutput}") {
        result.value(concurrentGpuTasksKey) shouldBe Some("5")
        result.combinedValue(concurrentGpuTasksKey) shouldBe Some("5")
        result.footprintComments shouldBe empty
      }
    }
  }

  test("unresolved plugin versions retain legacy behavior despite footprint evidence") {
    val jarScenarios = Seq(
      "missing" -> Seq.empty[String],
      "multiple" -> Seq(
        "rapids-4-spark_2.12-25.08.0.jar",
        "rapids-4-spark_2.12-25.10.0.jar"),
      "malformed" -> Seq("rapids-4-spark_2.12-not-a-version.jar"))

    jarScenarios.foreach { case (label, jars) =>
      val result = runFootprintScenario(rapidsJars = Some(jars))
      withClue(s"$label: ${result.commentText}") {
        result.value(concurrentGpuTasksKey) shouldBe Some("3")
        result.combinedValue(concurrentGpuTasksKey) shouldBe Some("3")
        result.commentText should not include GpuConcurrencySeed.FootprintMetricName
        result.commentText should not include s"'$concurrentGpuTasksKey' was not set."
      }
    }
  }

  test("conflicting skip and preserve controls fail target models closed") {
    import RapidsRmmPoolModel.{Properties => RmmProperties}

    val scenarios = Seq(
      ("RMM fraction", RmmProperties.AllocationFraction, "0.5", "nominal target RMM pool"),
      ("exact allocation", RmmProperties.ExactAllocation, "1g", "nominal target RMM pool"),
      ("task CPUs", SparkTaskSlotModel.Properties.TaskCpus, "2",
        "effective target task slots per GPU"),
      ("plugin count cap", GpuBatchAndConcurrencyKeys.MaxConcurrentGpuTasks, "2",
        "plugin task-count ceiling"))

    scenarios.foreach { case (label, property, sourceValue, expectedModel) =>
      val result = runFootprintScenario(
        pluginVersion = "25.10.0",
        sourceExtra = Map(property -> sourceValue),
        targetPreserve = List(property),
        skipList = Seq(property))

      withClue(s"$label: ${result.renderedOutput}") {
        result.value(concurrentGpuTasksKey) shouldBe None
        result.combinedValue(concurrentGpuTasksKey) shouldBe None
        result.combinedValue(property) shouldBe Some(sourceValue)
        result.commentText should include(expectedModel)
        result.commentText should include("conflicting target controls")
        result.commentText should include("both skipped and preserved or caller-limited")
      }
    }
  }

  test("target RMM overrides are resolved into distinct nominal pool layers") {
    import RapidsRmmPoolModel.{Properties => RmmProperties}

    val targetRmmOverrides = Map(
      RmmProperties.AllocationFraction -> "0.5",
      RmmProperties.MaximumAllocationFraction -> "0.75",
      RmmProperties.MinimumAllocationFraction -> "0.1",
      RmmProperties.BaseReserve -> "1g",
      RmmProperties.RmmAllocatorMode -> RapidsRmmPoolModel.RmmAllocatorModes.Async,
      RmmProperties.ShuffleMode -> RapidsRmmPoolModel.ShuffleModes.Ucx,
      RmmProperties.UcxBounceBufferSize -> "64m",
      RmmProperties.ChunkedPackPoolSize -> "512m")
    val result = runFootprintScenario(targetEnforced = targetRmmOverrides)

    result.value(concurrentGpuTasksKey) shouldBe Some("1")
    result.combinedValue(concurrentGpuTasksKey) shouldBe Some("1")
    targetRmmOverrides.foreach { case (key, value) =>
      withClue(key) {
        result.combinedValue(key) shouldBe Some(value)
      }
    }
    result.footprintComments should have size 1
    val footprintComment = result.footprintComments.head
    Seq(
      "nominal target RMM pool is 7717519360b",
      "subtracts the 536870912b chunked-pack allocation",
      "leaves 16642998272b nominal free",
      "1073741824b base reserve",
      "effective reserve is 1207959552b",
      "ASYNC allocator with UCX shuffle",
      "alloc/min/max fractions 0.5/0.1/0.75",
      "effective reserve includes 134217728b").foreach { expected =>
      withClue(footprintComment) {
        footprintComment should include(expected)
      }
    }
  }

  test("pool integration fails closed and reports target capacity source") {
    val exactAllocation = runFootprintScenario(
      sourceExtra = Map(RapidsRmmPoolModel.Properties.ExactAllocation -> "1g"))
    exactAllocation.value(concurrentGpuTasksKey) shouldBe None
    exactAllocation.commentText should include("exact allocation is not modeled")

    val malformedTargetMemory = runFootprintScenario(targetGpuMemory = Some("1.5g"))
    malformedTargetMemory.value(concurrentGpuTasksKey) shouldBe None
    malformedTargetMemory.commentText should include("target GPU device capacity")
    malformedTargetMemory.commentText should include("could not be resolved")

    val catalogMemory = runFootprintScenario(targetGpuMemory = None)
    catalogMemory.value(concurrentGpuTasksKey) shouldBe Some("5")
    catalogMemory.combinedValue(concurrentGpuTasksKey) shouldBe Some("5")
    catalogMemory.commentText should include("device catalog as both CUDA total")
  }

  test("footprint tuning neither requires spill nor suppresses independent shuffle tuning") {
    val withoutSpill = runFootprintScenario()
    withoutSpill.value(concurrentGpuTasksKey) shouldBe Some("3")

    val mismatchWithSpill = runFootprintScenario(
      sourceBatch = Some("1g"),
      sourceExtra = Map("spark.sql.shuffle.partitions" -> "200"),
      shuffleStagesWithPosSpilling = Set(3L))

    mismatchWithSpill.value(concurrentGpuTasksKey) shouldBe None
    mismatchWithSpill.combinedValue(concurrentGpuTasksKey) shouldBe None
    mismatchWithSpill.value("spark.sql.shuffle.partitions") shouldBe Some("400")
    mismatchWithSpill.combinedValue("spark.sql.shuffle.partitions") shouldBe Some("400")
    mismatchWithSpill.commentText should include(
      "Shuffle partitions should be increased since spilling occurred in shuffle stages.")
    mismatchWithSpill.commentText should include("cross-batch footprint scaling is not modeled")
  }

}
