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

import com.nvidia.spark.rapids.tool._
import com.nvidia.spark.rapids.tool.profiling.BaseDriverLogInfoProvider
import com.nvidia.spark.rapids.tool.tuning.config.{TuningConfigEntry, TuningConfiguration}
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.sql.rapids.tool.RecommendedClusterInfo
import org.apache.spark.sql.rapids.tool.util.PropertiesLoader

class ClusterSizingStrategySuite extends BaseAutoTunerSuite {
  override val autoTunerHelper: AutoTunerHelper = QualificationAutoTunerHelper

  private val autoTunerHelpers = Seq(
    "qualification" -> QualificationAutoTunerHelper,
    "profiling" -> ProfilingAutoTunerHelper)
  private val strategies = Seq(
    SourceCoresPreservingStrategy, ConstantTotalCoresStrategy, ConstantGpuCountStrategy)

  // The code should return the 3 strategy names users can put in the tuning config file.
  // These names must not include the "$" that Scala adds at the end.
  test("strategy names match the documented tuning config values") {
    strategies.map(_.getNameAsString) shouldBe Seq(
      "SourceCoresPreservingStrategy", "ConstantTotalCoresStrategy", "ConstantGpuCountStrategy")
  }

  private val platformCases = Seq(
    (PlatformNames.ONPREM, false),
    (PlatformNames.ONPREM, true),
    (PlatformNames.EMR, false))

  autoTunerHelpers.foreach { case (toolName, tunerHelper) =>
    platformCases.foreach { case (platformName, withWorker) =>
      // With auto, onprem without target cluster worker details should keep the original
      // cores and executors.
      // Otherwise qualification should keep the total cores, and profiling should keep GPU count.
      // Spark settings alone in a target cluster file do not provide target cluster worker details.
      test(s"$toolName auto sizing on $platformName with target worker=$withWorker") {
        val target = if (withWorker) {
          ToolTestUtils.buildTargetClusterInfo(cpuCores = Some(32), memoryGB = Some(128L),
            gpuCount = Some(1), gpuDevice = Some(GpuTypes.T4))
        } else {
          ToolTestUtils.buildTargetClusterInfo(
            enforcedSparkProperties = Map("spark.executor.memory" -> "32g"))
        }
        val platform = PlatformFactory.createInstance(platformName, Some(target))
        platform.getTargetWorkerInfo.isDefined shouldBe withWorker
        val expected = if (platformName == PlatformNames.ONPREM && !withWorker) {
          SourceCoresPreservingStrategy
        } else if (toolName == "qualification") {
          ConstantTotalCoresStrategy
        } else {
          ConstantGpuCountStrategy
        }
        tunerHelper.recommendedClusterSizingStrategy(platform) shouldBe expected
        tunerHelper.recommendedClusterSizingStrategy(platform, "auto") shouldBe expected
      }
    }
  }

  private def recommend(
      tunerHelper: AutoTunerHelper,
      config: Option[TuningConfiguration],
      target: Option[TargetClusterProps] = None): (Int, Int) = {
    val props = mutable.Map(
      "spark.master" -> "yarn",
      "spark.executor.cores" -> "2",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "16g")
    val info = getMockInfoProvider(0, Seq(0), Seq(0.0), props, Some(testSparkVersion))
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM, target)
    configureEventLogClusterInfoForTest(platform, numCores = 2, numWorkers = 2,
      sparkProperties = props.toMap)
    val tuner = tunerHelper.createAutoTunerInstance(info, platform,
      BaseDriverLogInfoProvider.noneDriverLog, config)
    tuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    val cluster = platform.recommendedClusterInfo.get
    (cluster.numExecutors, cluster.coresPerExecutor)
  }

  autoTunerHelpers.foreach { case (toolName, tunerHelper) =>
    // With no tuning config file or target cluster worker details, both tools should keep
    // the original setup:
    // 2 executors with 2 cores each.
    test(s"$toolName preserves two executors with two cores by default") {
      recommend(tunerHelper, None) shouldBe ((2, 2))
    }

    val strategyCases = Seq(
      ConstantTotalCoresStrategy.getNameAsString -> ((1, 4)),
      ConstantGpuCountStrategy.getNameAsString -> ((2, 4)))
    strategyCases.foreach { case (strategy, expected) =>
      // Both tools should use the choice in the default section of the tuning config file.
      // Keeping total cores should give 1 executor with 4 cores. Keeping GPU count should give
      // 2 executors with 4 cores each. Both results differ from auto's 2 executors with 2 cores.
      test(s"$toolName uses configured $strategy from the shared tuning config") {
        val defaultConfig = PropertiesLoader[TuningConfiguration].loadFromContent(
          s"""default:
             |  - name: CLUSTER_SIZING_STRATEGY
             |    default: $strategy
             |""".stripMargin)
        recommend(tunerHelper, defaultConfig) shouldBe expected
      }
    }

    val overrideCases = Seq(
      ConstantTotalCoresStrategy.getNameAsString -> ((1, 4)), "auto" -> ((2, 2)))
    overrideCases.foreach { case (strategy, expected) =>
      // A choice for 1 tool should win over a choice shared by both tools.
      // Keeping total cores should give 1 executor with 4 cores. Choosing auto should give
      // 2 executors with 2 cores each.
      test(s"$toolName override $strategy takes precedence over the shared default") {
        val config = PropertiesLoader[TuningConfiguration].loadFromContent(
          s"""default:
             |  - name: CLUSTER_SIZING_STRATEGY
             |    default: ${ConstantGpuCountStrategy.getNameAsString}
             |$toolName:
             |  - name: CLUSTER_SIZING_STRATEGY
             |    default: $strategy
             |""".stripMargin)
        recommend(tunerHelper, config) shouldBe expected
      }
    }

    // Values the user requires should win over the chosen strategy: 5 executors with 3
    // cores each. If the user asks to keep the original values, expect 2 executors with 2
    // cores each.
    test(s"$toolName sizing strategies retain enforced and preserved Spark properties") {
      Seq(ConstantTotalCoresStrategy, ConstantGpuCountStrategy).foreach { strategy =>
        val config = PropertiesLoader[TuningConfiguration].loadFromContent(
          s"""default:
             |  - name: CLUSTER_SIZING_STRATEGY
             |    default: ${strategy.getNameAsString}
             |""".stripMargin)
        val enforced = ToolTestUtils.buildTargetClusterInfo(enforcedSparkProperties = Map(
          "spark.executor.cores" -> "3", "spark.executor.instances" -> "5"))
        recommend(tunerHelper, config, Some(enforced)) shouldBe ((5, 3))
        val preserved = ToolTestUtils.buildTargetClusterInfo(preserveSparkProperties =
          List("spark.executor.cores", "spark.executor.instances"))
        recommend(tunerHelper, config, Some(preserved)) shouldBe ((2, 2))
      }
    }
  }

  autoTunerHelpers.foreach { case (toolName, tunerHelper) =>
    Seq(PlatformNames.EMR, PlatformNames.ONPREM).foreach { platformName =>
      def recommendSourceSizing(
          sourceExecutors: Int = 2,
          sourceCores: Int = 2,
          enforced: Map[String, String] = Map.empty,
          preserved: List[String] = List.empty,
          eventLogExecutors: Option[Int] = None,
          dynamicAllocationEnabled: Boolean = false):
          (RecommendedClusterInfo, Map[String, String], Seq[String]) = {
        val config = PropertiesLoader[TuningConfiguration].loadFromContent(
          s"""default:
             |  - name: CLUSTER_SIZING_STRATEGY
             |    default: ${SourceCoresPreservingStrategy.getNameAsString}
             |""".stripMargin)
        val props = mutable.Map(
          "spark.master" -> "yarn",
          "spark.executor.cores" -> sourceCores.toString,
          "spark.executor.instances" -> sourceExecutors.toString,
          "spark.executor.memory" -> "16g")
        if (dynamicAllocationEnabled) {
          props("spark.dynamicAllocation.enabled") = "true"
        }
        val target = if (platformName == PlatformNames.ONPREM) {
          ToolTestUtils.buildTargetClusterInfo(
            cpuCores = Some(16), memoryGB = Some(64L),
            gpuCount = Some(2), gpuDevice = Some(GpuTypes.L4),
            enforcedSparkProperties = enforced, preserveSparkProperties = preserved)
        } else {
          ToolTestUtils.buildTargetClusterInfo(
            enforcedSparkProperties = enforced, preserveSparkProperties = preserved)
        }
        val platform = PlatformFactory.createInstance(platformName, Some(target))
        platform.recommendedWorkerNode.isDefined shouldBe true
        val info = getMockInfoProvider(0, Seq(0), Seq(0.0), props, Some(testSparkVersion))
        configureEventLogClusterInfoForTest(platform, numCores = sourceCores,
          numWorkers = eventLogExecutors.getOrElse(sourceExecutors),
          sparkProperties = props.toMap)
        val tuner = tunerHelper.createAutoTunerInstance(info, platform,
          BaseDriverLogInfoProvider.noneDriverLog, config)
        val (recommendations, comments) =
          tuner.getRecommendedProperties(showOnlyUpdatedProps = false)
        val properties = recommendations.map(p => p.name -> p.getTuneValue()).toMap
        (platform.recommendedClusterInfo.get, properties, comments.map(_.comment))
      }

      // A larger target worker should not change the original 2 cores per executor.
      // Keep all 18 executors, with matching worker counts and output Spark settings.
      test(s"$toolName keeps 18 executors with 2 cores on $platformName") {
        val (cluster, properties, _) = recommendSourceSizing(sourceExecutors = 18)
        cluster.coresPerExecutor shouldBe 2
        cluster.numExecutors shouldBe 18
        cluster.numWorkerNodes shouldBe (if (platformName == PlatformNames.ONPREM) -1 else 18)
        properties("spark.executor.cores") shouldBe "2"
        properties("spark.executor.instances") shouldBe "18"
      }

      // Values the user requires should win over keeping the original setup.
      // Expect 5 executors with 3 cores each in both the results and output Spark settings.
      test(s"$toolName uses required cores and executor count on $platformName") {
        val (cluster, properties, _) = recommendSourceSizing(enforced = Map(
          "spark.executor.cores" -> "3", "spark.executor.instances" -> "5"))
        cluster.coresPerExecutor shouldBe 3
        cluster.numExecutors shouldBe 5
        cluster.numWorkerNodes shouldBe (if (platformName == PlatformNames.ONPREM) -1 else 5)
        properties("spark.executor.cores") shouldBe "3"
        properties("spark.executor.instances") shouldBe "5"
      }

      if (platformName == PlatformNames.EMR) {
        // The original 32 cores per executor will not fit on the target worker.
        // Keep 2 executors and use 16 cores each on EMR.
        test(s"$toolName limits source cores to what fits on $platformName") {
          val (cluster, properties, _) = recommendSourceSizing(sourceCores = 32)
          cluster.coresPerExecutor shouldBe 16
          cluster.numExecutors shouldBe 2
          cluster.numWorkerNodes shouldBe 2
          properties("spark.executor.cores") shouldBe "16"
          properties("spark.executor.instances") shouldBe "2"
        }
      }

      // With dynamic allocation, the observed count can differ from the source setting.
      // Preserving spark.executor.instances must use the source value, not the observed count.
      test(s"$toolName preserves source executor count over observed count on $platformName") {
        val (cluster, properties, _) = recommendSourceSizing(sourceExecutors = 18,
          preserved = List("spark.executor.instances"), eventLogExecutors = Some(2),
          dynamicAllocationEnabled = true)
        cluster.numExecutors shouldBe 18
        cluster.numWorkerNodes shouldBe (if (platformName == PlatformNames.ONPREM) -1 else 18)
        properties("spark.executor.instances") shouldBe "18"
      }

      if (platformName == PlatformNames.ONPREM) {
        test(s"$toolName fits both source executors on a 16-core, 2-GPU worker") {
          val (cluster, properties, _) = recommendSourceSizing(sourceCores = 12)
          cluster.coresPerExecutor shouldBe 8
          cluster.numExecutors shouldBe 2
          cluster.numWorkerNodes shouldBe -1
          properties("spark.executor.cores") shouldBe "8"
          properties("spark.executor.instances") shouldBe "2"
        }

        test(s"$toolName reports user-overridden cores even above per-GPU capacity") {
          val (enforcedCluster, enforcedProps, enforcedComments) = recommendSourceSizing(enforced =
            Map("spark.executor.cores" -> "12"))
          enforcedCluster.coresPerExecutor shouldBe 12
          enforcedProps("spark.executor.cores") shouldBe "12"
          enforcedComments.exists(_.contains("only 8 CPU core(s) per GPU")) shouldBe true

          val (preservedCluster, preservedProps, preservedComments) = recommendSourceSizing(
            sourceCores = 12, preserved = List("spark.executor.cores"))
          preservedCluster.coresPerExecutor shouldBe 12
          preservedProps("spark.executor.cores") shouldBe "12"
          preservedComments.exists(_.contains("only 8 CPU core(s) per GPU")) shouldBe true
        }
      }
    }
  }

  test("reject target workers with more GPUs than CPU cores") {
    val error = intercept[IllegalArgumentException] {
      ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(2), memoryGB = Some(64L),
        gpuCount = Some(4), gpuDevice = Some(GpuTypes.L4))
    }
    error.getMessage should include ("2 CPU core(s) for 4 GPU(s)")
  }

  test("reject catalog workers that cannot run one executor per GPU") {
    val target = ToolTestUtils.buildTargetClusterInfo(
      workerNodeInstanceType = Some("n1-standard-1"), gpuCount = Some(4))
    val error = intercept[IllegalArgumentException] {
      PlatformFactory.createInstance(PlatformNames.DATAPROC, Some(target))
    }
    error.getMessage should include ("1 CPU core(s) for 4 GPU(s)")
  }

  test("the strategy core fallback stays positive for an unresolved worker shape") {
    val platform = PlatformFactory.createInstance(PlatformNames.ONPREM)
    platform.recommendedWorkerNode = Some(InstanceInfo(
      cores = 1, memoryMB = 1024L, name = "unresolved", numGpus = 4, gpuDevice = L4Gpu))
    platform.recommendedCoresPerExec shouldBe 1
  }

  private def sizingComments(
      tunerHelper: AutoTunerHelper,
      platform: Platform,
      strategy: String): Seq[String] = {
    val props = mutable.Map(
      "spark.master" -> "yarn",
      "spark.executor.cores" -> "2",
      "spark.executor.instances" -> "2",
      "spark.executor.memory" -> "16g")
    val info = getMockInfoProvider(0, Seq(0), Seq(0.0), props, Some(testSparkVersion))
    configureEventLogClusterInfoForTest(platform, numCores = 2, numWorkers = 2,
      sparkProperties = props.toMap)
    val config = ToolTestUtils.buildTuningConfigs(default = List(
      TuningConfigEntry(name = "CLUSTER_SIZING_STRATEGY", default = strategy)))
    val tuner = tunerHelper.createAutoTunerInstance(info, platform,
      BaseDriverLogInfoProvider.noneDriverLog, Some(config))
    val (properties, comments) = tuner.getRecommendedProperties(showOnlyUpdatedProps = false)
    properties should not be empty
    comments.map(_.comment)
  }

  autoTunerHelpers.foreach { case (toolName, tunerHelper) =>
    // Without target cluster worker details, keeping total cores should add the memory warning.
    // Tuning should still return Spark settings.
    test(s"$toolName warns about total-core sizing on onprem without a target cluster") {
      val comments = sizingComments(tunerHelper,
        PlatformFactory.createInstance(PlatformNames.ONPREM),
        ConstantTotalCoresStrategy.getNameAsString)
      comments.count(_ == missingTargetWorkerInfoForConstantCoresComment) shouldBe 1
    }

    // Spark settings in a target cluster file do not provide target cluster worker details.
    // Keeping total cores should still add the memory warning and return Spark settings.
    test(s"$toolName warns when the target cluster only supplies Spark settings") {
      val target = ToolTestUtils.buildTargetClusterInfo(
        enforcedSparkProperties = Map("spark.sql.shuffle.partitions" -> "200"))
      val comments = sizingComments(tunerHelper,
        PlatformFactory.createInstance(PlatformNames.ONPREM, Some(target)),
        ConstantTotalCoresStrategy.getNameAsString)
      comments should contain (missingTargetWorkerInfoForConstantCoresComment)
    }

    // A target worker without a memory value cannot give a reliable memory estimate.
    // Keeping total cores should add the memory warning and still return Spark settings.
    test(s"$toolName warns when target cluster worker memory is missing") {
      val target = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(4), gpuCount = Some(1), gpuDevice = Some(GpuTypes.L4))
      val comments = sizingComments(tunerHelper,
        PlatformFactory.createInstance(PlatformNames.ONPREM, Some(target)),
        ConstantTotalCoresStrategy.getNameAsString)
      comments should contain (missingTargetWorkerInfoForConstantCoresComment)
    }

    // Supplying target cluster worker cores, memory, and GPU details should avoid the warning.
    test(s"$toolName does not warn when onprem target cluster worker details are supplied") {
      val target = ToolTestUtils.buildTargetClusterInfo(
        cpuCores = Some(4), memoryGB = Some(64L), gpuCount = Some(1), gpuDevice = Some(GpuTypes.L4))
      val comments = sizingComments(tunerHelper,
        PlatformFactory.createInstance(PlatformNames.ONPREM, Some(target)),
        ConstantTotalCoresStrategy.getNameAsString)
      comments should not contain missingTargetWorkerInfoForConstantCoresComment
    }

    // EMR supplies a default target worker with known memory, so it should avoid the warning.
    test(s"$toolName does not warn for total-core sizing with the EMR default worker") {
      val comments = sizingComments(tunerHelper, PlatformFactory.createInstance(PlatformNames.EMR),
        ConstantTotalCoresStrategy.getNameAsString)
      comments should not contain missingTargetWorkerInfoForConstantCoresComment
    }

    val otherChoices = Seq("auto", SourceCoresPreservingStrategy.getNameAsString,
      ConstantGpuCountStrategy.getNameAsString)
    otherChoices.foreach { strategy =>
      // The new warning is only for keeping total cores, so these choices should avoid it.
      test(s"$toolName does not add the total-core memory warning for $strategy") {
        val comments = sizingComments(tunerHelper,
          PlatformFactory.createInstance(PlatformNames.ONPREM), strategy)
        comments should not contain missingTargetWorkerInfoForConstantCoresComment
      }
    }
  }

  private val invalidCases = Seq(
    "default" -> "unknown", "qualification" -> "unknown", "profiling" -> "unknown",
    "default" -> "AUTO", "default" -> "constanttotalcoresstrategy")

  invalidCases.foreach { case (section, invalid) =>
    // Unknown names and names with the wrong capital letters should cause an error.
    // The error should show the bad value and list the choices the user can use.
    test(s"reject invalid sizing strategy '$invalid' in $section") {
      val error = intercept[IllegalArgumentException] {
        PropertiesLoader[TuningConfiguration].loadFromContent(
          s"""$section:
             |  - name: CLUSTER_SIZING_STRATEGY
             |    default: $invalid
             |""".stripMargin)
      }
      error.getMessage should include (s"Invalid CLUSTER_SIZING_STRATEGY '$invalid'")
      strategies.foreach(strategy => error.getMessage should include (strategy.getNameAsString))
      error.getMessage should include ("auto")
    }
  }
}
