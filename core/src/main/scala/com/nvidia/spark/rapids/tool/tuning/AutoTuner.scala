/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

import scala.beans.BeanProperty
import scala.collection.mutable
import scala.util.control.NonFatal
import scala.util.matching.Regex

import com.nvidia.spark.rapids.tool._
import com.nvidia.spark.rapids.tool.profiling._
import com.nvidia.spark.rapids.tool.tuning.config.{CategoryEnum, ProfTuningConfigProvider, TuningConfigProvider, TuningConfiguration, TuningEntryDefinition}
import com.nvidia.spark.rapids.tool.tuning.plugins.TuningPluginManager
import org.yaml.snakeyaml.constructor.ConstructorException

import org.apache.spark.internal.Logging
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.ToolUtils
import org.apache.spark.sql.rapids.tool.util.{StringUtils, ValidatableProperties, WebCrawlerUtil}

/**
 * A wrapper class that stores all the GPU properties.
 * The BeanProperty enables loading and parsing the YAML formatted content using the
 * Constructor SnakeYaml approach.
 */
class GpuWorkerProps(
    @BeanProperty var memory: String,
    @BeanProperty var count: Int,
    private var name: String) extends ValidatableProperties {

  var device: Option[GpuDevice] = None

  def this() = {
    this("0m", 0, "")
  }

  /**
   * Define custom getter for GPU name.
   */
  def getName: String = name

  /**
   * Define custom setter for GPU name to ensure it is always in lower case.
   *
   * @see [[com.nvidia.spark.rapids.tool.GpuTypes]]
   */
  def setName(newName: String): Unit = {
    this.name = newName.toLowerCase
  }

  override def validate(): Unit = {
    if (getName != null && getName.nonEmpty) {
      device = GpuDevice.createInstance(getName).orElse {
        val supportedGpus = GpuDevice.deviceMap.keys.mkString(", ")
        throw new IllegalArgumentException(
          s"Unsupported GPU type provided: $getName. Supported GPU types: $supportedGpus")
      }
    }
  }

  def isMissingInfo: Boolean = {
    memory == null || memory.isEmpty || name == null || name.isEmpty ||
       count == 0 || memory.startsWith("0") || name == "None"
  }
  def isEmpty: Boolean = {
    count == 0 && (memory == null || memory.isEmpty || memory.startsWith("0")) &&
      (name == null || name.isEmpty || name == "None")
  }
  /**
   * If the GPU count is missing, it will set 1 as a default value
   *
   * @return true if the value has been updated.
   */
  def setDefaultGpuCountIfMissing(tuningConfigs: TuningConfigProvider): Boolean = {
    // TODO - do we want to recommend 1 or base it on core count?  32 cores to 1 gpu may be to much.
    if (count == 0) {
      count = tuningConfigs.getEntry("WORKER_GPU_COUNT").getDefault.toInt
      true
    } else {
      false
    }
  }
  def setDefaultGpuNameIfMissing(platform: Platform): Boolean = {
    if (!GpuDevice.deviceMap.contains(name)) {
      name = platform.gpuDevice.getOrElse(platform.defaultGpuDevice).toString
      true
    } else {
      false
    }
  }

  /**
   * If the GPU memory is missing, it will sets a default valued based on the GPU device type.
   * If it is still missing, it sets a default to 15109m (T4).
   *
   * @return true if the value has been updated.
   */
  def setDefaultGpuMemIfMissing(): Boolean = {
    if (memory == null || memory.isEmpty || memory.startsWith("0")) {
      memory = try {
        GpuDevice.createInstance(getName).getOrElse(GpuDevice.DEFAULT).getMemory
      } catch {
        case _: IllegalArgumentException => GpuDevice.DEFAULT.getMemory
      }
      true
    } else {
      false
    }
  }

  override def toString: String =
    s"{count: $count, memory: $memory, name: $name}"
}

/**
 * Represents different Spark master types.
 */
sealed trait SparkMaster {
  // Default executor memory to use in case not set by the user.
  val defaultExecutorMemoryMB: Long
}
case object Local extends SparkMaster {
  val defaultExecutorMemoryMB: Long = 1024L
}
case object Yarn extends SparkMaster {
  val defaultExecutorMemoryMB: Long = 1024L
}
case object Kubernetes extends SparkMaster {
  val defaultExecutorMemoryMB: Long = 1024L
}
case object Standalone extends SparkMaster {
  // Would be the entire node memory by default
  val defaultExecutorMemoryMB: Long = 1024L
}

object SparkMaster {
  def apply(master: Option[String]): Option[SparkMaster] = {
    master.flatMap {
      case url if url.contains("yarn") => Some(Yarn)
      case url if url.contains("k8s") => Some(Kubernetes)
      // Check for standalone Spark master before local mode as it can also contain "local"
      // E.g. spark://localhost:7077
      case url if url.contains("spark://") => Some(Standalone)
      case url if url.contains("local") => Some(Local)
      case _ => None
    }
  }
}

/**
 * AutoTuner module that uses event logs and worker's system properties to recommend Spark
 * RAPIDS configuration based on heuristics.
 *
 * Example:
 * a. Success:
 *    Input:
 *      system:
 *        num_cores: 64
 *        cpu_arch: x86_64
 *        memory: 512gb
 *        free_disk_space: 800gb
 *        time_zone: America/Los_Angeles
 *        num_workers: 4
 *      gpu:
 *        count: 8
 *        memory: 32gb
 *        name: NVIDIA V100
 *      softwareProperties:
 *        spark.driver.maxResultSize: 7680m
 *        spark.driver.memory: 15360m
 *        spark.executor.cores: '8'
 *        spark.executor.instances: '2'
 *        spark.executor.memory: 47222m
 *        spark.executorEnv.OPENBLAS_NUM_THREADS: '1'
 *        spark.extraListeners: com.google.cloud.spark.performance.DataprocMetricsListener
 *        spark.scheduler.mode: FAIR
 *        spark.sql.cbo.enabled: 'true'
 *        spark.ui.port: '0'
 *        spark.yarn.am.memory: 640m
 *
 *    Output:
 *       Spark Properties:
 *       --conf spark.executor.cores=8
 *       --conf spark.executor.instances=20
 *       --conf spark.executor.memory=16384m
 *       --conf spark.executor.memoryOverhead=5734m
 *       --conf spark.rapids.memory.pinnedPool.size=4096m
 *       --conf spark.rapids.sql.concurrentGpuTasks=2
 *       --conf spark.sql.files.maxPartitionBytes=4096m
 *       --conf spark.task.resource.gpu.amount=0.125
 *
 *       Comments:
 *       - 'spark.rapids.sql.concurrentGpuTasks' was not set.
 *       - 'spark.executor.memoryOverhead' was not set.
 *       - 'spark.rapids.memory.pinnedPool.size' was not set.
 *       - 'spark.sql.adaptive.enabled' should be enabled for better performance.
 *
 * b. Failure:
 *    Input: Incorrect File
 *    Output:
 *      Cannot recommend properties. See Comments.
 *
 *      Comments:
 *      - 'spark.executor.memory' should be set to at least 2GB/core.
 *      - 'spark.executor.instances' should be set to (gpuCount * numWorkers).
 *      - 'spark.task.resource.gpu.amount' should be set to Max(1, (numCores / gpuCount)).
 *      - 'spark.rapids.sql.concurrentGpuTasks' should be set to Min(4, (gpuMemory / 7.5G)).
 *      - 'spark.rapids.memory.pinnedPool.size' should be set to 2048m.
 *      - 'spark.sql.adaptive.enabled' should be enabled for better performance.
 *
 * @param appInfoProvider the container holding the profiling result.
 */
abstract class AutoTuner(
    val appInfoProvider: AppSummaryInfoBaseProvider,
    val platform: Platform,
    val driverInfoProvider: DriverLogInfoProvider,
    val userProvidedTuningConfigs: Option[TuningConfiguration],
    val autoTunerHelper: AutoTunerHelper)
  extends Logging with AutoTunerCommentsWithTuningConfigs with AutoTunerStaticComments {

  /** Type of config provider - defined by subclasses */
  type ConfigProviderType <: TuningConfigProvider

  /** Config provider instance - created using the factory method */
  lazy val configProvider: ConfigProviderType = createConfigProvider(userProvidedTuningConfigs)

  // Plugin manager to handle all tuning plugins.
  // By default it creates a manager that sort rules across plugins allowing better control
  // of the order of rule application.
  lazy val pluginManager = createPluginManager(true)

  var comments = new mutable.ListBuffer[String]()
  var recommendations: mutable.LinkedHashMap[String, TuningEntryTrait] =
    mutable.LinkedHashMap[String, TuningEntryTrait]()
  // Set of properties for which recommendations will be skipped.
  // Recommendations for these properties will not be computed, ensuring that dependent properties
  // are also affected correctly.
  private val skippedRecommendations: mutable.HashSet[String] = mutable.HashSet[String]()
  // Set of properties for which only source application values are used and
  // no calculations are performed.
  protected val limitedLogicRecommendations: mutable.HashSet[String] = mutable.HashSet[String]()
  // When enabled, the profiler recommendations should only include updated settings.
  private var filterByUpdatedPropertiesEnabled: Boolean = true
  // Non-executor memory (reserved for OS, resource manager, etc), configurable via tuning configs
  private lazy val nonExecutorMemory = configProvider.getEntry("NON_EXECUTOR_MEM")
    .getDefaultAsMemory(ByteUnit.MiB)

  // Non-executor memory fraction -> fraction of node memory not available for executors.
  // Executor available memory = total * (1 - nonExecutorMemFraction).
  // If value is in valid range [0, 1), use it; if negative, use platform default.
  // Values >= 1 are invalid and fall back to platform default with a warning.
  private lazy val nonExecutorMemFraction: Double = {
    val configValue = configProvider.getEntry("NON_EXECUTOR_MEM_FRACTION").getDefault.toDouble
    val platformDefault = platform.nonExecutorMemoryFraction
    if (configValue >= 0.0 && configValue < 1.0) {
      // Valid range [0, 1) - use the configured value
      // 0.0 means 100% available for executors (no memory reserved)
      configValue
    } else if (configValue < 0.0) {
      // Negative means "use platform default"
      platformDefault
    } else {
      // Invalid value (>= 1.0) - log warning and use platform default
      logWarning(s"Invalid NON_EXECUTOR_MEM_FRACTION value: $configValue. " +
        s"Must be between 0.0 and 1.0 (exclusive), or negative for platform default. " +
        s"Using platform default: $platformDefault")
      platformDefault
    }
  }

  // Executor available memory fraction = 1 - non-executor fraction
  private lazy val executorAvailableMemFraction: Double = 1.0 - nonExecutorMemFraction

  // Check if off-heap limit is enabled - centralized to avoid repeated property lookups
  private lazy val isOffHeapLimitUserEnabled: Boolean = {
    getBaselineSparkProperty("spark.rapids.memory.host.offHeapLimit.enabled")
      .exists(_.trim.equalsIgnoreCase("true"))
  }

  private lazy val sparkMaster: Option[SparkMaster] = {
    SparkMaster(appInfoProvider.getProperty("spark.master"))
  }

  /** Factory method to create the config provider - must be implemented by subclasses */
  protected def createConfigProvider(config: Option[TuningConfiguration]): ConfigProviderType

  private def createPluginManager(sortAcrossPlugins: Boolean): TuningPluginManager = {
    TuningPluginManager.builder
      .withTunerInst(this)
      .withSortRulesAcrossPlugins(sortAcrossPlugins)
      .build()
  }

  private def isCalculationEnabled(prop: String) : Boolean = {
    !limitedLogicRecommendations.contains(prop)
  }

  /**
   * Determines whether a tuning entry should be included in the final recommendations
   * returned to the user.
   * Subclasses can override this method to implement tool-specific filtering logic.
   *
   * @param tuningEntry the tuning entry to evaluate for inclusion
   * @return true if the entry should be included in the final recommendations, false otherwise
   */
  def shouldIncludeInFinalRecommendations(tuningEntry: TuningEntryTrait): Boolean = {
    if (platform.isPropertyPreserved(tuningEntry.name)) {
      // If the property is preserved, it should be included in the final recommendations.
      return true
    }
    if (filterByUpdatedPropertiesEnabled) {
      tuningEntry.isTuned()
    } else {
      tuningEntry.isEnabled()
    }
  }

  /**
   * Executes all tuning plugins to apply their rules and generate recommendations.
   */
  private def executeTuningPlugins(): Unit = {
    pluginManager.applyRules()
  }
  /**
   * Used to get the property value from the source properties
   * (i.e. from app info and cluster properties)
   */
  private def getPropertyValueFromSource(key: String): Option[String] = {
    getAllSourceProperties.get(key)
  }

  /**
   * Used to get the property value in the following priority order:
   * 1. Recommendations (this also includes the user-enforced properties)
   * 2. Source Spark properties (i.e. from app info and cluster properties)
   */
  def getPropertyValue(key: String): Option[String] = {
    AutoTuner.getCombinedPropertyFn(recommendations, getAllSourceProperties)(key)
  }

  /**
   * Get combined properties from the app info and cluster properties.
   */
  private lazy val getAllSourceProperties: Map[String, String] = {
    // the cluster properties override the app properties as
    // it is provided by the user.
    appInfoProvider.getAllProperties
  }

  /**
   * Combined tuning table that merges the default tuning definitions with user-defined ones.
   * Properties in the 'exclude' list are excluded from the final table.
   * Properties in the 'preserve' list are added with enabled=true and bootstrapEntry=true.
   * Properties in the 'enforced' map are added with enabled=true
   * Mutable to allow adding new definitions at runtime.
   */
  private lazy val finalTuningTable: Map[String, TuningEntryDefinition] = {
    // Start with the base tuning table and any tuning definitions from the target cluster
    val baseMap = scala.collection.mutable.Map.empty[String, TuningEntryDefinition] ++
      TuningEntryDefinition.TUNING_TABLE ++
      platform.targetCluster
        .map(_.getSparkProperties.tuningDefinitionsMap)
        .getOrElse(Map.empty[String, TuningEntryDefinition])

    // Enable platform-specific tuning entries.
    // Uses getEntryDefinition to preserve the full YAML metadata for disabled entries.
    platform.platformEnabledTuningEntries.foreach { key =>
      val tuningDefn = baseMap.getOrElseUpdate(key,
        TuningEntryDefinition.getEntryDefinition(key).getOrElse(TuningEntryDefinition(key)))
      tuningDefn.markAsEnable()
    }

    // Exclude properties specified in the skip list (Tool specific or
    // user specified using `exclude` section in target cluster)
    skippedRecommendations.foreach(baseMap.remove)

    // Add or update tuning definitions for limited logic properties (Tool specific
    // or user specified using `preserve` section in target cluster)
    limitedLogicRecommendations.foreach { key =>
      val tuningDefn = baseMap.getOrElseUpdate(key, TuningEntryDefinition(key))
      tuningDefn.markAsEnable()
    }

    // Add or update tuning definitions for user-enforced properties
    platform.userEnforcedRecommendations.keys.foreach { key =>
      // All user-enforced properties should be enabled and have bootstrap entries.
      val tuningDefn = baseMap.getOrElseUpdate(key, TuningEntryDefinition(key))
      tuningDefn.markAsEnable()
    }
    baseMap.toMap
  }

  def initRecommendations(): Unit = {
    finalTuningTable.keys.foreach { key =>
      // no need to add new records if they are missing from props
      getPropertyValueFromSource(key).foreach { propVal =>
        val recommendationVal = TuningEntry.build(key, Option(propVal), None,
          finalTuningTable.get(key))
        recommendations(key) = recommendationVal
      }
    }

    // Add properties with limited logic to the recommendations.
    // These properties should preserve their values from the source application.
    limitedLogicRecommendations.foreach { key =>
      getPropertyValueFromSource(key).foreach { sourceValue =>
        val recomRecord = recommendations.getOrElseUpdate(key,
          TuningEntry.build(key, Some(sourceValue), None, finalTuningTable.get(key)))
        recomRecord.setRecommendedValue(sourceValue)
      }
    }

    // Add the enforced properties to the recommendations.
    platform.userEnforcedRecommendations.foreach {
      case (key, value) =>
        val recomRecord = recommendations.getOrElseUpdate(key,
          TuningEntry.build(key, getPropertyValueFromSource(key), None, finalTuningTable.get(key)))
        recomRecord.setRecommendedValue(value)
        appendComment(getEnforcedPropertyComment(key))
    }
  }

  /**
   * Marks a Spark property as unresolved in the recommendations.
   *
   * This function is used when AutoTuner cannot determine a value for a required property
   * (e.g., executor memory).  This will cause a placeholder ("[FILL_IN_VALUE]") to be shown
   * in the AutoTuner output. However, if the property is excluded (i.e. not in the final tuning
   * table), this does nothing.
   */
  private def markAsUnresolved(sparkProperty: String, fillInValue: Option[String] = None): Unit = {
    finalTuningTable.get(sparkProperty).foreach { tuningDef =>
      val recomRecord = recommendations.getOrElseUpdate(sparkProperty,
        TuningEntry.build(sparkProperty, getPropertyValueFromSource(sparkProperty),
          None, Some(tuningDef)))
      recomRecord.markAsUnresolved(fillInValue)
    }
  }

  /**
   * Append a comment to the list by looking up the missing comment if any in the tuningEntry
   * table.
   * @param key the property set by the autotuner.
   */
  private def appendMissingComment(key: String): Unit = {
    val missingComment = finalTuningTable.get(key)
      .flatMap(_.getMissingComment())
      .getOrElse(s"was not set.")
    appendComment(key, missingComment)
  }

  /**
   * Append a comment to the list by looking up the persistent comment if any.
   * @param key the property set by the autotuner.
   */
  private def appendPersistentComment(key: String): Unit = {
    finalTuningTable.get(key).foreach { eDef =>
      eDef.getPersistentComment().foreach { comment =>
        appendComment(key, comment)
      }
    }
  }

  /**
   * Append a comment to the list by looking up the updated comment if any in the tuningEntry
   * table. If it is not defined in the table, then add nothing.
   * @param key the property set by the autotuner.
   */
  private def appendUpdatedComment(key: String): Unit = {
    finalTuningTable.get(key).foreach { eDef =>
      eDef.getUpdatedComment().foreach { comment =>
        appendComment(key, comment)
      }
    }
  }

  /**
   * Append the description of the property as a comment.
   * @param key the property set by the autotuner.
   */
  private def appendDescriptionAsComment(key: String): Unit = {
    finalTuningTable.get(key).foreach { eDef =>
      val comment = if (eDef.getDescription.isEmpty) {
        s"No description available."
      } else {
        eDef.getDescription
      }
      appendComment(key, comment)
    }
  }

  /**
   * Determines if the recommendation should be ignored.
   * Criteria:
   * - The property is in the skipped recommendations list
   * - The property is in the limited logic recommendations list
   * - The property is preserved from source values or otherwise marked for limited logic
   * - The property is enforced by the user
   *
   * Preserved and enforced properties are handled during initRecommendations() and
   * initialization of finalTuningTable, so later recommendation logic should not overwrite them.
   * @param key the property to check
   * @return true if the recommendation should be ignored, false otherwise
   */
  private def ignoreRecommendation(key: String): Boolean = {
    skippedRecommendations.contains(key) || limitedLogicRecommendations.contains(key) ||
      platform.getUserEnforcedSparkProperty(key).isDefined
  }

  def appendRecommendation(key: String, value: String): Unit = {
    if (ignoreRecommendation(key)) {
      return
    }
    // Update the recommendation entry or update the existing one.
    val recomRecord = recommendations.getOrElseUpdate(key,
      TuningEntry.build(key, getPropertyValue(key), None, finalTuningTable.get(key)))
    // if the value is not null, then proceed to add the recommendation.
    Option(value).foreach { nonNullValue =>
      recomRecord.setRecommendedValue(nonNullValue)
      recomRecord.getOriginalValue match {
        case None =>
          // add missing comment if any
          appendMissingComment(key)
        case Some(originalValue) if originalValue != recomRecord.getTuneValue() =>
          // add updated comment if any
          appendUpdatedComment(key)
        case _ =>
          // do not add any comment if the tuned value is the same as the original value
      }
      // add the persistent comment if any.
      appendPersistentComment(key)
    }
  }

  /**
   * Safely appends the recommendation to the given key.
   * It skips if the value is 0.
   */
  def appendRecommendation(key: String, value: Long): Unit = {
    if (value > 0L) {
      appendRecommendation(key: String, s"$value")
    }
  }

  /**
   * Safely appends the recommendation to the given key.
   * It skips if the value is 0.0.
   */
  def appendRecommendation(key: String, value: Double): Unit = {
    if (value > 0.0) {
      appendRecommendation(key: String, s"$value")
    }
  }
  /**
   * Safely appends the recommendation to the given key.
   * It appends "m" to the string value. It skips if the value is 0 or null.
   */
  def appendRecommendationForMemoryMB(key: String, value: String): Unit = {
    if (value != null && value.toDouble > 0.0) {
      appendRecommendation(key, s"${value}m")
    }
  }

  /**
   * Remove the recommendation for the given key if it exists.
   */
  private def removeRecommendation(key: String): Unit = {
    this.recommendations.get(key).foreach(_.markAsRemoved())
  }

  /**
   * Try to figure out the recommended instance type to use and set
   * the executor cores and instances based on that instance type.
   * Returns None if the platform doesn't support specific instance types.
   */
  private def configureGPURecommendedInstanceType(): Unit = {
    platform.createRecommendedGpuClusterInfo(recommendations, getAllSourceProperties,
      autoTunerHelper.recommendedClusterSizingStrategy(platform))
    platform.recommendedClusterInfo.foreach { gpuClusterRec =>
      // TODO: Should we skip recommendation if cores per executor is lower than a min value?
      appendRecommendation("spark.executor.cores", gpuClusterRec.coresPerExecutor)
      if (gpuClusterRec.numExecutors > 0) {
        // Note: This may change later if dynamic allocation is enabled.
        appendRecommendation("spark.executor.instances", gpuClusterRec.numExecutors)
      }
    }
  }

  /**
   * Returns the label of the multithread read core multiplier property from
   * the tuning table, if present.
   * This is used when calculating the number of threads for
   * 'spark.rapids.sql.multiThreadedRead.numThreads'.
   */
  private def getMultithreadReadCoreMultiplierProperty: Option[String] = {
    val coreMultiplierDefs = finalTuningTable.values
      .filter(_.getCategoryAsEnum == CategoryEnum.MultiThreadReadCoreMultiplier)
    // If more than one property is found for the multithread read core multiplier category,
    // we do not know which one to use. Therefore, raise an error.
    require(coreMultiplierDefs.size <= 1,
      s"Only one multithread read core multiplier property is allowed. " +
        s"Found: ${coreMultiplierDefs.map(_.label).mkString(", ")}")

    coreMultiplierDefs.headOption.map(_.label)
  }

  /**
   * Recommendation for 'spark.rapids.sql.concurrentGpuTasks' based on gpu memory.
   * Assumption - cluster properties were updated to have a default values if missing.
   */
  private def calcGpuConcTasks(): Long = {
    Math.min(configProvider.getEntry("CONC_GPU_TASKS").getMax.toLong,
      platform.recommendedGpuDevice.getGpuConcTasks(
        configProvider.getEntry("GPU_MEM_PER_TASK").getDefaultAsMemory(ByteUnit.MiB)))
  }

  /**
   * Extracts the unique RAPIDS plugin jar version from the application's classpath
   * entries. Returns None if no version (or more than one distinct version) is found.
   */
  private def getRapidsPluginJarVersion: Option[String] = {
    appInfoProvider.getRapidsJars
      .flatMap(autoTunerHelper.pluginJarRegEx.findAllMatchIn(_).map(_.group(1)))
      .distinct match {
        case Seq(ver) => Some(ver)
        case _ => None
      }
  }

  /**
   * Returns true when the application uses a RAPIDS plugin version that already
   * auto-tunes `spark.rapids.sql.concurrentGpuTasks` at runtime, in which case
   * the AutoTuner should drop its recommendation for that property.
   * Reference: https://github.com/NVIDIA/spark-rapids/pull/12374
   */
  private def isConcurrentGpuTasksAutoTunedByPlugin: Boolean = {
    getRapidsPluginJarVersion.exists { jarVer =>
      ToolUtils.compareVersions(jarVer, autoTunerHelper.pluginVersionAutoConcurrentGpuTasks)
        .exists(_ >= 0)
    }
  }

  /**
   * Recommendation for initial heap size based on certain amount of memory per core.
   * Note that we will later reduce this if needed for off heap memory.
   */
  def calcInitialExecutorHeapInMB(executorContainerMemCalculator: () => Double,
      numExecCores: Int): Long = {
    val maxExecutorHeap = Math.max(0, executorContainerMemCalculator()).toInt
    // give up to 2GB of heap to each executor core
    // TODO - revisit this in future as we could let heap be bigger
    Math.min(maxExecutorHeap,
      configProvider.getEntry("HEAP_PER_CORE").getDefaultAsMemory(ByteUnit.MiB) * numExecCores)
  }

  /**
   * Spark property value to use as an input baseline for recommendation calculations.
   * User-enforced values take precedence over preserved source values.
   */
  private def getBaselineSparkProperty(key: String): Option[String] = {
    platform.getEnforcedOrPreservedSparkProperty(key, getPropertyValueFromSource)
  }

  /**
   * Note: All memory values are in MB.
   */
  private case class MemorySettings(
    executorHeap: Option[Long],
    executorMemOverhead: Option[Long],
    pinnedMem: Option[Long],
    spillMem: Option[Long],
    sparkOffHeapMem: Option[Long]
  ) {
    def hasAnyMemorySettings: Boolean = {
      executorMemOverhead.isDefined ||
        pinnedMem.isDefined ||
        spillMem.isDefined ||
        sparkOffHeapMem.isDefined
    }
  }

  private lazy val baselineMemorySettings: MemorySettings = {
    def baseline(key: String): Option[Long] =
      getBaselineSparkProperty(key).map(StringUtils.convertToMB(_, Some(ByteUnit.BYTE)))
    val executorHeap = baseline("spark.executor.memory")
    val executorMemOverhead = baseline("spark.executor.memoryOverhead")
    val pinnedMem = baseline("spark.rapids.memory.pinnedPool.size")
    val spillMem = baseline("spark.rapids.memory.spillPool.size")
    val sparkOffHeapMem = baseline("spark.memory.offHeap.size")
    MemorySettings(executorHeap, executorMemOverhead, pinnedMem, spillMem, sparkOffHeapMem)
  }

  private def generateInsufficientMemoryComment(
      executorHeap: Long,
      finalExecutorMemOverhead: Long,
      sparkOffHeapMemMB: Long,
      pySparkMemMB: Long): String = {
    val executorMemRequired =
      executorHeap + finalExecutorMemOverhead + sparkOffHeapMemMB + pySparkMemMB
    // Calculate total system memory needed, consistent with actual allocation logic:
    // - If nonExecutorMemory > 0: add absolute reservation
    // - Otherwise: divide by available fraction to account for container manager reservation
    val minTotalExecMemRequired: Long = if (nonExecutorMemory > 0) {
      executorMemRequired + nonExecutorMemory
    } else {
      (executorMemRequired / executorAvailableMemFraction).toLong
    }
    notEnoughMemComment(minTotalExecMemRequired)
  }

  // scalastyle:off line.size.limit
  /**
   * Calculates recommended memory settings for a Spark executor container.
   *
   * The total memory for the executor is the sum of:
   *   executorHeap (spark.executor.memory)
   *   + executorMemOverhead (spark.executor.memoryOverhead)
   *   + sparkOffHeapMemMB (spark.memory.offHeap.size)
   *   + pySparkMemMB (spark.executor.pyspark.memory)
   *
   * Note: In the below examples, `0.8` is the fraction of the physical system memory
   * that is available to Spark executors (0.2 is reserved by Dataproc YARN).
   *
   * Example 1: g2-standard-8 machine (32 GB total memory) — Just enough memory
   *   - actualMemForExec =  32 GB * 0.8 = 25.6 GB
   *   - executorHeap = 16 GB
   *   - sparkOffHeapMemMB = 4 GB
   *   - execMemLeft = 25.6 GB - 16 GB - 4 GB = 5.6 GB
   *   - minOverhead = 1.6 GB (10% of executor heap) + 2 GB (min pinned) + 2 GB (min spill) = 5.6 GB
   *   - Since execMemLeft (5.6 GB) == minOverhead (5.6 GB), proceed with minimum memory recommendations:
   *   - Recommendation:
   *       - executorHeap = 16 GB, executorMemOverhead = 5.6 GB (with pinnedMem = 2 GB and spillMem = 2 GB)
   *
   * Example 2: g2-standard-16 machine (64 GB total memory) — Not enough memory
   *   - actualMemForExec = 64 GB * 0.8 = 51.2 GB
   *   - executorHeap = 32 GB
   *   - sparkOffHeapMemMB = 20 GB
   *   - execMemLeft = 51.2 GB - 32 GB - 20 GB = -0.8 GB
   *   - minOverhead = 2 GB (min pinned) + 2 GB (min spill) + 3.2 GB (10% of executor heap) = 7.2 GB
   *   - Since execMemLeft (-0.8 GB) < minOverhead (7.2 GB), do not proceed with recommendations
   *       - Add a warning comment indicating that the current setup is not optimal
   *           - minTotalExecMemRequired = (32 GB + 20 GB + 7.2 GB) / 0.8 = (59.2 GB / 0.8) = 74 GB (as we are using 80% of system memory)
   *           - Reduce off-heap size or use a larger machine with at least 74 GB system memory.
   *
   * Example 3: g2-standard-16 machine (64 GB total memory) — More memory available
   *   - actualMemForExec = 64 GB * 0.8 = 51.2 GB
   *   - executorHeap = 32 GB
   *   - sparkOffHeapMemMB = 10 GB
   *   - execMemLeft = 51.2 GB - 32 GB - 10 GB = 9.2 GB
   *   - minOverhead = 2 GB (min pinned) + 2 GB (min spill) + 3.2 GB (10% of executor heap) = 7.2 GB
   *   - Since execMemLeft (9.2 GB) > minOverhead (7.2 GB), proceed with recommendations.
   *       - Increase pinned and spill memory based on remaining memory (up to 4 GB max)
   *       - executorMemOverhead = 3 GB (pinned) + 3 GB (spill) + 3.2 GB = 9.2 GB
   *   - Recommendation:
   *       - executorHeap = 32 GB, executorMemOverhead = 9.2 GB (with pinnedMem = 3 GB and spillMem = 3 GB)
   *
   *
   * @param execHeapCalculator    Function that returns the executor heap size in MB
   * @param numExecutorCores      Number of executor cores
   * @param totalMemForExecExpr   Function that returns total memory available to the executor (MB)
   * @return Either a String with an error message if memory is insufficient,
   *         or a tuple containing:
   *           - pinned memory size (MB)
   *           - executor memory overhead size (MB)
   *           - executor heap size (MB)
   */
   // scalastyle:on line.size.limit
  private def calcOverallMemory(
      execHeapCalculator: () => Long,
      numExecutorCores: Int,
      totalMemForExecExpr: () => Double): Either[String, MemorySettings] = {

    // Set executor heap using a baseline value, if present, otherwise max of
    // calculator result and 2GB/core.
    val executorHeapMB = baselineMemorySettings.executorHeap.getOrElse {
      Math.max(
        execHeapCalculator(),
        configProvider
          .getEntry("HEAP_PER_CORE")
          .getDefaultAsMemory(ByteUnit.MiB) * numExecutorCores)
    }
    // Calculate total available memory for executors based on OS reserved memory:
    // - If nonExecutorMemory > 0: use absolute subtraction (for on-prem environments)
    // - If nonExecutorMemory = 0: use platform fraction (for CSPs with container managers)
    val totalMemForExecutors = totalMemForExecExpr.apply().toLong
    val totalMemMinusReserved: Long = if (nonExecutorMemory > 0) {
      totalMemForExecutors - nonExecutorMemory
    } else {
      // Our CSP instance map stores full node memory, but container managers
      // (e.g., YARN) may reserve a portion. Adjust to get the memory
      // actually available to the executor.
      totalMemForExecutors * executorAvailableMemFraction
    }.toLong
    // Calculate off-heap memory size using new hybrid scan detection logic
    val sparkOffHeapMemMB: Long = baselineMemorySettings.sparkOffHeapMem.getOrElse(
      calculateOffHeapMemorySize(numExecutorCores)
    )
    val pySparkMemMB = platform.getPySparkMemoryMB(getPropertyValue).getOrElse(0L)
    // Calculate executor memory overhead using new formula if OffHeapLimit.enabled=true
    val executorMemOverhead = if (isOffHeapLimitUserEnabled) {
      calculateExecutorMemoryOverhead(
        totalMemMinusReserved, executorHeapMB, sparkOffHeapMemMB)
    } else {
      // If OffHeapLimit.enabled=false, use the old formula
      executorHeapMB * configProvider.getEntry("HEAP_OVERHEAD_FRACTION").getDefault.toDouble
    }.toLong
    val execMemLeft = totalMemMinusReserved - executorHeapMB - sparkOffHeapMemMB - pySparkMemMB
    val defaultPinnedMem = configProvider.getEntry("PINNED_MEMORY").getDefaultAsMemory(ByteUnit.MiB)
    val defaultSpillMem = configProvider.getEntry("SPILL_MEMORY").getDefaultAsMemory(ByteUnit.MiB)
    val minOverhead: Long = baselineMemorySettings.executorMemOverhead.getOrElse {
      if (isOffHeapLimitUserEnabled) {
        executorMemOverhead
      } else {
        executorMemOverhead + defaultPinnedMem + defaultSpillMem
      }
    }
    logDebug(s"Memory calculations:  totalMemMinusReserved=$totalMemMinusReserved MB, " +
      s"executorHeap=$executorHeapMB MB, sparkOffHeapMem=$sparkOffHeapMemMB MB, " +
      s"pySparkMem=$pySparkMemMB MB minOverhead=$minOverhead MB")
    if (execMemLeft >= minOverhead) {
      // this is hopefully path in the majority of cases because CSPs generally have a good
      // memory to core ratio
      // Calculate host off-heap limit size for pinned memory calculation
      // (only for onPrem when offHeapLimit is enabled)
      val hostOffHeapLimitSizeMB = if (!platform.isPlatformCSP &&
        isOffHeapLimitUserEnabled) {
        val userOffHeapLimitOpt =
          getBaselineSparkProperty("spark.rapids.memory.host.offHeapLimit.size")
        if (userOffHeapLimitOpt.isDefined) {
          StringUtils.convertToMB(
            userOffHeapLimitOpt.get,
            Some(ByteUnit.BYTE))
        } else {
          executorMemOverhead + sparkOffHeapMemMB
        }
      } else {
        0L // Not used for CSP platforms or when offHeapLimit is disabled
      }

      // Pinned memory calculation - use new formula for onPrem, original logic for CSP
      var pinnedMem = baselineMemorySettings.pinnedMem.getOrElse {
        if (!platform.isPlatformCSP && hostOffHeapLimitSizeMB > 0) {
          // Use new formula for onPrem platform
          calculatePinnedMemorySize(numExecutorCores, hostOffHeapLimitSizeMB)
        } else {
          // Use original logic for CSP platforms or when host off-heap limit calculation fails
          Math.min(configProvider.getEntry("PINNED_MEMORY").getMaxAsMemory(ByteUnit.MiB),
            (execMemLeft - executorMemOverhead) / 2)
        }
      }
      // Spill storage is set to the pinned size by default. Its not guaranteed to use just pinned
      // memory though so the size worst case would be doesn't use any pinned memory and uses
      // all off heap memory.
      var spillMem = baselineMemorySettings.spillMem.getOrElse(pinnedMem)
      var finalExecutorMemOverhead = baselineMemorySettings.executorMemOverhead.getOrElse {
        if (isOffHeapLimitUserEnabled) {
          executorMemOverhead
        } else {
          // Budget-aware: claim the full available memory (execMemLeft) as overhead
          // so the container request (heap + overhead) covers the total node memory.
          // This ensures freed memory from HEAP_PER_CORE capping is not wasted, and
          // any residual budget is available as non-pinned spill fallback, JVM off-heap
          // headroom, and prevents K8s/YARN from over-scheduling the node.
          Math.max(executorMemOverhead + pinnedMem + spillMem, execMemLeft)
        }
      }
      // Handle the case when the final executor memory overhead is larger than the
      // available memory left for the executor.
      if (execMemLeft < finalExecutorMemOverhead) {
        // If there are any baseline memory settings (user-enforced or preserved),
        // add a warning comment indicating that the current setup is not optimal
        // and no memory-related tunings are recommended.
        if (baselineMemorySettings.hasAnyMemorySettings) {
          return Left(generateInsufficientMemoryComment(executorHeapMB, finalExecutorMemOverhead,
            sparkOffHeapMemMB, pySparkMemMB))
        }
        // Else update pinned and spill memory to use default values
        pinnedMem = defaultPinnedMem
        spillMem = defaultSpillMem
        finalExecutorMemOverhead = if (isOffHeapLimitUserEnabled) {
          executorMemOverhead
        } else {
          executorMemOverhead + defaultPinnedMem + defaultSpillMem
        }
      }
      // Add recommendations for executor memory settings
      Right(MemorySettings(Some(executorHeapMB), Some(finalExecutorMemOverhead), Some(pinnedMem),
        Some(spillMem), Some(sparkOffHeapMemMB)))
    } else {
      // Add a warning comment indicating that the current setup is not optimal
      // and no memory-related tunings are recommended.
      // TODO: For CSPs, we should recommend a different instance type.
      Left(generateInsufficientMemoryComment(executorHeapMB, minOverhead,
        sparkOffHeapMemMB, pySparkMemMB))
    }
  }

  private def configureShuffleReaderWriterNumThreads(numExecutorCores: Int): Unit = {
    // if on a CSP using blob store recommend more threads for certain sizes. This is based on
    // testing on customer jobs on Databricks
    // didn't test with > 16 thread so leave those as numExecutorCores
    if (numExecutorCores < 4) {
      // leave as defaults - should we reduce less then default of 20? need more testing
    } else if (numExecutorCores >= 4 && numExecutorCores < 16) {
      appendRecommendation("spark.rapids.shuffle.multiThreaded.reader.threads", 20)
      appendRecommendation("spark.rapids.shuffle.multiThreaded.writer.threads", 20)
    } else if (numExecutorCores >= 16 && numExecutorCores < 20 && platform.isPlatformCSP) {
      appendRecommendation("spark.rapids.shuffle.multiThreaded.reader.threads", 28)
      appendRecommendation("spark.rapids.shuffle.multiThreaded.writer.threads", 28)
    } else {
      val numThreads = (numExecutorCores * 1.5).toLong
      appendRecommendation("spark.rapids.shuffle.multiThreaded.reader.threads", numThreads.toInt)
      appendRecommendation("spark.rapids.shuffle.multiThreaded.writer.threads", numThreads.toInt)
    }
  }

  // Currently only applies many configs for CSPs where we have an idea what network/disk
  // configuration is like. On prem we don't know so don't set these for now.
  private def configureMultiThreadedReaders(numExecutorCores: Int): Unit = {

    // Helper function to get the bounded number of threads
    def getBoundedNumThreads(coreMultiplier: Double): Int = {
      val numThreads = (numExecutorCores * coreMultiplier).toInt
      val numThreadsTuningEntry = configProvider.getEntry("MULTITHREAD_READ_NUM_THREADS")
      val minThreads = numThreadsTuningEntry.getMin.toInt
      val maxThreads = numThreadsTuningEntry.getMax.toInt
      val boundedThreads = Math.max(minThreads, Math.min(maxThreads, numThreads))
      logDebug(s"Bounded numThreads: $boundedThreads " +
        s"(raw=$numThreads, min=$minThreads, max=$maxThreads)")
      boundedThreads
    }

    val coreMultiplierProp =
      getMultithreadReadCoreMultiplierProperty.flatMap(getPropertyValue).map(_.toDouble)
    // If a core multiplier is defined in the property, use it to calculate
    // the number of threads for multithreaded reads.
    if (coreMultiplierProp.isDefined && !platform.isPlatformCSP) {
      appendRecommendation("spark.rapids.sql.multiThreadedRead.numThreads",
        getBoundedNumThreads(coreMultiplierProp.get))
      return
    }
    if (numExecutorCores < 4) {
      appendRecommendation("spark.rapids.sql.multiThreadedRead.numThreads",
        Math.max(20, numExecutorCores))
    } else if (numExecutorCores >= 4 && numExecutorCores < 8 && platform.isPlatformCSP) {
      appendRecommendation("spark.rapids.sql.multiThreadedRead.numThreads",
        Math.max(20, numExecutorCores))
    } else if (numExecutorCores >= 8 && numExecutorCores < 16 && platform.isPlatformCSP) {
      appendRecommendation("spark.rapids.sql.multiThreadedRead.numThreads",
        Math.max(40, numExecutorCores))
    } else if (numExecutorCores >= 16 && numExecutorCores < 20 && platform.isPlatformCSP) {
      appendRecommendation("spark.rapids.sql.multiThreadedRead.numThreads",
        Math.max(80, numExecutorCores))
      appendRecommendation("spark.rapids.sql.reader.multithreaded.combine.sizeBytes",
        configProvider.getEntry("READER_MULTITHREADED_COMBINE_THRESHOLD").getDefault)
      appendRecommendation("spark.rapids.sql.format.parquet.multithreaded.combine.waitTime",
        configProvider.getEntry("READER_MULTITHREADED_COMBINE_WAIT_TIME").getDefault)
    } else {
      // For 20+ cores, use the core multiplier defined in tuning configs
      // to calculate the number of threads for multithreaded reads.
      val coreMultiplier =
        configProvider.getEntry("MULTITHREAD_READ_CORE_MULTIPLIER").getDefault.toDouble
      appendRecommendation("spark.rapids.sql.multiThreadedRead.numThreads",
        getBoundedNumThreads(coreMultiplier))
      if (platform.isPlatformCSP) {
        appendRecommendation("spark.rapids.sql.reader.multithreaded.combine.sizeBytes",
          configProvider.getEntry("READER_MULTITHREADED_COMBINE_THRESHOLD").getDefault)
        appendRecommendation("spark.rapids.sql.format.parquet.multithreaded.combine.waitTime",
          configProvider.getEntry("READER_MULTITHREADED_COMBINE_WAIT_TIME").getDefault)
      }
    }
  }

  /**
   * Recommend dynamic allocation configurations for GPU runs.
   * Adjusts initialExecutors, minExecutors, and maxExecutors based on the ratio
   * of source cores to target cores.
   *
   * Formula: adjusted_value = max(1, floor(original_value × source_cores / target_cores))
   *
   * Note: It also updates the executor instances to match the initial executors.
   *
   * @param gpuExecCores Number of cores per executor for GPU runs
   */
  private def recommendDynamicAllocationConfigs(gpuExecCores: Int): Unit = {
    val isDynamicAllocationEnabled = getPropertyValue("spark.dynamicAllocation.enabled")
      .exists(_.trim.equalsIgnoreCase("true"))

    if (!isDynamicAllocationEnabled) {
      // If dynamic allocation is disabled, remove the recommendations for the
      // dynamic allocation properties
      removeRecommendation("spark.dynamicAllocation.initialExecutors")
      removeRecommendation("spark.dynamicAllocation.minExecutors")
      removeRecommendation("spark.dynamicAllocation.maxExecutors")
      return
    }

    // Get the original executor cores from the event log
    val sourceExecCores = getPropertyValueFromSource("spark.executor.cores")
      .map(_.toInt)
      .getOrElse(1)

    if (sourceExecCores <= 0 || gpuExecCores <= 0) {
      return
    }

    // Scale factor to adjust executor counts based on core ratio.
    // Example: source=8 cores, target=16 cores → ratio=0.5 → need fewer executors
    // Example: source=16 cores, target=8 cores → ratio=2.0 → need more executors
    val adjustRatio = sourceExecCores.toDouble / gpuExecCores

    // Helper function to get the adjusted value for a property
    // Formula: adjusted_value = max(1, floor(original_value × source_cores / target_cores))
    def adjustedValue(property: String): Option[Int] = {
      if (ignoreRecommendation(property)) {
        None
      } else {
        val cpuValue = getPropertyValueFromSource(property).map(_.toInt)
        cpuValue.map(v => math.max(1, math.floor(v * adjustRatio).toInt))
      }
    }

    // Keep track of which properties were adjusted for comment generation
    val adjustedProperties = mutable.ListBuffer[String]()

    // Helper function to add to adjusted properties if the value is different
    // from the source value.
    def markAsAdjustedProperty(property: String, value: Int): Unit = {
      val sourceValue = getPropertyValueFromSource(property).map(_.toInt).filter(_ > 0)
      if (sourceValue.exists(_ != value)) {
        adjustedProperties += property
      }
    }

    // Handle initialExecutors and executor.instances together
    // Ref: https://spark.apache.org/docs/3.5.7/configuration.html#dynamic-allocation
    adjustedValue("spark.dynamicAllocation.initialExecutors").foreach { v =>
      // First, find the max recommended value for executor instances
      // as max of executor.instances, initialExecutors, and minExecutors
      // This handles if there are any enforced values for either of the
      // above properties.
      val recInstanceValues: Seq[Int] = Seq(
        recommendations.get("spark.executor.instances"),
        recommendations.get("spark.dynamicAllocation.initialExecutors"),
        recommendations.get("spark.dynamicAllocation.minExecutors")
      ).flatMap(_.flatMap(_.tunedValue).map(_.toInt))
      val recInstancesOpt = if (recInstanceValues.nonEmpty) Some(recInstanceValues.max) else None
      // Use the max of the adjusted value and the recommended value
      // of executor instances to avoid reducing the number of executors.
      val valueToUse = recInstancesOpt match {
        case Some(recInst) if recInst > v =>
          recInst
        case _ =>
          // Track this only if the selected value is the adjusted value.
          markAsAdjustedProperty("spark.dynamicAllocation.initialExecutors", v)
          v
      }
      appendRecommendation("spark.dynamicAllocation.initialExecutors", valueToUse.toLong)
      // Set spark.executor.instances to match initialExecutors to maintain consistency.
      appendRecommendation("spark.executor.instances", valueToUse.toLong)
    }

    // Simpler logic for min and max executors
    adjustedValue("spark.dynamicAllocation.minExecutors").foreach { v =>
      markAsAdjustedProperty("spark.dynamicAllocation.minExecutors", v)
      appendRecommendation("spark.dynamicAllocation.minExecutors", v)
    }

    adjustedValue("spark.dynamicAllocation.maxExecutors").foreach { v =>
      markAsAdjustedProperty("spark.dynamicAllocation.maxExecutors", v)
      appendRecommendation("spark.dynamicAllocation.maxExecutors", v)
    }

    // Enforce Spark's invariant: minExecutors <= initialExecutors <= maxExecutors
    // This is needed because initialExecutors may have been boosted by executor.instances
    // while maxExecutors was independently scaled down by the core ratio.
    enforceDynamicAllocationInvariants()

    if (adjustedProperties.nonEmpty) {
      appendComment(commentForDynamicAllocationAdjustment(adjustedProperties.toList,
        sourceExecCores, gpuExecCores))
    }
  }

  /**
   * Enforces Spark's dynamic allocation invariant:
   *   minExecutors <= initialExecutors <= maxExecutors
   *
   * This is necessary because initialExecutors may be boosted to match executor.instances
   * (from cluster sizing), while maxExecutors is independently scaled by the core ratio,
   * potentially resulting in initialExecutors > maxExecutors.
   */
  private def enforceDynamicAllocationInvariants(): Unit = {
    def getRecValue(prop: String): Option[Int] = {
      recommendations.get(prop).flatMap(_.tunedValue).map(_.toInt)
    }

    val adjusted = mutable.ListBuffer[String]()

    // Skip ignored properties, mirroring appendRecommendation
    def trackAdjustment(prop: String, from: Int, to: Int): Unit = {
      if (!ignoreRecommendation(prop)) {
        adjusted += s"'$prop' adjusted from $from to $to"
      }
    }

    val minOpt = getRecValue(
      "spark.dynamicAllocation.minExecutors")
    val initialOpt = getRecValue(
      "spark.dynamicAllocation.initialExecutors")
    val maxOpt = getRecValue(
      "spark.dynamicAllocation.maxExecutors")

    // Cap initialExecutors and executor.instances to maxExecutors
    (initialOpt, maxOpt) match {
      case (Some(initial), Some(max)) if initial > max =>
        appendRecommendation(
          "spark.dynamicAllocation.initialExecutors", max.toLong)
        appendRecommendation(
          "spark.executor.instances", max.toLong)
        trackAdjustment(
          "spark.dynamicAllocation.initialExecutors",
          initial, max)
        trackAdjustment(
          "spark.executor.instances", initial, max)
      case _ =>
    }

    // Cap minExecutors to maxExecutors
    (minOpt, maxOpt) match {
      case (Some(min), Some(max)) if min > max =>
        appendRecommendation(
          "spark.dynamicAllocation.minExecutors", max.toLong)
        trackAdjustment(
          "spark.dynamicAllocation.minExecutors", min, max)
      case _ =>
    }

    // Ensure minExecutors <= initialExecutors
    val adjustedInitialOpt = getRecValue(
      "spark.dynamicAllocation.initialExecutors")
    val adjustedMinOpt = getRecValue(
      "spark.dynamicAllocation.minExecutors")
    (adjustedMinOpt, adjustedInitialOpt) match {
      case (Some(min), Some(initial)) if min > initial =>
        appendRecommendation(
          "spark.dynamicAllocation.initialExecutors",
          min.toLong)
        appendRecommendation(
          "spark.executor.instances", min.toLong)
        trackAdjustment(
          "spark.dynamicAllocation.initialExecutors",
          initial, min)
        trackAdjustment(
          "spark.executor.instances", initial, min)
      case _ =>
    }

    if (adjusted.nonEmpty) {
      val comment =
        s"""
           |Adjusted dynamic allocation properties to enforce
           |minExecutors <= initialExecutors <= maxExecutors:
           |${adjusted.mkString("; ")}.
           |""".stripMargin.trim.replaceAll("\n", "\n  ")
      appendComment(comment)
    }
  }

  def calculateClusterLevelRecommendations(): Unit = {
    // only if we were able to figure out a node type to recommend do we make
    // specific recommendations
    if (platform.recommendedClusterInfo.isDefined) {
      // Set to low value for Spark RAPIDS usage as task parallelism will be honoured
      // by `spark.executor.cores`.
      recommendExecutorResourceGpuProps()
      appendRecommendation("spark.task.resource.gpu.amount",
        configProvider.getEntry("TASK_GPU_RESOURCE_AMT").getDefault.toDouble)
      val concGpuTasksKey = "spark.rapids.sql.concurrentGpuTasks"
      // Target cluster `enforced` and `preserve` overrides take precedence; only drop the
      // recommendation when neither is set and the plugin already auto-tunes it.
      if (!platform.isPropertyUserOverridden(concGpuTasksKey) &&
          isConcurrentGpuTasksAutoTunedByPlugin) {
        // Plugin version auto-tunes concurrent GPU tasks based on memory usage,
        // so suppress the AutoTuner recommendation and the corresponding missing comment.
        // Reference: https://github.com/NVIDIA/spark-rapids/pull/12374
        skippedRecommendations += concGpuTasksKey
      } else {
        appendRecommendation(concGpuTasksKey, calcGpuConcTasks())
      }
      val execCores = platform.recommendedClusterInfo.map(_.coresPerExecutor).getOrElse(1)
      val availableMemPerExec =
        platform.recommendedWorkerNode.map(_.getMemoryPerExec).getOrElse(0.0)
      if (availableMemPerExec > 0.0) {
        val availableMemPerExecExpr = () => availableMemPerExec
        val executorHeapInMB = calcInitialExecutorHeapInMB(availableMemPerExecExpr, execCores)
        val executorHeapExpr = () => executorHeapInMB
        calcOverallMemory(executorHeapExpr, execCores, availableMemPerExecExpr) match {
          case Right(recomMemorySettings: MemorySettings) =>
            // Sufficient memory available, proceed with recommendations
            appendRecommendationForMemoryMB("spark.rapids.memory.pinnedPool.size",
              s"${recomMemorySettings.pinnedMem.get}")
            // scalastyle:off line.size.limit
            // For YARN and Kubernetes, we need to set the executor memory overhead
            // Ref: https://spark.apache.org/docs/latest/configuration.html#:~:text=This%20option%20is%20currently%20supported%20on%20YARN%20and%20Kubernetes.
            // scalastyle:on line.size.limit
            if (sparkMaster.contains(Yarn) || sparkMaster.contains(Kubernetes)) {
              appendRecommendationForMemoryMB("spark.executor.memoryOverhead",
                s"${recomMemorySettings.executorMemOverhead.get}")
            }
            appendRecommendationForMemoryMB("spark.executor.memory",
              s"${recomMemorySettings.executorHeap.get}")

            // Add off-heap memory recommendation based on hybrid scan detection
            val offHeapSizeMB = recomMemorySettings.sparkOffHeapMem.getOrElse(0L)
            if (offHeapSizeMB > 0) {
              appendRecommendationForMemoryMB("spark.memory.offHeap.size", s"$offHeapSizeMB")
              // Enable off-heap memory if we're recommending a size
              appendRecommendation("spark.memory.offHeap.enabled", "true")

              // Calculate host off-heap limit size for onPrem platform only when
              // offHeapLimit is enabled
              if (!platform.isPlatformCSP && isOffHeapLimitUserEnabled) {
                val hostOffHeapLimitSizeMB = recomMemorySettings.executorMemOverhead.get +
                  offHeapSizeMB - nonExecutorMemory
                if (hostOffHeapLimitSizeMB > 0) {
                  appendRecommendationForMemoryMB("spark.rapids.memory.host.offHeapLimit.size",
                    s"$hostOffHeapLimitSizeMB")
                }
              }
            }
          case Left(notEnoughMemComment) =>
            // Not enough memory available, add warning comments
            appendComment(notEnoughMemComment)
            // Helper function to append not enough memory comment for a specific key
            def appendCommentForNotEnoughMem(key: String): Unit = {
              appendComment(key, notEnoughMemCommentForKey(key), prependKey = false)
              // Mark the recommendation as unresolved since AutoTuner could not recommend a value
              markAsUnresolved(key)
            }
            appendCommentForNotEnoughMem("spark.rapids.memory.pinnedPool.size")
            if (sparkMaster.contains(Yarn) || sparkMaster.contains(Kubernetes)) {
              appendCommentForNotEnoughMem("spark.executor.memoryOverhead")
            }
            appendCommentForNotEnoughMem("spark.executor.memory")
            // Skip off-heap related comments when offHeapLimit is enabled
            if (!platform.isPlatformCSP && isOffHeapLimitUserEnabled) {
              appendCommentForNotEnoughMem("spark.memory.offHeap.size")
              appendCommentForNotEnoughMem("spark.rapids.memory.host.offHeapLimit.size")
            }
        }
      } else {
        logInfo("Available memory per exec is not specified")
        addMissingMemoryComments()
      }
      configureShuffleReaderWriterNumThreads(execCores)
      configureMultiThreadedReaders(execCores)
      recommendDynamicAllocationConfigs(execCores)
      // TODO: Should we recommend AQE even if cluster properties are not enabled?
      recommendAQEProperties()
    } else {
      addDefaultComments()
    }
    appendRecommendation("spark.rapids.sql.batchSizeBytes",
      configProvider.getEntry("BATCH_SIZE_BYTES").getDefault)
    appendRecommendation("spark.locality.wait",
      configProvider.getEntry("LOCALITY_WAIT").getDefault)
  }

  def calculateJobLevelRecommendations(): Unit = {
    // TODO - do we do anything with 200 shuffle partitions or maybe if its close
    // set the Spark config  spark.shuffle.sort.bypassMergeThreshold
    if (platform.getUserEnforcedSparkProperty("spark.shuffle.manager").isEmpty) {
      // Process shuffle manager only if not user-enforced
      getShuffleManagerClassName match {
        case Right(smClassName) => appendRecommendation("spark.shuffle.manager", smClassName)
        case Left(comment) => appendComment("spark.shuffle.manager", comment, prependKey = false)
      }
    }
    appendComment(classPathComments("rapids.shuffle.jars"))
    recommendFileCache()
    recommendMaxPartitionBytes()
    // Recommend shuffle partitions here.
    // Note that this may get overridden if AQE is enabled.
    recommendShufflePartitions()
    recommendKryoSerializerSetting()
    recommendGCProperty()
    if (platform.requirePathRecommendations) {
      recommendClassPathEntries()
    }
    recommendSystemProperties()
  }

  // if the user set the serializer to use Kryo, make sure we recommend using the GPU version
  // of it.
  def recommendKryoSerializerSetting(): Unit = {
    getPropertyValue("spark.serializer")
      .filter(_.contains("org.apache.spark.serializer.KryoSerializer")).foreach { _ =>
      // Recommend adding the GPU Kryo registrator if not already present
      recommendClassNameProperty("spark.kryo.registrator",
        autoTunerHelper.gpuKryoRegistratorClassName)
      // set the kryo serializer buffer size to prevent OOMs
      val desiredBufferMaxMB =
        configProvider.getEntry("KRYO_SERIALIZER_BUFFER").getMaxAsMemory(ByteUnit.MiB)
      val currentBufferMaxMB = getPropertyValue("spark.kryoserializer.buffer.max")
        .map(StringUtils.convertToMB(_, Some(ByteUnit.MiB)))
        .getOrElse(0L)
      if (currentBufferMaxMB < desiredBufferMaxMB) {
        appendRecommendationForMemoryMB("spark.kryoserializer.buffer.max", s"$desiredBufferMaxMB")
      }
    }
  }

  /**
   * Resolves the RapidsShuffleManager class name based on the Spark version.
   * If a valid class name is not found, an error message is returned.
   *
   * Example:
   * sparkVersion: "3.2.0-amzn-1"
   * return: Right("com.nvidia.spark.rapids.spark320.RapidsShuffleManager")
   *
   * sparkVersion: "3.1.2"
   * return: Left("Cannot recommend RAPIDS Shuffle Manager for unsupported '3.1.2' version.")
   *
   * @return Either an error message (Left) or the RapidsShuffleManager class name (Right)
   */
  def getShuffleManagerClassName: Either[String, String] = {
    appInfoProvider.getSparkVersion match {
      case Some(sparkVersion) =>
        platform.getShuffleManagerVersion(sparkVersion) match {
          case Some(smVersion) =>
            Right(autoTunerHelper.buildShuffleManagerClassName(smVersion))
          case None =>
            Left(shuffleManagerCommentForUnsupportedVersion(
              sparkVersion, platform))
        }
      case None =>
        Left(shuffleManagerCommentForMissingVersion)
    }
  }

  private def recommendGCProperty(): Unit = {
    val jvmGCFraction = appInfoProvider.getJvmGCFractions
    if (jvmGCFraction.nonEmpty) { // avoid zero division
      if ((jvmGCFraction.sum / jvmGCFraction.size) >
        configProvider.getEntry("JVM_GCTIME_FRACTION").getMax.toDouble) {
        // TODO - or other cores/memory ratio
        appendComment("Average JVM GC time is very high. " +
          "Other Garbage Collectors can be used for better performance.")
      }
    }
  }

  protected def recommendAQEProperties(): Unit = {
    // Spark configuration (AQE is enabled by default)
    val aqeEnabled = getPropertyValue("spark.sql.adaptive.enabled")
      .getOrElse("true").toLowerCase
    if (aqeEnabled == "false") {
      // TODO: Should we recommend enabling AQE if not set?
      appendComment(commentsForMissingProps("spark.sql.adaptive.enabled"))
    }
    appInfoProvider.getSparkVersion match {
      case Some(version) =>
        if (ToolUtils.isSpark320OrLater(version)) {
          // AQE configs changed in 3.2.0
          if (getPropertyValue("spark.sql.adaptive.coalescePartitions.minPartitionSize").isEmpty) {
            appendRecommendation("spark.sql.adaptive.coalescePartitions.minPartitionSize",
              configProvider.getEntry("AQE_MIN_PARTITION_SIZE").getDefault)
          }
        } else {
          if (getPropertyValue("spark.sql.adaptive.coalescePartitions.minPartitionNum").isEmpty) {
            // TODO: Should this be based on the recommended
            //  cluster instead of the cluster info from the event log
            // The ideal setting is same as the parallelism of the cluster
            platform.clusterInfoFromEventLog match {
              case Some(clusterInfo) =>
                // Use cluster info from event log to calculate total parallelism
                val total = clusterInfo.numExecutors * clusterInfo.coresPerExecutor
                appendRecommendation("spark.sql.adaptive.coalescePartitions.minPartitionNum",
                  total.toString)
              case None =>
            }
          }
        }
      case None =>
    }

    val aqeInputSizeThresholdBytes = configProvider.getEntry("AQE_INPUT_SIZE_THRESHOLD")
      .getDefaultAsMemory(ByteUnit.BYTE)
    val advisoryPartitionSizeProperty =
      getPropertyValue("spark.sql.adaptive.advisoryPartitionSizeInBytes")
    if (appInfoProvider.getMeanInput < aqeInputSizeThresholdBytes) {
      if (advisoryPartitionSizeProperty.isEmpty) {
        // get the default advisory partition size from the tuning config
        appendRecommendation("spark.sql.adaptive.advisoryPartitionSizeInBytes",
          configProvider.getEntry("AQE_ADVISORY_PARTITION_SIZE").getDefault)
      }
    }
    var recInitialPartitionNum = 0
    val aqeShuffleReadBytesThresholdBytes = configProvider.getEntry("AQE_SHUFFLE_READ_THRESHOLD")
      .getDefaultAsMemory(ByteUnit.BYTE)
    if (appInfoProvider.getMeanInput > aqeInputSizeThresholdBytes &&
      appInfoProvider.getMeanShuffleRead > aqeShuffleReadBytesThresholdBytes) {
      // AQE Recommendations for large input and large shuffle reads
      platform.recommendedGpuDevice.getAdvisoryPartitionSizeInBytes.foreach { size =>
        appendRecommendation("spark.sql.adaptive.advisoryPartitionSizeInBytes", size)
      }
      if (shufflePartitionValue <=
            configProvider.getEntry("AQE_MIN_INITIAL_PARTITION_NUM").getDefault.toInt) {
        recInitialPartitionNum = platform.recommendedGpuDevice.getInitialPartitionNum.getOrElse(0)
      }
      appendRecommendation("spark.sql.adaptive.coalescePartitions.parallelismFirst",
        configProvider.getEntry("AQE_COALESCE_PARALLELISM_FIRST").getDefault)
    }

    // Update the recommended shuffle partitions (both AQE initial partition number
    // and shuffle partitions) based on the AQE recommendations and ColumnarExchange data size
    val isSkipped = applyToAllPartitionProperties[Boolean](skippedRecommendations.contains(_))
      .exists(identity)
    if (!isSkipped) {
      var finalPartitionValue = Math.max(shufflePartitionValue, recInitialPartitionNum)
      // Adjust based on ColumnarExchange data size if available
      aqePartitionProperty.foreach { initialPartitionNumKey =>
        appInfoProvider.getMaxColumnarExchangeDataSizeBytes match {
          case Some(maxDataSize) =>
            // Get GPU batch size (use actual value if set, otherwise use default)
            val gpuBatchSize: Long = getPropertyValue("spark.rapids.sql.batchSizeBytes") match {
              case Some(value) =>
                // Parse the actual batch size value (could be with units like "2g", "1GB", etc.)
                StringUtils.convertMemorySizeToBytes(value, Some(ByteUnit.BYTE))
              case None =>
                // Use default batch size from tuning configs
                configProvider.getEntry("BATCH_SIZE_BYTES").getDefaultAsMemory(ByteUnit.BYTE)
            }
            // Calculate ratio of ColumnarExchange data to GPU batch size.
            // Only increase finalPartitionValue since ColumnarExchange is GPU-only
            // and doesn't capture CPU shuffle data.
            val columnarExchangeRatio = (maxDataSize.toDouble / gpuBatchSize).ceil.toInt
            if (columnarExchangeRatio > finalPartitionValue) {
              appendComment(s"'$initialPartitionNumKey' adjusted from " +
                s"$finalPartitionValue to $columnarExchangeRatio based on " +
                s"ColumnarExchange data size (${maxDataSize} bytes) and " +
                s"GPU batch size (${gpuBatchSize} bytes)")
              finalPartitionValue = columnarExchangeRatio
            }
          case None =>
            // No ColumnarExchange data size metrics found, use original logic
        }
      }
      aqePartitionProperty.foreach(appendRecommendation(_, finalPartitionValue))
      appendRecommendation("spark.sql.shuffle.partitions", finalPartitionValue)
    }

    // Handle Databricks-specific AQE auto shuffle
    if (platform.isInstanceOf[DatabricksPlatform]) {
      val aqeAutoShuffle = getPropertyValue("spark.databricks.adaptive.autoOptimizeShuffle.enabled")
      if (aqeAutoShuffle.isDefined) {
        // If the user has enabled AQE auto shuffle, override with the default
        // recommendation for that property.
        appendRecommendation("spark.databricks.adaptive.autoOptimizeShuffle.enabled",
          configProvider.getEntry("DATABRICKS_AUTO_OPTIMIZE_SHUFFLE_ENABLED").getDefault)
      }
    }

    // TODO - can we set spark.sql.autoBroadcastJoinThreshold ???
    val autoBroadcastJoinKey = "spark.sql.adaptive.autoBroadcastJoinThreshold"
    val autoBroadcastJoinThresholdPropertyMB =
      getPropertyValue(autoBroadcastJoinKey).map(StringUtils.convertToMB(_, Some(ByteUnit.BYTE)))
    val autoBroadcastJoinThresholdDefaultMB =
      configProvider.getEntry("AQE_AUTO_BROADCAST_JOIN_THRESHOLD").getDefaultAsMemory(ByteUnit.MiB)
    // If the property is not set, append a missing comment and mark it
    // as unresolved so the user knows to look at it.
    if (autoBroadcastJoinThresholdPropertyMB.isEmpty) {
      appendMissingComment(autoBroadcastJoinKey)
      markAsUnresolved(autoBroadcastJoinKey)
    } else if (autoBroadcastJoinThresholdPropertyMB.get > autoBroadcastJoinThresholdDefaultMB) {
      appendComment(s"Setting '$autoBroadcastJoinKey' > ${autoBroadcastJoinThresholdDefaultMB}m" +
        s" could lead to performance\n" +
        "  regression. Should be set to a lower number.")
    }
  }

  /**
   * Checks the system properties and give feedback to the user.
   * For example file.encoding=UTF-8 is required for some ops like GpuRegEX.
   */
  private def recommendSystemProperties(): Unit = {
    appInfoProvider.getSystemProperty("file.encoding").collect {
      case encoding if !ToolUtils.isFileEncodingRecommended(encoding) =>
        appendComment(s"file.encoding should be [${ToolUtils.SUPPORTED_ENCODINGS.mkString}]" +
            " because GPU only supports the charset when using some expressions.")
    }
  }

  /**
   * Check the class path entries with the following rules:
   * 1- If ".*rapids-4-spark.*jar" is missing then add a comment that the latest jar should be
   *    included in the classpath unless it is part of the spark
   * 2- If there are more than 1 entry for ".*rapids-4-spark.*jar", then add a comment that
   *    there should be only 1 jar in the class path.
   * 3- If there are cudf jars, ignore that for now.
   * 4- If there is a new release recommend that to the user
   */
  private def recommendClassPathEntries(): Unit = {
    val missingRapidsJarsEntry = classPathComments("rapids.jars.missing")
    val multipleRapidsJarsEntry = classPathComments("rapids.jars.multiple")

    appInfoProvider.getRapidsJars match {
      case Seq() =>
        // No rapids jars
        appendComment(missingRapidsJarsEntry)
      case s: Seq[String] =>
        s.flatMap(e =>
          autoTunerHelper.pluginJarRegEx.findAllMatchIn(e).map(_.group(1))) match {
            case Seq() => appendComment(missingRapidsJarsEntry)
            case v: Seq[String] if v.length > 1 =>
              val comment = s"$multipleRapidsJarsEntry [${v.mkString(", ")}]"
              appendComment(comment)
            case Seq(jarVer) =>
              // compare jarVersion to the latest release
              val latestPluginVersion = WebCrawlerUtil.getLatestPluginRelease
              latestPluginVersion match {
                case Some(ver) =>
                  if (ToolUtils.compareVersions(jarVer, ver).exists(_ < 0)) {
                    val jarURL = WebCrawlerUtil.getPluginMvnDownloadLink(ver)
                    appendComment(
                      "A newer RAPIDS Accelerator for Apache Spark plugin is available:\n" +
                        s"  $jarURL\n" +
                        s"  Version used in application is $jarVer.")
                  }
                case None =>
                  logError("Could not pull the latest release of RAPIDS-plugin jar.")
                  val pluginRepoUrl = WebCrawlerUtil.getMVNArtifactURL("rapids.plugin")
                  appendComment(
                    "Failed to validate the latest release of Apache Spark plugin.\n" +
                    s"  Verify that the version used in application ($jarVer) is the latest on:\n" +
                    s"  $pluginRepoUrl")

            }
        }
    }
  }

  /**
   * Calculate max partition bytes using the max task input size and existing setting
   * for maxPartitionBytes. Note that this won't apply the same on iceberg.
   * The max bytes here does not distinguish between GPU and CPU reads so we could
   * improve that in the future.
   * Eg,
   * TASK_INPUT_SIZE_THRESHOLD = {min: 128m, max: 256m}
   * (1) Input:  currentMaxPartitionBytes = 512m
   *             actualTaskInputSizeMB = 12m (below min threshold -> increase maxPartitionBytes)
   *     Output: recommendedMaxPartitionBytes = 512m * (128m/12m) = 4g (hit max value)
   * (2) Input:  currentMaxPartitionBytes = 2g
   *             actualTaskInputSizeMB = 512m (above max threshold -> decrease maxPartitionBytes)
   *     Output: recommendedMaxPartitionBytes = 2g / (512m/128m) = 512m
   */
  protected def calculateMaxPartitionBytesInMB(currentMaxPartitionBytes: String): Option[Long] = {
    // AutoTuner only supports a single app right now, so we get whatever value is here
    val actualTaskInputSizeMB = appInfoProvider.getMaxInput / 1024 / 1024
    val currentMaxPartitionBytesMB = StringUtils.convertToMB(
      currentMaxPartitionBytes, Some(ByteUnit.BYTE))
    // Get the min and max thresholds for the task input size
    val taskInputSizeThreshold = configProvider.getEntry("TASK_INPUT_SIZE_THRESHOLD")
    val minTaskInputSizeThresholdMB = taskInputSizeThreshold.getMinAsMemory(ByteUnit.MiB)
    val maxTaskInputSizeThresholdMB = taskInputSizeThreshold.getMaxAsMemory(ByteUnit.MiB)
    // Get the upper bound for the max partition bytes
    val maxAllowedPartitionBytesMB =
      configProvider.getEntry("MAX_PARTITION_BYTES").getMaxAsMemory(ByteUnit.MiB)

    if (actualTaskInputSizeMB == 0.0) {
      Some(currentMaxPartitionBytesMB)
    } else {
      if (actualTaskInputSizeMB > 0 && actualTaskInputSizeMB < minTaskInputSizeThresholdMB) {
        // If task input too small (< min threshold): increase partition size to get bigger tasks
        val recommendedMaxPartitionBytesMB = Math.min(
          currentMaxPartitionBytesMB * (minTaskInputSizeThresholdMB / actualTaskInputSizeMB),
          maxAllowedPartitionBytesMB)
        Some(recommendedMaxPartitionBytesMB.toLong)
      } else if (actualTaskInputSizeMB > maxTaskInputSizeThresholdMB) {
        //  If task input too large (> max threshold): decrease partition size to get smaller tasks
        val recommendedMaxPartitionBytesMB = Math.min(
          currentMaxPartitionBytesMB / (actualTaskInputSizeMB / maxTaskInputSizeThresholdMB),
          maxAllowedPartitionBytesMB)
        Some(recommendedMaxPartitionBytesMB.toLong)
      } else {
        // If task input within range: no adjustment needed
        None
      }
    }
  }

  /**
   * Recommendation for 'spark.rapids.file.cache' based on read characteristics of job.
   */
  private def recommendFileCache(): Unit = {
    if (appInfoProvider.getDistinctLocationPct <
        configProvider.getEntry("DISTINCT_READ_THRESHOLD").getDefault.toDouble &&
      appInfoProvider.getRedundantReadSize >
        configProvider.getEntry("READ_SIZE_THRESHOLD").getDefaultAsMemory(ByteUnit.BYTE)) {
      appendRecommendation("spark.rapids.filecache.enabled",
        configProvider.getEntry("FILE_CACHE_ENABLED").getDefault)
      appendComment("Enable file cache only if Spark local disks bandwidth is > 1 GB/s" +
        " and you have sufficient disk space available to fit both cache and normal Spark" +
        " temporary data.")
    }
  }

  /**
   * Recommendation for 'spark.sql.files.maxPartitionBytes' based on input size for each task.
   * Note that the logic can be disabled by adding the property to "limitedLogicRecommendations"
   * which is one of the arguments of [[getRecommendedProperties]].
   */
  private def recommendMaxPartitionBytes(): Unit = {
    val maxPartitionProp =
      getPropertyValue("spark.sql.files.maxPartitionBytes")
        .getOrElse(configProvider.getEntry("MAX_PARTITION_BYTES").getDefault)
    val recommended =
      if (isCalculationEnabled("spark.sql.files.maxPartitionBytes")) {
        calculateMaxPartitionBytesInMB(maxPartitionProp).map(_.toString).orNull
      } else {
        s"${StringUtils.convertToMB(maxPartitionProp, Some(ByteUnit.BYTE))}"
      }
    appendRecommendationForMemoryMB("spark.sql.files.maxPartitionBytes", recommended)
  }

  /**
   * Internal method to recommend 'spark.sql.shuffle.partitions' based on spills and skew.
   * This method can be overridden by Profiling/Qualification AutoTuners to provide custom logic.
   */
  protected def recommendShufflePartitionsInternal(): Int = {
    var inputShufflePartitions = shufflePartitionValue
    val shuffleStagesWithPosSpilling = appInfoProvider.getShuffleStagesWithPosSpilling
    if (shuffleStagesWithPosSpilling.nonEmpty) {
      val shuffleSkewStages = appInfoProvider.getShuffleSkewStages
      if (shuffleSkewStages.exists(id => shuffleStagesWithPosSpilling.contains(id))) {
        appendComment(
          "Shuffle skew exists (when task's Shuffle Read Size > 3 * Avg Stage-level size) in\n" +
            s"  stages with spilling. Increasing shuffle partitions is not recommended in this\n" +
            s"  case since keys will still hash to the same task.")
      } else {
        inputShufflePartitions *=
          configProvider.getEntry("SHUFFLE_PARTITION_MULTIPLIER").getDefault.toInt
        // Could be memory instead of partitions
        appendComment(shufflePartitionsCommentForSpilling)
      }
    }
    inputShufflePartitions
  }

  /**
   * Coordinates recommendations for partition-related properties to ensure
   * they are properly aligned.
   *
   * This method determines the appropriate partition property to set based on:
   * - AQE initial partition requirements
   * - Shuffle spills and skew requirements
   * - Whether AQE coalescing is enabled
   *
   * Uses a unified approach to determine which property should be set rather than
   * setting multiple conflicting properties.
   */
  private def recommendShufflePartitions(): Unit = {
    val isSkipped = applyToAllPartitionProperties[Boolean](skippedRecommendations.contains(_))
      .exists(identity)
    if (!isSkipped) {
      // Apply shuffle-specific logic (spills, skew) if calculation is enabled
      // on all partition properties (i.e. spark.sql.shuffle.partitions and AQE initial partition)
      val isCalcEnabled = applyToAllPartitionProperties[Boolean](isCalculationEnabled(_))
        .forall(identity)
      val recommendedShufflePartitions = if (isCalcEnabled) {
        recommendShufflePartitionsInternal()
      } else {
        shufflePartitionValue
      }
      appendRecommendation("spark.sql.shuffle.partitions", recommendedShufflePartitions)
    }
  }

  /**
   * Analyzes unsupported driver logs and generates recommendations for configuration properties.
   */
  private def recommendFromDriverLogs(): Unit = {
    // Iterate through unsupported operators' reasons and check for matching properties
    driverInfoProvider.getUnsupportedOperators.map(_.reason).foreach { operatorReason =>
      autoTunerHelper.unsupportedOperatorRecommendations.collect {
        case (config, recommendedValue) if operatorReason.contains(config) =>
          appendRecommendation(config, recommendedValue)
          appendComment(commentForExperimentalConfig(config))
      }
    }
  }

  /**
   * Internal method to recommend plugin properties based on the Tool.
   */
  protected def recommendPluginPropsInternal(): Unit

  private def recommendPluginProps(): Unit = {
    val isRapidsPluginConfigured = getPropertyValue("spark.plugins") match {
      case Some(f) => f.contains(autoTunerHelper.rapidsPluginClassName)
      case None => false
    }
    if (!isRapidsPluginConfigured) {
      recommendPluginPropsInternal()
    }
    // Always recommend setting 'spark.rapids.sql.enabled=true', regardless of current setting.
    appendRecommendation("spark.rapids.sql.enabled", "true")
  }

  /**
   * Recommend additional executor resource GPU properties.
   * - spark.executor.resource.gpu.amount: recommended if unset or set to 0
   * - spark.executor.resource.gpu.discoveryScript: comment if YARN, k8s or Standalone (On-Prem)
   * - spark.executor.resource.gpu.vendor: recommended if k8s (On-Prem)
   */
  private def recommendExecutorResourceGpuProps(): Unit = {
    val gpuAmountKey = "spark.executor.resource.gpu.amount"
    val gpuAmountValueOpt = getPropertyValue(gpuAmountKey)
    val isUnsetOrZero = gpuAmountValueOpt.forall { v =>
      v.trim.isEmpty || scala.util.Try(v.toLong).toOption.contains(0L)
    }
    if (isUnsetOrZero) {
      val recommendedGpuAmount =
        configProvider.getEntry("EXECUTOR_GPU_RESOURCE_AMT").getDefault
      appendRecommendation(gpuAmountKey, recommendedGpuAmount)
    }

    // Include additional executor resource GPU properties for On-Prem
    // Avoid recommending these for CSPs as they are handled by the platform.
    if (!platform.isPlatformCSP) {
      // If YARN,Kubernetes or Standalone, recommend GPU discovery script
      // See: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
      val isYarnK8sOrStandalone = sparkMaster.exists {
        case Yarn | Kubernetes | Standalone => true
        case _ => false
      }
      // If the GPU discovery script is not set or is empty
      val gpuDiscoveryScriptIsMissing =
        getPropertyValue("spark.executor.resource.gpu.discoveryScript")
        .forall(_.trim.isEmpty)
      if (isYarnK8sOrStandalone && gpuDiscoveryScriptIsMissing) {
        appendComment(missingGpuDiscoveryScriptComment)
      }

      // For Kubernetes, recommend setting the GPU vendor property
      if (sparkMaster.contains(Kubernetes)) {
        appendRecommendation(
          "spark.executor.resource.gpu.vendor",
          autoTunerHelper.kubernetesGpuVendor
        )
      }
    }
  }

  def appendOptionalComment(lookup: String, comment: String): Unit = {
    if (!skippedRecommendations.contains(lookup)) {
      appendComment(comment)
    }
  }

  /**
   * Recommend class name properties (like spark.plugins, spark.kryo.registrator).
   * Logic:
   * - Trim whitespace, filter out empty entries and remove duplicates.
   * - Finally, append the specified class name to the existing set if not already present
   * Note:
   *  - ListSet preserves the original order of class names
   *
   * @param propertyKey The Spark property key to update
   * @param className The class name to add if missing
   * @return True if a recommendation was made, false if the class name was already present
   */
  protected def recommendClassNameProperty(propertyKey: String, className: String): Boolean = {
    val existingClasses = scala.collection.immutable.ListSet(
      getPropertyValue(propertyKey)
        .map(v => v.split(",").map(_.trim).filter(_.nonEmpty))
        .getOrElse(Array.empty): _*)

    if (!existingClasses.contains(className)) {
      appendRecommendation(propertyKey, (existingClasses + className).mkString(","))
      true
    } else {
      false
    }
  }

  def appendComment(comment: String): Unit = {
    comments += comment
  }

  /**
   * Adds a comment for a configuration key.
   */
  def appendComment(
      key: String,
      comment: String,
      prependKey: Boolean = true): Unit = {
    if (!skippedRecommendations.contains(key)) {
      val finalComment = if (prependKey) {
        s"'$key' $comment"
      } else {
        comment
      }
      appendComment(finalComment)
    }
  }
  /**
   * Adds a comment for a configuration key by looking up the comment
   * from the lookup table based on the source field.
   * This is useful when the caller does not want to hardcode the comment.
   * @param key The configuration key
   * @param sourceField The source field to lookup the comment from. It can be one of:
   *                    "missing", "updated", "persistent", "description. Default is "persistent".
   * @param prependKey Whether to prepend the key to the comment. Default is true.
   */
  def appendCommentFromLookup(
      key: String,
      sourceField: String = "persistent",
      prependKey: Boolean = true): Unit = {
    if (!skippedRecommendations.contains(key)) {
      // Get the comment from the lookup table
      sourceField match {
        case "missing" =>
          appendMissingComment(key)
        case "updated" =>
          appendUpdatedComment(key)
        case "persistent" =>
          appendPersistentComment(key)
        case "description" =>
          appendDescriptionAsComment(key)
        case _ =>
          // Do nothing as we cannot find the source field
      }
    }
  }

  /**
   * Add default comments for missing properties except the ones
   * which should be skipped.
   */
  private def addDefaultComments(): Unit = {
    appendComment("Could not infer the cluster configuration, recommendations " +
      "are generated using default values!")
    commentsForMissingProps.foreach {
      case (key, value) =>
        if (!skippedRecommendations.contains(key)) {
          appendComment(value)
        }
    }
  }

  private def addMissingMemoryComments(): Unit = {
    commentsForMissingMemoryProps.foreach {
      case (key, value) =>
        if (!skippedRecommendations.contains(key)) {
          appendComment(value)
        }
    }
  }

  private def toCommentProfileResult: Seq[RecommendedCommentResult] = {
    comments.map(RecommendedCommentResult).toSeq.sortBy(_.comment)
  }

  private def toRecommendationsProfileResult: Seq[TuningEntryTrait] = {
    recommendations.values.filter(shouldIncludeInFinalRecommendations).toSeq.sortBy(_.name)
  }

  protected def finalizeTuning(): Unit = {
    recommendations.values.foreach(_.commit())
  }

  /**
   * The Autotuner loads the spark properties from the eventlog.
   * 1- runs the calculation for each criterion and saves it as a [[TuningEntryTrait]].
   * 2- The final list of recommendations include any [[TuningEntryTrait]] that has a
   *    recommendation that is different from the original property.
   * 3- Null values are excluded.
   * 4- A comment is added for each missing property in the spark property.
   *
   * @param skipList a list of properties to be skipped. If none, all recommendations are
   *                 returned. Note that the recommendations will be computed anyway internally
   *                 in case there are dependencies between the recommendations.
   *                 Default is empty.
   * @param limitedLogicList a list of properties that will do simple recommendations based on
   *                         static default values.
   * @param showOnlyUpdatedProps When enabled, the profiler recommendations should only include
   *                             updated settings.
   * @return pair of recommendations and comments. Both sequence can be empty.
   */
  def getRecommendedProperties(
      skipList: Option[Seq[String]] = Some(Seq()),
      limitedLogicList: Option[Seq[String]] = Some(Seq()),
      showOnlyUpdatedProps: Boolean = true):
      (Seq[TuningEntryTrait], Seq[RecommendedCommentResult]) = {
    if (appInfoProvider.isAppInfoAvailable) {
      limitedLogicList.foreach(limitedSeq => limitedLogicRecommendations ++= limitedSeq)
      platform.targetCluster.foreach { cluster =>
        cluster.getSparkProperties.preservePropertiesSet.foreach { property =>
          getPropertyValueFromSource(property) match {
            case Some(_) =>
              // If the property is found in the source properties, add a comment and
              // add the property to the limited logic recommendations.
              appendComment(getPreservedPropertyComment(property))
              limitedLogicRecommendations += property
            case None =>
              appendComment(getPreservedPropertyNotFoundComment(property))
          }
        }
      }
      skipList.foreach(skipSeq => skippedRecommendations ++= skipSeq)
      skippedRecommendations ++= platform.recommendationsToExclude
      platform.targetCluster.foreach { cluster =>
        cluster.getSparkProperties.excludePropertiesSet.foreach { property =>
          appendComment(getExcludedPropertyComment(property))
          skippedRecommendations += property
        }
      }
      initRecommendations()
      // configured GPU recommended instance type NEEDS to happen before any of the other
      // recommendations as they are based on
      // the instance type
      configureGPURecommendedInstanceType()
      // Makes recommendations based on information extracted from the AppInfoProvider
      filterByUpdatedPropertiesEnabled = showOnlyUpdatedProps
      executeTuningPlugins()
      recommendPluginProps()
      calculateJobLevelRecommendations()
      calculateClusterLevelRecommendations()

      // Add all platform specific recommendations
      platform.platformSpecificRecommendations.collect {
        case (property, value) if getPropertyValueFromSource(property).isEmpty =>
          appendRecommendation(property, value)
      }
    }
    recommendFromDriverLogs()
    finalizeTuning()
    (toRecommendationsProfileResult, toCommentProfileResult)
  }

  // Process the properties keys. This is needed in case there are some properties that should not
  // be listed in the final combined results. For example:
  // - The UUID of the app is not part of the submitted spark configurations
  // - make sure that we exclude the skipped list
  private def processPropKeys(
      srcMap: collection.Map[String, String]): collection.Map[String, String] = {
    srcMap -- skippedRecommendations
  }

  // Combines the original Spark properties with the recommended ones.
  def combineSparkProperties(
      recommendedSet: Seq[TuningEntryTrait]): Seq[RecommendedPropertyResult] = {
    // get the original properties after filtering and removing unnecessary keys
    val originalPropsFiltered = processPropKeys(getAllSourceProperties)
    // Combine the original properties with the recommended properties.
    // The recommendations should always override the original ones
    val combinedProps = (originalPropsFiltered
      ++ recommendedSet.map(r => r.name -> r.getTuneValue()).toMap).toSeq.sortBy(_._1)
    combinedProps.collect {
      case (pK, pV) => RecommendedPropertyResult(pK, pV)
    }
  }

  protected lazy val aqePartitionProperty: Option[String] = {
    val aqeEnabled = getPropertyValue("spark.sql.adaptive.enabled")
      .getOrElse("true").toLowerCase == "true" // enabled by default in Spark
    val coalesceEnabled = getPropertyValue("spark.sql.adaptive.coalescePartitions.enabled")
      .getOrElse("true").toLowerCase == "true" // enabled by default when AQE is enabled

    if (aqeEnabled && coalesceEnabled) {
      val maxNumPostShufflePartitions = "spark.sql.adaptive.maxNumPostShufflePartitions"
      val initialPartitionNumKey = "spark.sql.adaptive.coalescePartitions.initialPartitionNum"
      // check if maxNumPostShufflePartitions is in final tuning table
      if (finalTuningTable.contains(maxNumPostShufflePartitions)) {
        Some(maxNumPostShufflePartitions)
      } else {
        Some(initialPartitionNumKey)
      }
    } else {
      None
    }
  }

  /**
   * Applies a function to all relevant shuffle partition property keys, based on AQE and
   * coalescing settings.
   *
   * Logic:
   *   - If AQE and coalescing are enabled, the relevant property is either
   *     'spark.sql.adaptive.maxNumPostShufflePartitions' or
   *     'spark.sql.adaptive.coalescePartitions.initialPartitionNum', depending on which is present
   *     in the final tuning table.
   *   - In all cases, 'spark.sql.shuffle.partitions' is also included as a relevant property.
   *
   * @param fn Function to apply to each selected property key.
   * @tparam T Return type of the function.
   * @return A sequence of results, one for each relevant property key.
   */
  protected def applyToAllPartitionProperties[T](fn: String => T): Seq[T] = {
    (aqePartitionProperty.toSeq :+ "spark.sql.shuffle.partitions").map(fn)
  }

  /**
   * Returns the shuffle partition value using the following logic:
   * - Considers all relevant partition properties (AQE initial partition properties and
   *   'spark.sql.shuffle.partitions').
   * - For each property, checks if a value is set (either in the event log or in recommendations).
   * - If multiple properties have values, returns the maximum value among them.
   * - If none are set, returns the default value from tuningConfigs.
   *
   * Note: This is a method (not a lazy val) to always reflect the latest value,
   * as recommendations may be updated after initial evaluation.
   */
  protected def shufflePartitionValue: Int = {
    // Gather all relevant partition property values, take the maximum if multiple are set
    applyToAllPartitionProperties[Option[Int]](prop => getPropertyValue(prop).map(_.toInt))
      .flatten
      .reduceOption(_ max _)
      .getOrElse(configProvider.getEntry("SHUFFLE_PARTITIONS").getDefault.toInt)
  }

  /**
   * Check if the application is using hybrid scan mode based on the following three properties:
   * - 'spark.sql.sources.useV1SourceList': 'parquet'
   * - 'spark.rapids.sql.hybrid.parquet.enabled': 'true'
   * - 'spark.rapids.sql.hybrid.loadBackend': 'false'
   *
   * @return true if all three conditions are met for hybrid scan
   */
  private def isHybridScanEnabled(): Boolean = {
    val useV1SourceList = getPropertyValue("spark.sql.sources.useV1SourceList").getOrElse("")
    val hybridParquetEnabled = getPropertyValue("spark.rapids.sql.hybrid.parquet.enabled")
      .getOrElse("false")
    val hybridLoadBackend = getPropertyValue("spark.rapids.sql.hybrid.loadBackend")
      .getOrElse("false")

    useV1SourceList.contains("parquet") &&
      hybridParquetEnabled.toLowerCase == "true" &&
      hybridLoadBackend.toLowerCase == "false"
  }

  /**
   * Calculate recommended off-heap memory size based on hybrid scan detection.
   * If hybrid scan is enabled and platform is onPrem, set off-heap size to OFFHEAP_PER_CORE *
   * executor cores.
   * Otherwise, use the existing logic from platform.getSparkOffHeapMemoryMB.
   *
   * @param numExecutorCores Number of executor cores
   * @return Recommended off-heap memory size in MB
   */
  private def calculateOffHeapMemorySize(numExecutorCores: Int): Long = {
    if (!platform.isPlatformCSP && isHybridScanEnabled()) {
      // For onPrem platform with hybrid scan, set off-heap size to
      // OFFHEAP_PER_CORE * executor cores
      // Hybrid scan will require more off-heap memory than the default value.
      configProvider.getEntry("OFFHEAP_PER_CORE")
        .getDefaultAsMemory(ByteUnit.MiB) * numExecutorCores
    } else {
      // Use existing logic for CSP platforms or non-hybrid scan
      platform.getSparkOffHeapMemoryMB(getPropertyValue).getOrElse(0L)
    }
  }

  /**
   * Calculate recommended pinned memory size using the new formula:
   * pinned pool-offHeap ratio * host.offHeapLimit.Size for onPrem platform.
   *
   * Note: This new formula is only used for onPrem platform.
   * For CSP platforms, the original calculation is used.
   *
   * @param numExecutorCores Number of executor cores
   * @param hostOffHeapLimitSizeMB Host off-heap limit size in MB
   * @return Recommended pinned memory size in MB
   */
  private def calculatePinnedMemorySize(numExecutorCores: Int,
                                        hostOffHeapLimitSizeMB: Long): Long = {
    // Use new formula only for onPrem platform
    if (!platform.isPlatformCSP && isOffHeapLimitUserEnabled) {
      // Calculate pinned pool-offHeap ratio * host.offHeapLimit.Size
      val ratioPinnedPoolSize = hostOffHeapLimitSizeMB *
        configProvider.getEntry("PINNED_MEM_OFFHEAP_RATIO").getDefault.toDouble
      // Return the minimum of the two values
      ratioPinnedPoolSize.toLong
    } else {
      // For CSP platforms, return a default value (this will be overridden by the original logic)
      configProvider.getEntry("PINNED_MEMORY").getDefaultAsMemory(ByteUnit.MiB)
    }
  }

  /**
   * Calculate recommended executor memory overhead using the new formula:
   * totalMemoryForExecutor - executor heap memory - offHeap.size - safeReserveMemory(5GB)
   *
   * Note: This new formula is only used for onPrem platform when offHeapLimit is enabled.
   * For CSP platforms or when offHeapLimit is disabled, the original calculation is used.
   *
   * @param totalMemMinusReserved Total memory available for executor in MB
   * @param executorHeapMB Executor heap memory in MB
   * @param offHeapMB Off-heap memory size in MB
   * @return Recommended executor memory overhead in MB
   */
  private def calculateExecutorMemoryOverhead(
    totalMemMinusReserved: Long,
    executorHeapMB: Long,
    offHeapMB: Long): Long = {

    // Use new formula only for onPrem platform when offHeapLimit is enabled
    if (!platform.isPlatformCSP && isOffHeapLimitUserEnabled) {
      val calculatedOverhead = totalMemMinusReserved - executorHeapMB - offHeapMB

      // Ensure the overhead is not negative and has a minimum value
      val minOverhead = executorHeapMB * configProvider.getEntry("HEAP_OVERHEAD_FRACTION")
        .getDefault.toDouble
      Math.max(calculatedOverhead.toLong, minOverhead.toLong)
    } else {
      // Use original calculation for CSP platforms or when offHeapLimit is disabled
      val minOverhead = executorHeapMB * configProvider.getEntry("HEAP_OVERHEAD_FRACTION")
        .getDefault.toDouble
      minOverhead.toLong
    }
  }
}

object AutoTuner {
  /**
   * Helper function to get a combined property function that can be used
   * to retrieve the value of a property in the following priority order:
   * 1. From the recommendations map
   *    - This will include the user-enforced Spark properties
   *    - This implies the properties to be present in the target application
   * 2. From the source Spark properties
   */
  def getCombinedPropertyFn(
    recommendations: mutable.LinkedHashMap[String, TuningEntryTrait],
    sourceSparkProperties: Map[String, String]): String => Option[String] = {
    (key: String) => {
      recommendations.get(key).map(_.getTuneValue())
        .orElse(sourceSparkProperties.get(key))
    }
  }
}

/**
 * Implementation of the `AutoTuner` specific for the Profiling Tool.
 * This class implements the logic to recommend AutoTuner configurations
 * specifically for GPU event logs.
 */
class ProfilingAutoTuner(
    appInfoProvider: BaseProfilingAppSummaryInfoProvider,
    platform: Platform,
    driverInfoProvider: DriverLogInfoProvider,
    userProvidedTuningConfigs: Option[TuningConfiguration])
  extends AutoTuner(appInfoProvider, platform, driverInfoProvider,
    userProvidedTuningConfigs, ProfilingAutoTunerHelper) {

  override type ConfigProviderType = ProfTuningConfigProvider

  override protected def createConfigProvider(
      config: Option[TuningConfiguration]): ProfTuningConfigProvider = {
    TuningConfigProvider
      .builder
      .withUserProvidedConfig(config)
      .build[ProfTuningConfigProvider]
  }

  /**
   * Overrides the calculation for 'spark.sql.files.maxPartitionBytes'.
   * Logic:
   * - First, calculate the recommendation based on input sizes (parent implementation).
   * - If GPU OOM errors occurred in scan stages,
   *     - If calculated value is defined, choose the minimum between the calculated value and
   *       half of the current value.
   *     - Else, halve the current value.
   * - Else, use the value from the parent implementation.
   */
  override def calculateMaxPartitionBytesInMB(maxPartitionBytes: String): Option[Long] = {
    // First, calculate the recommendation based on input sizes
    val calculatedValueFromInputSize = super.calculateMaxPartitionBytesInMB(maxPartitionBytes)
    getPropertyValue("spark.sql.files.maxPartitionBytes") match {
      case Some(currentValue) if appInfoProvider.scanStagesWithGpuOom.nonEmpty =>
        // GPU OOM detected. We may want to reduce max partition size.
        val halvedValue = StringUtils.convertToMB(currentValue, Some(ByteUnit.BYTE)) / 2
        // Choose the minimum between the calculated value and half of the current value.
        calculatedValueFromInputSize match {
          case Some(calculatedValue) => Some(math.min(calculatedValue, halvedValue))
          case None => Some(halvedValue)
        }
      case _ =>
        // Else, use the value from the parent implementation
        calculatedValueFromInputSize
    }
  }

  /**
   * Overrides the calculation for 'spark.sql.shuffle.partitions'.
   * This method checks for task OOM errors in shuffle stages and recommends to increase
   * shuffle partitions if task OOM errors occurred.
   */
  override def recommendShufflePartitionsInternal(): Int = {
    val calculatedValue = super.recommendShufflePartitionsInternal()
    if (appInfoProvider.gpuShuffleStagesWithContainerOom.nonEmpty) {
      // Shuffle Stages with Task OOM detected. We may want to increase shuffle partitions.
      val recShufflePartitions = shufflePartitionValue *
        configProvider.getEntry("SHUFFLE_PARTITION_MULTIPLIER").getDefault.toInt
      appendComment(shufflePartitionsCommentForGpuOOM)
      math.max(calculatedValue, recShufflePartitions)
    } else {
      // Else, return the calculated value from the parent implementation
      calculatedValue
    }
  }

  /**
   * Profiling AutoTuner retains existing "spark.plugins" property and
   * RAPIDS plugin is added to it.
   */
  override def recommendPluginPropsInternal(): Unit = {
    recommendClassNameProperty("spark.plugins", autoTunerHelper.rapidsPluginClassName)
  }

}

/**
 * Helper trait for the AutoTuner
 */
trait AutoTunerHelper extends Logging {
  /**
   * Strategy for cluster shape recommendation.
   * See [[com.nvidia.spark.rapids.tool.ClusterSizingStrategy]] for different strategies.
   */
  def recommendedClusterSizingStrategy(platform: Platform): ClusterSizingStrategy
  // the plugin jar is in the form of rapids-4-spark_scala_binary-(version)-*.jar
  lazy val pluginJarRegEx: Regex = "rapids-4-spark_\\d\\.\\d+-(\\d{2}\\.\\d{2}\\.\\d+).*\\.jar".r
  // Starting with this plugin version, the RAPIDS plugin auto-tunes the number of
  // concurrent GPU tasks based on memory usage (see spark-rapids#12374), so the
  // AutoTuner should no longer recommend `spark.rapids.sql.concurrentGpuTasks`.
  lazy val pluginVersionAutoConcurrentGpuTasks: String = "25.06.0"
  lazy val gpuKryoRegistratorClassName = "com.nvidia.spark.rapids.GpuKryoRegistrator"
  lazy val rapidsPluginClassName = "com.nvidia.spark.SQLPlugin"
  lazy val kubernetesGpuVendor = "nvidia.com"

  // Recommended values for specific unsupported configurations
  lazy val unsupportedOperatorRecommendations: Map[String, String] = Map(
    "spark.rapids.sql.incompatibleDateFormats.enabled" -> "true"
  )

  /**
   * Abstract method to create an instance of the AutoTuner.
   */
  def createAutoTunerInstance(
    appInfoProvider: AppSummaryInfoBaseProvider,
    platform: Platform,
    driverInfoProvider: DriverLogInfoProvider,
    userProvidedTuningConfigs: Option[TuningConfiguration]): AutoTuner

  def handleException(
      ex: Throwable,
      appInfo: AppSummaryInfoBaseProvider,
      platform: Platform,
      driverInfoProvider: DriverLogInfoProvider,
      userProvidedTuningConfigs: Option[TuningConfiguration]): AutoTuner = {
    logError("Exception: " + ex.getStackTrace.mkString("Array(", ", ", ")"))
    val tuning = createAutoTunerInstance(appInfo,
      platform, driverInfoProvider, userProvidedTuningConfigs)
    val msg = ex match {
      case cEx: ConstructorException => cEx.getContext
      case _ => if (ex.getCause != null) ex.getCause.toString else ex.toString
    }
    tuning.appendComment(msg)
    tuning
  }

  /**
   * Similar to [[buildAutoTuner]] but it allows constructing the AutoTuner without an
   * existing file. This can be used in testing.
   *
   * @param singleAppProvider the wrapper implementation that accesses the properties of the profile
   *                          results.
   * @param platform represents the environment created as a target for recommendations.
   * @param driverInfoProvider wrapper implementation that accesses the information from driver log.
   * @return a new AutoTuner object.
   */
  def buildAutoTunerFromProps(
      singleAppProvider: AppSummaryInfoBaseProvider,
      platform: Platform = PlatformFactory.createInstance(),
      driverInfoProvider: DriverLogInfoProvider = BaseDriverLogInfoProvider.noneDriverLog,
      userProvidedTuningConfigs: Option[TuningConfiguration] = None
  ): AutoTuner = {
    try {
      createAutoTunerInstance(
        singleAppProvider,
        platform,
        driverInfoProvider,
        userProvidedTuningConfigs)
    } catch {
      case NonFatal(e) =>
        handleException(e, singleAppProvider, platform, driverInfoProvider,
          userProvidedTuningConfigs)
    }
  }

  /**
   * This is used to build the AutoTuner from an existing event log file.
   * This is what gets called by Qualification/Profiling tools
   */
  def buildAutoTuner(
      singleAppProvider: AppSummaryInfoBaseProvider,
      platform: Platform,
      driverInfoProvider: DriverLogInfoProvider = BaseDriverLogInfoProvider.noneDriverLog,
      userProvidedTuningConfigs: Option[TuningConfiguration] = None
  ): AutoTuner = {
    try {
      val autoT = createAutoTunerInstance(
        singleAppProvider, platform, driverInfoProvider, userProvidedTuningConfigs)
      autoT
    } catch {
      case NonFatal(e) =>
        handleException(e, singleAppProvider, platform, driverInfoProvider,
          userProvidedTuningConfigs)
    }
  }

  def buildShuffleManagerClassName(smVersion: String): String = {
    s"com.nvidia.spark.rapids.spark$smVersion.RapidsShuffleManager"
  }
}

/**
 * Provides configuration settings for the Profiling Tool's AutoTuner. This object is as a concrete
 * implementation of the `AutoTunerHelper` interface.
 */
object ProfilingAutoTunerHelper extends AutoTunerHelper {
  /**
   * On-prem profiling without a target cluster: use SourceCoresPreservingStrategy
   * to preserve the source cluster's cores and executor count (the hardware is fixed).
   * For CSP platforms or when a target cluster is provided, use ConstantGpuCountStrategy.
   */
  def recommendedClusterSizingStrategy(platform: Platform): ClusterSizingStrategy = {
    // true when user provided target hardware (workerInfo with cores/memory/GPU)
    val hasTargetWorkerInfo = platform.targetCluster.exists(!_.getWorkerInfo.isEmpty)
    if (!platform.isPlatformCSP && !hasTargetWorkerInfo) {
      SourceCoresPreservingStrategy
    } else {
      ConstantGpuCountStrategy
    }
  }

  def createAutoTunerInstance(
      appInfoProvider: AppSummaryInfoBaseProvider,
      platform: Platform,
      driverInfoProvider: DriverLogInfoProvider,
      userProvidedTuningConfigs: Option[TuningConfiguration]): AutoTuner = {
    appInfoProvider match {
      case profilingAppProvider: BaseProfilingAppSummaryInfoProvider =>
        new ProfilingAutoTuner(profilingAppProvider, platform,
          driverInfoProvider, userProvidedTuningConfigs)
      case _ =>
        throw new IllegalArgumentException("'appInfoProvider' must be an instance of " +
          s"${classOf[BaseProfilingAppSummaryInfoProvider]}")
    }
  }
}

/**
 * Trait providing static comments for the AutoTuner.
 * The static comments are used in unit tests as well.
 */
trait AutoTunerStaticComments {
  // scalastyle:off line.size.limit
  private lazy val advancedConfigDocUrl = "https://nvidia.github.io/spark-rapids/docs/additional-functionality/advanced_configs.html#advanced-configuration"
  private lazy val shuffleManagerDocUrl = "https://docs.nvidia.com/spark-rapids/user-guide/latest/additional-functionality/rapids-shuffle.html#rapids-shuffle-manager"

  val classPathComments: Map[String, String] = Map(
    "rapids.jars.missing" ->
      ("RAPIDS Accelerator for Apache Spark plugin jar is missing\n" +
        "  from the classpath entries.\n" +
        "  If the Spark RAPIDS jar is being bundled with your\n" +
        "  Spark distribution, this step is not needed."),
    "rapids.jars.multiple" ->
      ("Multiple RAPIDS Accelerator for Apache Spark plugin jar\n" +
        "  exist on the classpath.\n" +
        "  Make sure to keep only a single jar."),
    "rapids.shuffle.jars" ->
      ("The RAPIDS Shuffle Manager requires spark.driver.extraClassPath\n" +
        "  and spark.executor.extraClassPath settings to include the\n" +
        "  path to the Spark RAPIDS plugin jar.\n" +
        "  If the Spark RAPIDS jar is being bundled with your Spark\n" +
        "  distribution, this step is not needed.")
  )

  def shuffleManagerCommentForUnsupportedVersion(sparkVersion: String, platform: Platform): String = {
    val (latestSparkVersion, latestSmVersion) = platform.latestSupportedShuffleManagerInfo
    s"""
       |Cannot recommend RAPIDS Shuffle Manager for unsupported ${platform.sparkVersionLabel}: '$sparkVersion'.
       |To enable RAPIDS Shuffle Manager, use a supported ${platform.sparkVersionLabel} (e.g., '$latestSparkVersion')
       |and set: '--conf spark.shuffle.manager=com.nvidia.spark.rapids.spark$latestSmVersion.RapidsShuffleManager'.
       |See supported versions: $shuffleManagerDocUrl.
       |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }
  // scalastyle:on line.size.limit

  def shuffleManagerCommentForMissingVersion: String = {
    "'spark.shuffle.manager' is not recommended as Spark version cannot be determined."
  }

  def shuffleManagerCommentForQualification: String = {
    "'spark.shuffle.manager' is not recommended because the Spark version on the " +
      "GPU cluster is unknown during Qualification."
  }

  def shufflePartitionsCommentForSpilling: String = {
    "Shuffle partitions should be increased since spilling occurred in shuffle stages."
  }

  def shufflePartitionsCommentForGpuOOM: String = {
    "Shuffle partitions should be increased since task OOM occurred in shuffle stages."
  }

  /**
   * Comment for missing GPU discovery script.
   * Since this comment is conditional, it is not included in the
   * tuningTable yaml.
   */
  def missingGpuDiscoveryScriptComment: String = {
    s"""
       |To enable Spark to discover and schedule GPU resources, set the
       |'spark.executor.resource.gpu.discoveryScript' property according to cluster
       |manager's documentation. Sample discovery script is available at
       |'$${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh'.
       |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }

  def additionalSparkPluginsComment: String = {
    """
      |To include additional plugins for the GPU cluster, specify 'spark.plugins' in the
      |'sparkProperties.enforced' section in '--target_cluster_info'.
      |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }

  def latestPluginJarComment(latestJarMvnUrl: String, currentJarVer: String): String = {
    s"""
       |A newer RAPIDS Accelerator for Apache Spark plugin is available:
       |$latestJarMvnUrl
       |Version used in application is $currentJarVer.
       |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }

  def notEnoughMemComment(minSizeInMB: Long): String = {
    s"""
       |This node/worker configuration is not ideal for using the RAPIDS Accelerator
       |for Apache Spark because it doesn't have enough memory for the executors.
       |We recommend either using nodes with more memory or reducing 'spark.memory.offHeap.size',
       |as off-heap memory is unused by the RAPIDS Accelerator, unless explicitly required by
       |the application. Need at least $minSizeInMB MB memory per executor.
       |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }

  def notEnoughMemCommentForKey(key: String): String = {
    s"Not enough memory to set '$key'. See comments for more details."
  }

  /**
   * Append a comment to the list indicating that the property was enforced by the user.
   * @param key the property set by the autotuner.
   */
  def getEnforcedPropertyComment(key: String): String = {
    s"'$key' was user-enforced in the target cluster properties."
  }

  /**
   * Append a comment to the list indicating that the property was preserved from source.
   * @param key the property preserved from source.
   */
  def getPreservedPropertyComment(key: String): String = {
    s"'$key' was preserved from source application properties as specified in target cluster."
  }

  /**
   * Append a comment to the list indicating that the property was specified in preserve list
   * but not found in source properties.
   * @param key the property specified in preserve list but not found in source.
   */
  def getPreservedPropertyNotFoundComment(key: String): String = {
    s"""
    |'$key' was specified in preserve list but not found in source properties.
    |AutoTuner will continue with its recommendation for this property.
    |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }

  /**
   * Append a comment to the list indicating that the property was excluded from source.
   * @param key the property excluded from source.
   */
  def getExcludedPropertyComment(key: String): String = {
    s"'$key' was excluded from tuning recommendations as specified in target cluster."
  }

  def commentForExperimentalConfig(config: String): String = {
    s"Using $config does not guarantee to produce the same results as CPU. " +
      s"Please refer to $advancedConfigDocUrl."
  }

  def commentForDynamicAllocationAdjustment(properties: List[String],
        sourceExecCores: Int, targetExecCores: Int): String = {
    s"""
       |Tuned dynamic allocation properties (${properties.mkString(", ")})
       |based on cores ratio (source: $sourceExecCores cores, target: $targetExecCores cores).
       |""".stripMargin.trim.replaceAll("\n", "\n  ")
  }
}

/**
 * Trait providing default comments for missing or recommended Spark properties,
 * using values from the provided tuning configuration.
 * Class mixing in this trait must provide a `tuningConfigs` instance.
 */
trait AutoTunerCommentsWithTuningConfigs {
  /** Type of config provider - defined by subclasses */
  type ConfigProviderType <: TuningConfigProvider
  val configProvider: ConfigProviderType

  /**
   * Helper function to generate a comment for a missing property.
   */
  private def generateMissingComment(property: String, recommendation: String): String = {
    s"'$property' should be set to $recommendation."
  }

  // scalastyle:off line.size.limit
  protected val commentsForMissingMemoryProps: Map[String, String] = Map(
    "spark.executor.memory" ->
      generateMissingComment("spark.executor.memory",
        s"${configProvider.getEntry("HEAP_PER_CORE").getDefault}/core"),
    "spark.rapids.memory.pinnedPool.size" ->
      generateMissingComment("spark.rapids.memory.pinnedPool.size",
        configProvider.getEntry("PINNED_MEMORY").getDefault))

  protected val commentsForMissingProps: Map[String, String] = Map(
    "spark.executor.cores" ->
      // TODO: This could be extended later to be platform specific.
      generateMissingComment("spark.executor.cores",
        configProvider.getEntry("CORES_PER_EXECUTOR").getDefault),
    "spark.executor.instances" ->
      generateMissingComment("spark.executor.instances",
        "(cpuCoresPerNode * numWorkers) / 'spark.executor.cores'"),
    "spark.task.resource.gpu.amount" ->
      generateMissingComment("spark.task.resource.gpu.amount",
        configProvider.getEntry("TASK_GPU_RESOURCE_AMT").getDefault),
    "spark.rapids.sql.concurrentGpuTasks" ->
      generateMissingComment("spark.rapids.sql.concurrentGpuTasks",
        s"Min(${configProvider.getEntry("CONC_GPU_TASKS").getMax.toLong}, " +
          s"(gpuMemory / ${configProvider.getEntry("GPU_MEM_PER_TASK").getDefault}))"),
    "spark.rapids.sql.enabled" ->
      "'spark.rapids.sql.enabled' should be true to enable SQL operations on the GPU.",
    "spark.sql.adaptive.enabled" ->
      "'spark.sql.adaptive.enabled' should be enabled for better performance."
  ) ++ commentsForMissingMemoryProps
  // scalastyle:off line.size.limit
}
