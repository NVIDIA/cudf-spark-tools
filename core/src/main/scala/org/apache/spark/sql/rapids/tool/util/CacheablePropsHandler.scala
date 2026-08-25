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

package org.apache.spark.sql.rapids.tool.util

import scala.jdk.CollectionConverters._

import com.nvidia.spark.rapids.tool.plugins.AppPropPlugContainerTrait
import com.nvidia.spark.rapids.tool.profiling.ProfileUtils

import org.apache.spark.scheduler.{SparkListenerEnvironmentUpdate, SparkListenerJobStart, SparkListenerLogStart}
import org.apache.spark.sql.rapids.tool.AppEventlogProcessException
import org.apache.spark.util.Utils.REDACTION_REPLACEMENT_TEXT


/**
 * SparkRuntime enumeration is used to identify the specific runtime environment
 * in which the application is being executed.
 */
object SparkRuntime extends Enumeration {
  type SparkRuntime = Value

  /**
   * Represents the default Apache Spark runtime environment.
   */
  val SPARK: SparkRuntime = Value

  /**
   * Represents the cuDF plugin runtime environment.
   */
  val SPARK_RAPIDS: SparkRuntime = Value

  /**
   * Represents the Photon runtime environment on Databricks.
   */
  val PHOTON: SparkRuntime = Value

  /**
   * Represents the Auron runtime environment.
   * Note that Auron is a spark incubator that supports a mixed execution.
   * For example, it has GPU and Iceberg supports.
   */
  val AURON: SparkRuntime = Value

  /**
   * Returns the SparkRuntime value based on the given parameters.
   * @param isGpu    Boolean flag indicating whether the application is running on GPU.
   * @return
   */
  def getRuntime(isGpu: Boolean): SparkRuntime.SparkRuntime = {
    if (isGpu) {
      SPARK_RAPIDS
    } else {
      SPARK
    }
  }
}

// Handles updating and caching Spark Properties for a Spark application.
// Properties stored in this container can be accessed to make decision about certain analysis
// that depends on the context of the Spark properties.
trait CacheablePropsHandler extends AppPropPlugContainerTrait {
  /**
   * Important system properties that should be retained.
   */
  protected def getRetainedSystemProps: Set[String] = Set(
    "file.encoding", "java.version", "os.arch", "os.name",
    "os.version", "sun.java.command", "user.timezone")

  // Patterns to be used to redact sensitive values from Spark Properties.
  private val REDACTED_PROPERTIES = Set[String](
    // S3
    "spark.hadoop.fs.s3a.secret.key",
    "spark.hadoop.fs.s3a.access.key",
    "spark.hadoop.fs.s3a.session.token",
    "spark.hadoop.fs.s3a.encryption.key",
    "spark.hadoop.fs.s3a.bucket.nightly.access.key",
    "spark.hadoop.fs.s3a.bucket.nightly.secret.key",
    "spark.hadoop.fs.s3a.bucket.nightly.session.token",
    // ABFS
    "spark.hadoop.fs.azure.account.oauth2.client.secret",
    "spark.hadoop.fs.azure.account.oauth2.client.id",
    "spark.hadoop.fs.azure.account.oauth2.refresh.token",
    "spark.hadoop.fs.azure.account.key\\..*",
    "spark.hadoop.fs.azure.account.auth.type\\..*",
    // GCS
    "spark.hadoop.google.cloud.auth.service.account.json.keyfile",
    "spark.hadoop.fs.gs.auth.client.id",
    "spark.hadoop.fs.gs.encryption.key",
    "spark.hadoop.fs.gs.auth.client.secret",
    "spark.hadoop.fs.gs.auth.refresh.token",
    "spark.hadoop.fs.gs.auth.impersonation.service.account.for.user\\..*",
    "spark.hadoop.fs.gs.auth.impersonation.service.account.for.group\\..*",
    "spark.hadoop.fs.gs.auth.impersonation.service.account",
    "spark.hadoop.fs.gs.proxy.username",
    // matches on any key that contains password in it.
    "(?i).*password.*"
  )

  // caches the spark-version from the eventlogs
  private var _sparkVersion: String = ""

  /** Getter for Spark version */
  def sparkVersion: String = {
    // get the spark version from the plugins if any.
    // If None is defined, fall back to the cached one.
    getSparkVersionPlugins match {
      case Some(ver) => ver
      case None => _sparkVersion
    }
  }

  /** Setter for Spark version */
  def sparkVersion_=(value: String): Unit = {
    _sparkVersion = value
  }

  // A flag to indicate whether the eventlog is an eventlog with the cuDF plugin runtime.
  var gpuMode = false
  // Indicates the ML eventlogType (i.e., Scala or pyspark). It is set only when MLOps are detected.
  // By default, it is empty.
  var mlEventLogType = ""
  // A flag to indicate that the eventlog is ML
  var pysparkLogFlag = false

  var sparkProperties = Map[String, String]()
  var classpathEntries = Map[String, String]()
  // set the fileEncoding to UTF-8 by default
  var systemProperties = Map[String, String]()

  private def processPropKeys(srcMap: Map[String, String]): Map[String, String] = {
    // Redact the sensitive values in the given map.
    val redactedKeys = REDACTED_PROPERTIES.collect {
      case rK if srcMap.keySet.exists(_.matches(rK)) => rK -> REDACTION_REPLACEMENT_TEXT
    }
    srcMap ++ redactedKeys
  }

  /**
   * Used to validate that the eventlog is allowed to be processed by the Tool
   * @throws org.apache.spark.sql.rapids.tool.AppEventlogProcessException if the eventlog fails the
   *                                                                      validation step
   */
  @throws(classOf[AppEventlogProcessException])
  def validateAppEventlogProperties(): Unit = { }

  def updatePredicatesFromSparkProperties(): Unit = {
    gpuMode ||= ProfileUtils.isPluginEnabled(sparkProperties)
    // auron, deltaOss, and iceberg enabled flags are updated in reEvaluate.
    reEvaluate(sparkProperties)
  }

  def handleEnvUpdateForCachedProps(event: SparkListenerEnvironmentUpdate): Unit = {
    sparkProperties ++= processPropKeys(event.environmentDetails("Spark Properties").toMap)
    classpathEntries ++= event.environmentDetails("Classpath Entries").toMap

    updatePredicatesFromSparkProperties()

    // Update the properties if system environments are set.
    // No need to capture all the properties in memory. We only capture important ones.
    systemProperties ++= event.environmentDetails("System Properties").toMap.filterKeys(
      getRetainedSystemProps.contains(_))

    // After setting the properties, validate the properties.
    validateAppEventlogProperties()
  }

  /**
   * Merges modifiedConfigs from SparkListenerSQLExecutionStart into sparkProperties.
   * Applies redaction (processPropKeys) and updates derived predicates (gpuMode, etc.).
   */
  def mergeModifiedConfigs(modifiedConfigs: Map[String, String]): Unit = {
    if (modifiedConfigs.nonEmpty) {
      sparkProperties ++= processPropKeys(modifiedConfigs)
      updatePredicatesFromSparkProperties()
    }
  }

  def handleJobStartForCachedProps(event: SparkListenerJobStart): Unit = {
    // TODO: we need to improve this in order to support per-job-level
    reEvaluateOnJobLevel(event.properties.asScala)
  }

  def handleLogStartForCachedProps(event: SparkListenerLogStart): Unit = {
    sparkVersion = event.sparkVersion
  }

  def isGPUModeEnabledForJob(event: SparkListenerJobStart): Boolean = {
    gpuMode || ProfileUtils.isPluginEnabled(event.properties.asScala)
  }

  /**
   * Returns the SparkRuntime environment in which the application is being executed.
   * This is calculated based on other cached properties.
   */
  def getSparkRuntime: SparkRuntime.SparkRuntime = {
    getSparkRuntimePlugins match {
      case Some(rt) => rt
      case _ => SparkRuntime.getRuntime(gpuMode)
    }
  }
}
