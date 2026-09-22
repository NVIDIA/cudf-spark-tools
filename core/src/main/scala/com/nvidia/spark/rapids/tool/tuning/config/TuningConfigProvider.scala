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

package com.nvidia.spark.rapids.tool.tuning.config

import java.util

import scala.jdk.CollectionConverters._

/**
 * Base class for providing tuning configuration entries with tool-specific overrides.
 *
 * This class encapsulates the logic for loading and merging tuning configurations,
 * combining default configurations with tool-specific overrides (qualification or profiling).
 * It provides a unified interface for accessing configuration entries regardless of the tool type.
 *
 * The configuration hierarchy is:
 * 1. Default configurations (applicable to all tools)
 * 2. Tool-specific overrides (qualification or profiling)
 * 3. Merged result (default + overrides, with overrides taking precedence)
 *
 * @param rawConfig The raw tuning configuration loaded from YAML files, containing
 *                  default, qualification, and profiling configuration lists
 */
abstract class TuningConfigProvider(
    rawConfig: TuningConfiguration,
    private val userProvidedConfig: Option[TuningConfiguration]) {

  /**
   * Tool-specific configuration overrides. Subclasses must define which override list
   * to use (qualification or profiling).
   */
  protected val toolOverrides: util.List[TuningConfigEntry]

  /** Returns the tool-specific entries from a user-provided configuration. */
  protected def getUserProvidedToolOverrides(
      config: TuningConfiguration): util.List[TuningConfigEntry]

  /**
   * Cached lookup map for fast configuration entry retrieval by name.
   * Built by merging default configurations with tool-specific overrides.
   * Marked as transient to avoid serialization issues.
   */
  @transient
  private lazy val tuningConfigsMap: Map[String, TuningConfigEntry] = {
    buildTuningConfigsMap()
  }

  /**
   * Retrieves a configuration entry by its name.
   *
   * @param key The configuration entry name (e.g., "EXECUTOR_CORES", "HEAP_PER_CORE")
   * @return The TuningConfigEntry corresponding to the given key
   */
  @throws[java.util.NoSuchElementException]
  def getEntry(key: String): TuningConfigEntry = {
    tuningConfigsMap(key)
  }

  private def getEntryField(
      entries: util.List[TuningConfigEntry],
      key: String,
      fieldValue: TuningConfigEntry => String): Option[String] = {
    Option(entries).toSeq.flatMap(_.asScala)
      .find(_.name == key)
      .flatMap(entry => Option(fieldValue(entry)).filter(_.nonEmpty))
  }

  /**
   * Returns the final field value only when its active value came from the user configuration.
   * A user tool-specific field has highest precedence. A shipped tool-specific field masks the
   * same field from a user default entry.
   */
  private def getActiveUserProvidedField(
      key: String,
      fieldValue: TuningConfigEntry => String): Option[String] = {
    userProvidedConfig.flatMap { config =>
      val userToolValue = getEntryField(
        getUserProvidedToolOverrides(config), key, fieldValue)
      val activeToolValue = getEntryField(toolOverrides, key, fieldValue)
      val userDefaultValue = getEntryField(config.default, key, fieldValue)
      val userOwnsActiveValue = userToolValue.isDefined ||
        (activeToolValue.isEmpty && userDefaultValue.isDefined)
      if (userOwnsActiveValue) {
        Option(fieldValue(getEntry(key))).filter(_.nonEmpty)
      } else {
        None
      }
    }
  }

  /** Returns the active default only when the user explicitly supplied it. */
  def getUserProvidedDefault(key: String): Option[String] = {
    getActiveUserProvidedField(key, _.default)
  }

  /** Returns whether the user explicitly supplied the active default value for an entry. */
  def isDefaultValueUserProvided(key: String): Boolean = {
    getUserProvidedDefault(key).isDefined
  }

  /**
   * Returns the final maximum only when the active user configuration explicitly supplied it.
   * A shipped maximum is intentionally not returned, even when it has the same value.
   *
   * @param key The configuration entry name
   * @return The merged maximum when it came from the user configuration, otherwise None
   */
  def getUserProvidedMax(key: String): Option[String] = {
    getActiveUserProvidedField(key, _.max)
  }

  /**
   * Returns the plugin configurations including tuning plugins and rules.
   *
   * @return TuningPluginsConfig containing all plugin and rule definitions
   */
  def getPluginsConfigs: TuningPluginsConfig = {
    rawConfig.pluginConfigs
  }

  /**
   * Builds the configuration lookup map by merging default configs with tool overrides.
   * This method creates a map from configuration entry names to their merged values,
   * where tool-specific overrides take precedence over defaults.
   *
   * @return A map of configuration entry names to their merged TuningConfigEntry objects
   */
  private def buildTuningConfigsMap(): Map[String, TuningConfigEntry] = {
    val mergedConfigs = TuningConfiguration.mergeConfigs(rawConfig.default, toolOverrides)
    mergedConfigs.asScala.map(e => e.name -> e).toMap
  }
}


/**
 * Profiling-specific tuning configuration provider.
 *
 * This implementation applies profiling tool overrides to the default configurations.
 * Profiling configurations are optimized for analyzing GPU event logs and providing
 * recommendations based on actual workload execution metrics.
 *
 * @param rawConfig The raw tuning configuration containing default, qualification,
 *                  and profiling configuration lists
 */
class ProfTuningConfigProvider(
    rawConfig: TuningConfiguration,
    userProvidedConfig: Option[TuningConfiguration] = None
) extends TuningConfigProvider(rawConfig, userProvidedConfig) {
  /** Returns the profiling-specific configuration overrides */
  override protected lazy val toolOverrides: util.List[TuningConfigEntry] = rawConfig.profiling

  override protected def getUserProvidedToolOverrides(
      config: TuningConfiguration): util.List[TuningConfigEntry] = config.profiling
}

/**
 * Qualification-specific tuning configuration provider.
 *
 * This implementation applies qualification tool overrides to the default configurations.
 * Qualification configurations are optimized for bootstrap scenarios where recommendations
 * are generated based on CPU event logs to help estimate GPU acceleration benefits.
 *
 * @param rawConfig The raw tuning configuration containing default, qualification,
 *                  and profiling configuration lists
 */
class QualTuningConfigProvider(
    rawConfig: TuningConfiguration,
    userProvidedConfig: Option[TuningConfiguration] = None
) extends TuningConfigProvider(rawConfig, userProvidedConfig) {
  /** Returns the qualification-specific configuration overrides */
  override protected lazy val toolOverrides: util.List[TuningConfigEntry] = rawConfig.qualification

  override protected def getUserProvidedToolOverrides(
      config: TuningConfiguration): util.List[TuningConfigEntry] = config.qualification
}

/**
 * Companion object providing factory methods for creating TuningConfigProvider instances.
 *
 * This object implements the Builder pattern to facilitate the creation of tool-specific
 * configuration providers (Qualification or Profiling) with optional user-provided
 * configuration overrides.
 *
 * Example usage:
 * {{{
 *   // Create a Qualification config provider with default configs only
 *   val qualProvider = TuningConfigProvider.builder
 *     .build[QualTuningConfigProvider]
 *
 *   // Create a Profiling config provider with user overrides
 *   val profProvider = TuningConfigProvider.builder
 *     .withUserProvidedConfig(Some(userConfig))
 *     .build[ProfTuningConfigProvider]
 * }}}
 */
object TuningConfigProvider {

  /**
   * Builder for creating TuningConfigProvider instances.
   *
   * The builder supports a fluent API for constructing configuration providers with
   * optional user-provided configurations. It automatically merges user configurations
   * with the default configurations loaded from the bootstrap YAML file.
   *
   * The build process:
   * 1. Load default configurations from tuningConfigs.yaml
   * 2. Optionally merge with user-provided configurations
   * 3. Create the appropriate provider subclass (Qualification or Profiling)
   */
  class Builder {
    /** Default tuning configurations loaded from the bootstrap YAML file */
    private val defaultConfig: TuningConfiguration = TuningConfiguration.getDefaultConfigs

    /** Optional user-provided configuration overrides */
    private var userProvidedConfig: Option[TuningConfiguration] = None

    /**
     * Sets user-provided configuration overrides.
     *
     * When user configurations are provided, they will be merged with the default
     * configurations, with user values taking precedence over defaults.
     *
     * @param config Optional user-provided tuning configuration
     * @return This builder instance for method chaining
     */
    def withUserProvidedConfig(
        config: Option[TuningConfiguration]): Builder = {
      this.userProvidedConfig = config
      this
    }

    /**
     * Computes the final merged configuration.
     *
     * If user-provided configurations exist, they are merged with the defaults.
     * Otherwise, only the default configurations are used.
     *
     * @return The final merged TuningConfiguration
     */
    private def finalConfigs: TuningConfiguration = {
      userProvidedConfig match {
        case Some(userConfig) =>
          // Merge user configs with defaults (user values take precedence)
          defaultConfig.merge(userConfig)
        case None =>
          // Use only default configs
          defaultConfig
      }
    }

    /**
     * Builds a tuning config provider of the specified type.
     *
     * This method uses Scala's ClassTag to determine at runtime which concrete
     * provider class to instantiate (QualTuningConfigProvider or ProfTuningConfigProvider).
     *
     * @tparam T The concrete TuningConfigProvider subclass to create
     * @param tag Implicit ClassTag for runtime type information
     * @return A new instance of the requested provider type
     *
     * Usage:
     * {{{
     *   val qualProvider = builder.build[QualTuningConfigProvider]
     *   val profProvider = builder.build[ProfTuningConfigProvider]
     * }}}
     */
    @throws[IllegalArgumentException]
    def build[T <: TuningConfigProvider](implicit tag: scala.reflect.ClassTag[T]): T = {
      tag.runtimeClass match {
        case c if c == classOf[QualTuningConfigProvider] =>
          new QualTuningConfigProvider(finalConfigs, userProvidedConfig).asInstanceOf[T]
        case c if c == classOf[ProfTuningConfigProvider] =>
          new ProfTuningConfigProvider(finalConfigs, userProvidedConfig).asInstanceOf[T]
        case _ =>
          throw new IllegalArgumentException(
            s"Unsupported TuningConfigProvider type: ${tag.runtimeClass.getName}. " +
              s"Supported types: QualTuningConfigProvider, ProfTuningConfigProvider")
      }
    }
  }

  /**
   * Creates a new Builder instance for constructing TuningConfigProvider objects.
   *
   * This is the entry point for creating configuration providers using the builder pattern.
   *
   * @return A new Builder instance
   */
  def builder: Builder = {
    new Builder()
  }
}
