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

package com.nvidia.spark.rapids.tool.tuning.plugins

import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids.tool.plugins.ConditionTrait
import com.nvidia.spark.rapids.tool.tuning.AutoTuner

/**
 * Trait defining the contract for tuning plugins in the cuDF plugin auto-tuning system.
 *
 * A tuning plugin is a collection of related tuning rules that share a common activation
 * condition. Plugins provide a way to group and organize rules based on functionality,
 * hardware requirements, or platform-specific optimizations.
 *
 * Each plugin consists of:
 * - A collection of tuning rules that implement specific optimizations
 * - An activation condition that determines when the plugin should be enabled
 * - An optional fire method for plugin-level initialization or setup
 *
 * Plugins are typically loaded from configuration files and activated dynamically based
 * on the application context and hardware environment. Only plugins whose activation
 * conditions are satisfied will have their rules executed.
 *
 * The typical lifecycle of a plugin:
 * 1. Plugin is loaded and instantiated with its rules
 * 2. Activation condition is evaluated against the AutoTuner context
 * 3. If active, the `fire()` method is called for initialization
 * 4. Individual rules are executed in priority order
 *
 * Example implementation:
 * {{{
 *   class MyTuningPlugin extends TuningPluginTrait {
 *     override val activationCond: ConditionTrait[AutoTuner] = {
 *       new ConditionTrait[AutoTuner] {
 *         override def eval(tuner: AutoTuner): Boolean = {
 *           tuner.hasGPU && tuner.isDataprocPlatform
 *         }
 *       }
 *     }
 *
 *     override def fire(): Unit = {
 *       // Optional plugin-level setup
 *       logInfo("Activating MyTuningPlugin")
 *     }
 *   }
 * }}}
 */
trait TuningPluginTrait {
  /**
   * The collection of tuning rules belonging to this plugin.
   *
   * Rules are stored in an ArrayBuffer to allow dynamic addition during plugin
   * initialization. Rules are typically executed in priority order (0 = highest).
   * Each rule in this collection will be evaluated and potentially applied when
   * the plugin is active.
   */
  val rules: ArrayBuffer[TuningRuleTrait] = ArrayBuffer()

  /**
   * The activation condition for this plugin.
   *
   * This condition determines whether the plugin should be enabled for a given
   * AutoTuner context. The condition is evaluated before any rules are executed,
   * and only if it returns true will the plugin's rules be applied.
   *
   * Common activation conditions include:
   * - Platform-specific checks (e.g., Databricks, Dataproc, EMR)
   * - Hardware requirements (e.g., specific GPU types)
   * - Application characteristics (e.g., shuffle-heavy workloads)
   * - Configuration flags (e.g., experimental features enabled)
   *
   * @return A ConditionTrait that evaluates to true when this plugin should be active
   */
  val activationCond: ConditionTrait[AutoTuner]

  /**
   * Executes all rules in this plugin, ordered by priority (highest first).
   * Each rule is triggered sequentially on the tuner instance.
   *
   * This method is invoked after the activation condition is satisfied and before
   * any rules are executed.
   *
   */
  def fire(): Unit = { }
}
