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

import com.nvidia.spark.rapids.tool.plugins.ConditionTrait
import com.nvidia.spark.rapids.tool.tuning.AutoTuner

/**
 * Trait defining the contract for tuning rules in the cuDF plugin auto-tuning system.
 *
 * A tuning rule encapsulates a specific optimization strategy that can be applied to
 * Spark applications to improve performance when running on GPU. Each rule consists of:
 * - A condition that determines when the rule should be triggered
 * - Logic to apply the optimization when triggered
 * - Priority to determine execution order relative to other rules
 *
 * Rules are executed in priority order (0 = highest, 100 = lowest), and only when their
 * associated conditions evaluate to true. The `trigger` method orchestrates the conditional
 * execution, while `apply` contains the actual optimization logic.
 *
 * Implementations of this trait are typically loaded dynamically from configuration files
 * and instantiated via reflection.
 */
trait TuningRuleTrait {
  /**
   * The unique name identifying this tuning rule.
   *
   * @return A string identifier for the rule, used for logging and configuration
   */
  def name: String

  /**
   * The condition that must be satisfied for this rule to be applied.
   *
   * @return A ConditionTrait[AutoTuner] instance that evaluates whether this rule should execute
   */
  def condition: ConditionTrait[AutoTuner]

  /**
   * Applies the tuning optimization logic defined by this rule.
   *
   * This method contains the core implementation of the rule's optimization strategy.
   * It is invoked by the `trigger` method when the rule's condition is satisfied.
   *
   * @param tuner The AutoTuner instance providing access to application data and tuning context
   */
  def apply(tuner: AutoTuner): Unit

  /**
   * Triggers the execution of this rule by evaluating its condition and applying it if satisfied.
   *
   * This method acts as the orchestration point for the rule, checking the condition
   * and delegating to `apply` when appropriate. Implementations typically follow the pattern:
   * {{{
   *   if (condition.evaluate(tuner)) {
   *     apply(tuner)
   *   }
   * }}}
   *
   * @param tuner The AutoTuner instance used for condition evaluation and optimization application
   */
  def trigger(tuner: AutoTuner): Unit

  /**
   * The execution priority of this rule.
   *
   * Rules with lower priority values are executed first. Valid range is 0-100,
   * where 0 represents the highest priority and 100 the lowest. The default priority
   * is typically 50.
   *
   * @return An integer representing the rule's priority in the execution order
   */
  def priority: Int
}
