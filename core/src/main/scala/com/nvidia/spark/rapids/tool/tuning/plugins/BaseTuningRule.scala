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

import scala.util.{Success, Try}
import scala.util.control.NonFatal

import com.nvidia.spark.rapids.tool.plugins.ConditionTrait
import com.nvidia.spark.rapids.tool.tuning.AutoTuner
import com.nvidia.spark.rapids.tool.tuning.config.TuningRuleConfigEntry

import org.apache.spark.internal.Logging

/**
 * Abstract base class for implementing tuning rules in the cuDF plugin auto-tuning system.
 *
 * This class provides a standard implementation of the TuningRuleTrait interface with
 * common functionality that most tuning rules need:
 * - Configuration management via TuningRuleConfig
 * - Application counting to track rule execution
 * - Default condition (always true) that can be overridden
 * - Post-apply hooks for cleanup or tracking
 *
 * Concrete implementations must:
 * 1. Extend this class
 * 2. Implement the `apply(tuner: AutoTuner)` method with optimization logic
 * 3. Optionally override `condition` to define when the rule should execute
 *
 * The rule execution lifecycle follows this pattern:
 * 1. `trigger` is called with an AutoTuner instance
 * 2. The `condition` is evaluated
 * 3. If true, `apply` is called to execute the optimization
 * 4. `postApply` is called for bookkeeping (increments application count)
 *
 * Example:
 * {{{
 *   class MyTuningRule extends BaseTuningRule {
 *     override def apply(tuner: AutoTuner): Unit = {
 *       // Implement optimization logic here
 *     }
 *
 *     override def condition: ConditionTrait[AutoTuner] = {
 *       // Custom condition (optional)
 *       TuningCondPredicates.myCustomCondition
 *     }
 *   }
 * }}}
 */
abstract class BaseTuningRule extends TuningRuleTrait {
  /**
   * The configuration for this tuning rule, containing metadata such as name,
   * description, priority, and enabled status. This field is set by the Builder
   * during rule instantiation.
   */
  var tuningRuleConfig: TuningRuleConfigEntry = _

  /**
   * Counter tracking the number of times this rule has been successfully applied.
   * Incremented by `postApply` after each successful execution.
   */
  var applicationCount: Int = 0

  /**
   * Returns the name of this tuning rule from the configuration.
   *
   * @return The rule's unique identifier
   */
  def name: String = tuningRuleConfig.name

  /**
   * Returns the execution priority of this rule from the configuration.
   *
   * @return The priority value (0 = highest, 100 = lowest)
   */
  def priority: Int = tuningRuleConfig.normalizedPriority

  /**
   * The condition that determines when this rule should be applied.
   * Default implementation always returns true. Override to provide custom conditions.
   *
   * @return A ConditionTrait[AutoTuner] that evaluates to true when the rule should execute
   */
  override def condition: ConditionTrait[AutoTuner] = TuningCondPredicates.alwaysTrue

  /**
   * Hook called after successful application of the rule.
   * Default implementation increments the application counter.
   * Override to add custom post-processing logic.
   *
   * @param tuner The AutoTuner instance used during rule application
   */
  def postApply(tuner: AutoTuner): Unit = {
    applicationCount += 1
  }

  /**
   * Triggers the execution of this rule by evaluating its condition and applying it if satisfied.
   *
   * This method orchestrates the rule execution lifecycle:
   * 1. Evaluates the rule's condition
   * 2. If true, calls `apply` to execute the optimization
   * 3. Calls `postApply` for bookkeeping
   *
   * @param tuner The AutoTuner instance providing access to application data and tuning context
   */
  override def trigger(tuner: AutoTuner): Unit = {
    if (condition.eval(tuner)) {
      apply(tuner)
      postApply(tuner)
    }
  }
}

/**
 * Companion object for BaseTuningRule providing factory methods and builder pattern support.
 *
 * This object facilitates the creation of tuning rule instances through a builder pattern
 * that handles configuration, reflection-based instantiation, and error handling.
 *
 * Example usage:
 * {{{
 *   val ruleOption = BaseTuningRule.builder()
 *     .withTuningRuleConfig(config)
 *     .build()
 *
 *   ruleOption match {
 *     case Some(rule) => // Successfully instantiated
 *     case None => // Instantiation failed (logged as warning)
 *   }
 * }}}
 */
object BaseTuningRule extends Logging {
  /**
   * Creates a new Builder instance for constructing tuning rules.
   *
   * @return A new Builder instance
   */
  def builder(): Builder = new Builder()

  /**
   * Builder class for constructing BaseTuningRule instances.
   *
   * This builder uses the TuningRuleConfig to dynamically instantiate rule classes
   * via reflection. It provides a fluent API for configuration and handles instantiation
   * errors gracefully by returning Option types.
   *
   * The builder pattern ensures:
   * - Type-safe construction
   * - Graceful error handling
   * - Clean separation of configuration and instantiation
   */
  class Builder {
    private var tuningRuleConfig: TuningRuleConfigEntry = _

    /**
     * Sets the tuning rule configuration for the rule being built.
     *
     * @param config The TuningRuleConfig containing rule metadata and class information
     * @return This builder instance for method chaining
     */
    def withTuningRuleConfig(config: TuningRuleConfigEntry): Builder = {
      this.tuningRuleConfig = config
      this
    }

    /**
     * Attempts to instantiate a tuning rule class using reflection.
     *
     * This method:
     * 1. Extracts the class name from the configuration
     * 2. Loads the class using Class.forName
     * 3. Instantiates it using the no-arg constructor
     * 4. Casts it to BaseTuningRule
     *
     * @return A Try containing the instantiated rule or a failure with error details
     * @throws IllegalArgumentException if instantiation fails for any reason
     */
    @throws[IllegalArgumentException]
    private def instantiateRule(): Try[BaseTuningRule] = Try {
      val className = tuningRuleConfig.getClassName
      val clazz = Class.forName(className)
      val rule = try {
        clazz.getDeclaredConstructor().newInstance().asInstanceOf[BaseTuningRule]
      } catch {
        case NonFatal(e) =>
          throw new IllegalArgumentException(
            s"Failed to instantiate Rule $tuningRuleConfig: ${e.getMessage}", e)
      }
      rule
    }

    /**
     * Builds and returns a configured tuning rule instance.
     *
     * This method:
     * 1. Attempts to instantiate the rule class via reflection
     * 2. On success, injects the configuration into the rule and returns Some(rule)
     * 3. On failure, logs a warning and returns None
     *
     * The Option return type allows callers to handle missing or invalid rules gracefully
     * without breaking the entire tuning pipeline.
     *
     * @return Some(TuningRuleTrait) if instantiation succeeds, None if it fails
     */
    def build(): Option[TuningRuleTrait] = {
      instantiateRule() match {
        case Success(rule) =>
          rule.tuningRuleConfig = this.tuningRuleConfig
          Some(rule)
        case scala.util.Failure(e) =>
          logWarning(s"Failed to instantiate rule ${tuningRuleConfig.getName}: ${e.getMessage}")
          None
      }
    }
  }
}
