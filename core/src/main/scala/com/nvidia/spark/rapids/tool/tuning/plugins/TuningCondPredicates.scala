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

import com.nvidia.spark.rapids.tool.plugins.{AlwaysTrueCondition, ConditionTrait, NotCondition}
import com.nvidia.spark.rapids.tool.tuning.AutoTuner
import org.apache.maven.artifact.versioning.ComparableVersion

/**
 * Checks if a property matches a literal value.
 */
class PropertyLiteralCondition(
  propertyKey: String,
  expectedValue: String,
  ignoreCase: Boolean = true) extends ConditionTrait[AutoTuner] {

  override def eval(tuner: AutoTuner): Boolean = {
    tuner.appInfoProvider.getProperty(propertyKey).exists { value =>
      if (ignoreCase) {
        value.equalsIgnoreCase(expectedValue)
      } else {
        value == expectedValue
      }
    }
  }
}

/**
 * Checks if a property value matches a predicate function.
 */
class PropertyPredicateCondition(
  propertyKey: String,
  predicate: String => Boolean) extends ConditionTrait[AutoTuner] {

  override def eval(tuner: AutoTuner): Boolean = {
    tuner.appInfoProvider.getProperty(propertyKey).exists(predicate)
  }
}


/**
 * Factory object providing common condition predicates for tuning rules.
 *
 * This object serves as a centralized registry of reusable condition predicates that
 * tuning rules can use to determine when they should be applied. It includes predicates
 * for checking GPU enablement, property enforcement, and always-true conditions.
 *
 * Predicates are implemented as `ConditionTrait[AutoTuner]` instances, allowing them to
 * evaluate application state and configuration to make decisions about rule activation.
 *
 * == Common Use Cases ==
 *  - Checking if GPU acceleration is enabled in the application
 *  - Verifying if properties are user-enforced (and should not be modified)
 *  - Creating custom conditions based on property values
 *
 * == Example Usage ==
 * {{{
 * class MyTuningRule extends BaseTuningRule {
 *   override def condition: ConditionTrait[AutoTuner] = {
 *     TuningCondPredicates.gpuEnabled
 *   }
 *
 *   override def apply(tuner: AutoTuner): Unit = {
 *     // Rule logic that should only run when GPU is enabled
 *   }
 * }
 * }}}
 *
 * @see ConditionTrait for the base condition interface
 * @see AutoTuner for the tuner context passed to predicates
 */
object TuningCondPredicates {

  /**
   * A condition that always evaluates to true.
   *
   * This is useful for rules that should always be applied regardless of application
   * state or configuration. It can also serve as a placeholder during development or
   * when a rule's condition logic is handled elsewhere.
   *
   * @return A condition that unconditionally returns true
   */
  val alwaysTrue: ConditionTrait[AutoTuner] = new AlwaysTrueCondition()

  /**
   * Checks if a property is user-enforced and should not be modified.
   *
   * User-enforced properties are those explicitly set by the user that should be
   * preserved by the tuning system. This is used internally by other predicates to
   * respect user preferences.
   *
   * @param tuner The AutoTuner instance containing platform and property information
   * @param propKey The property key to check for enforcement
   * @return True if the property is user-enforced, false otherwise
   */
  def rawPropertyEnforced(tuner: AutoTuner, propKey: String): Boolean = {
    tuner.platform.userEnforcedRecommendations.contains(propKey)
  }

  /**
   * Checks if the application Spark version is at least the given minimum version.
   *
   * Uses ComparableVersion to properly compare platform-specific version suffixes
   * such as "-amzn-1".
   *
   * @param tuner The AutoTuner instance containing Spark version information
   * @param minSparkVersion The minimum Spark version required
   * @return True if app Spark version exists and is >= minSparkVersion, false otherwise
   */
  def sparkVersionAtLeast(tuner: AutoTuner, minSparkVersion: String): Boolean = {
    val minVersion = new ComparableVersion(minSparkVersion)
    tuner.appInfoProvider.getSparkVersion
      .map(new ComparableVersion(_))
      .exists(_.compareTo(minVersion) >= 0)
  }

  /**
   * Creates a condition that checks if a property is NOT user-enforced.
   *
   * This predicate is useful for rules that should only modify properties when they
   * haven't been explicitly set by the user. It respects user preferences by preventing
   * tuning rules from overriding intentional configuration choices.
   *
   * == Example Usage ==
   * {{{
   * class MyRule extends BaseTuningRule {
   *   override def condition: ConditionTrait[AutoTuner] = {
   *     TuningCondPredicates.nonEnforcedProperty("spark.executor.memory")
   *   }
   *
   *   override def apply(tuner: AutoTuner): Unit = {
   *     // Only runs if spark.executor.memory was not explicitly set by user
   *   }
   * }
   * }}}
   *
   * @param propKey The property key to check
   * @return A condition that returns true if the property is not user-enforced
   */
  def nonEnforcedProperty(propKey: String): ConditionTrait[AutoTuner] = {
    (tuner: AutoTuner) => {
      !rawPropertyEnforced(tuner, propKey)
    }
  }

  /**
   * Checks if GPU acceleration is enabled in the application.
   *
   * This predicate verifies two conditions:
   * 1. The cuDF plugin is registered in `spark.plugins`
   * 2. The `spark.rapids.sql.enabled` property is set to true (or defaults to true)
   *
   * This is one of the most commonly used predicates, as many tuning rules only
   * make sense when GPU acceleration is active.
   *
   * == Example Usage ==
   * {{{
   * class GpuMemoryTuningRule extends BaseTuningRule {
   *   override def condition: ConditionTrait[AutoTuner] = {
   *     TuningCondPredicates.gpuEnabled
   *   }
   *
   *   override def apply(tuner: AutoTuner): Unit = {
   *     // GPU-specific tuning logic
   *   }
   * }
   * }}}
   *
   * @return A condition that returns true if GPU acceleration is enabled
   */
  val gpuEnabled: ConditionTrait[AutoTuner] = {
    (tuner: AutoTuner) => {
      tuner.appInfoProvider.getProperty("spark.plugins") match {
        case Some(plugins) if plugins.contains("com.nvidia.spark.SQLPlugin") =>
          tuner.appInfoProvider.getProperty("spark.rapids.sql.enabled")
            .getOrElse("true")
            .equalsIgnoreCase("true")
        case _ => false
      }
    }
  }

  /**
   * Checks if GPU acceleration is NOT enabled in the application.
   *
   * This is the logical negation of `gpuEnabled`. It's useful for rules that should
   * only apply to CPU-only Spark applications or for rules that recommend enabling
   * GPU acceleration.
   *
   * == Example Usage ==
   * {{{
   * class EnableGpuRecommendationRule extends BaseTuningRule {
   *   override def condition: ConditionTrait[AutoTuner] = {
   *     TuningCondPredicates.gpuNotEnabled
   *   }
   *
   *   override def apply(tuner: AutoTuner): Unit = {
   *     // Recommend enabling GPU for applicable workloads
   *   }
   * }
   * }}}
   *
   * @return A condition that returns true if GPU acceleration is not enabled
   */
  val gpuNotEnabled: ConditionTrait[AutoTuner] = {
    new NotCondition(gpuEnabled)
  }
}
