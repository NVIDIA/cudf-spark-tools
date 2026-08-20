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

package com.nvidia.spark.rapids.tool.plugins

import org.apache.spark.internal.Logging

/**
 * A trait that defines a condition that can be evaluated against a target object.
 *
 * This trait provides a foundation for building composable, reusable conditions
 * that can be used in plugin activation logic, rule execution, and other
 * decision-making contexts throughout the cuDF plugin tools framework.
 *
 * Conditions are type-parameterized to work with different target types (e.g.,
 * AutoTuner, ApplicationInfo, PlatformContext) and can be combined using
 * logical operators (AND, OR, NOT) to create complex evaluation logic.
 *
 * @tparam T The type of object this condition evaluates against
 *
 * Example usage:
 * {{{
 *   class IsPlatformCondition(platform: String) extends ConditionTrait[AutoTuner] {
 *     override def eval(target: AutoTuner): Boolean = {
 *       target.platform.platformName == platform
 *     }
 *   }
 *
 *   val isDatabricks = new IsPlatformCondition("databricks")
 *   val isEnabled = isDatabricks.eval(autoTuner)
 * }}}
 */
trait ConditionTrait[T] extends Logging {
  /**
   * Evaluates this condition against the given target object.
   *
   * @param target The object to evaluate this condition against
   * @return true if the condition is satisfied, false otherwise
   */
  def eval(target: T): Boolean
}

/**
 * A condition that always evaluates to true regardless of the target object.
 *
 * This is useful as a default or placeholder condition when a plugin should
 * always be active, or when creating conditional logic that needs a fallback
 * true case.
 *
 * @tparam T The type of object this condition evaluates against
 *
 * Example usage:
 * {{{
 *   val alwaysActive = new AlwaysTrueCondition[AutoTuner]
 *   assert(alwaysActive.eval(anyAutoTuner) == true)
 * }}}
 */
class AlwaysTrueCondition[T] extends ConditionTrait[T] {
  override def eval(target: T): Boolean = true
}

/**
 * A composite condition that evaluates to true if ANY of its child conditions are true.
 *
 * This implements logical OR semantics, evaluating conditions in sequence and
 * short-circuiting (returning true) as soon as any condition evaluates to true.
 * If all conditions are false, the overall result is false. An empty sequence
 * of conditions evaluates to false.
 *
 * @param conditions The sequence of conditions to evaluate with OR logic
 * @tparam T The type of object this condition evaluates against
 *
 * Example usage:
 * {{{
 *   val isDatabricks = new IsPlatformCondition("databricks")
 *   val isDataproc = new IsPlatformCondition("dataproc")
 *   val isCloudPlatform = new OrCondition(Seq(isDatabricks, isDataproc))
 *   // Returns true if platform is either Databricks OR Dataproc
 * }}}
 */
class OrCondition[T](conditions: Seq[ConditionTrait[T]]) extends ConditionTrait[T] {
  override def eval(target: T): Boolean = {
    conditions.exists(_.eval(target))
  }
}

/**
 * A composite condition that evaluates to true only if ALL of its child conditions are true.
 *
 * This implements logical AND semantics, evaluating conditions in sequence and
 * short-circuiting (returning false) as soon as any condition evaluates to false.
 * If all conditions are true, the overall result is true. An empty sequence
 * of conditions evaluates to true.
 *
 * @param conditions The sequence of conditions to evaluate with AND logic
 * @tparam T The type of object this condition evaluates against
 *
 * Example usage:
 * {{{
 *   val isDatabricks = new IsPlatformCondition("databricks")
 *   val hasGPU = new HasGPUCondition()
 *   val requiresBoth = new AndCondition(Seq(isDatabricks, hasGPU))
 *   // Returns true only if platform is Databricks AND GPUs are available
 * }}}
 */
class AndCondition[T](conditions: Seq[ConditionTrait[T]]) extends ConditionTrait[T] {
  override def eval(target: T): Boolean = {
    conditions.forall(_.eval(target))
  }
}

/**
 * A condition that inverts the result of another condition.
 *
 * This implements logical NOT semantics, evaluating the wrapped condition
 * and returning the opposite result. If the wrapped condition is true,
 * this returns false, and vice versa.
 *
 * @param condition The condition to negate
 * @tparam T The type of object this condition evaluates against
 *
 * Example usage:
 * {{{
 *   val isDatabricks = new IsPlatformCondition("databricks")
 *   val isNotDatabricks = new NotCondition(isDatabricks)
 *   // Returns true if platform is NOT Databricks
 * }}}
 */
class NotCondition[T](condition: ConditionTrait[T]) extends ConditionTrait[T] {
  override def eval(target: T): Boolean = {
    !condition.eval(target)
  }
}
