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

package org.apache.spark.sql.rapids.tool.store

import scala.collection.mutable

/** Retains per-executor Python virtual-memory peaks for exact stage attempts. */
class StageExecutorMetricsManager {
  private val stageIdToAttempts = mutable.HashMap.empty[
    Int, mutable.HashMap[Int, mutable.HashMap[String, Long]]]

  def addPythonVMemory(
      stageId: Int,
      stageAttemptId: Int,
      executorId: String,
      pythonVMemory: Long): Unit = {
    if (executorId != "driver" && pythonVMemory > 0L) {
      val attempts = stageIdToAttempts.getOrElseUpdate(stageId, mutable.HashMap.empty)
      val executors = attempts.getOrElseUpdate(stageAttemptId, mutable.HashMap.empty)
      executors.update(executorId, math.max(executors.getOrElse(executorId, 0L), pythonVMemory))
    }
  }

  /** Returns a snapshot of executor peaks for the requested exact stage attempt. */
  def getPythonVMemory(stageId: Int, stageAttemptId: Int): Map[String, Long] = {
    stageIdToAttempts.get(stageId).flatMap(_.get(stageAttemptId)).map(_.toMap).getOrElse(Map.empty)
  }

  /** Removes every attempt retained for the supplied stage IDs. */
  def removeStages(stageIds: Iterable[Int]): Unit = {
    stageIdToAttempts --= stageIds
  }
}
