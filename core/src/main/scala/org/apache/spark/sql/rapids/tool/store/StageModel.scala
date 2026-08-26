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

package org.apache.spark.sql.rapids.tool.store

import com.nvidia.spark.rapids.tool.profiling.ProfileUtils

import org.apache.spark.scheduler.StageInfo
import org.apache.spark.sql.rapids.tool.annotation.{Calculated, Since, WallClock}
import org.apache.spark.sql.rapids.tool.util.stubs.StageInfoStub

/**
 * StageModel is a class to store the information of a stage.
 * Typically, a new instance is created while handling StageSubmitted/StageCompleted events.
 *
 * @param sInfo Snapshot from the stage info loaded from the eventlog
 */

@Since("24.02.3")
class StageModel private(sInfo: StageInfo) {

  var stageInfo: StageInfoStub = _
  updateInfo(sInfo)

  /**
   * @param newStageInfo
   * @return a new StageInfo object.
   * TODO: https://github.com/NVIDIA/cudf-spark-tools/issues/1260
   */
  private def initStageInfo(newStageInfo: StageInfo): StageInfoStub = {
    StageInfoStub.fromStageInfo(newStageInfo)
  }

  @WallClock
  @Calculated("Calculated as (submissionTime - completionTime)")
  var duration: Option[Long] = None

  /**
   * Updates the snapshot of Spark's stageInfo to point to the new value and recalculate the
   * duration. Typically, a new StageInfo object is created with both StageSubmitted/StageCompleted
   * events
   * @param newStageInfo Spark's StageInfo loaded from StageSubmitted/StageCompleted events.
   */
  private def updateInfo(newStageInfo: StageInfo): Unit = {
    // TODO issue: https://github.com/NVIDIA/cudf-spark-tools/issues/1260
    stageInfo = initStageInfo(newStageInfo)
    calculateDuration()
  }

  /**
   * Calculate the duration of the stage.
   * This is called automatically whenever the stage info is updated.
   */
  private def calculateDuration(): Unit = {
    duration =
      ProfileUtils.optionLongMinusOptionLong(stageInfo.completionTime, stageInfo.submissionTime)
  }

  /**
   * Returns true if a stage attempt has failed.
   * There can be multiple attempts( retries ) of a stage
   * that can fail until the last attempt succeeds.
   *
   * @return true if a stage attempt has failed.
   */
  def hasFailed: Boolean = {
    stageInfo.failureReason.isDefined
  }

  /**
   * Returns the failure reason if the stage has failed.
   * Failure reason being set is the sure shot of a failed stage.
   *
   * @return the failure reason if the stage has failed, or an empty string otherwise
   */
  def getFailureReason: String = {
    stageInfo.failureReason.getOrElse("")
  }

  /**
   * Duration won't be defined when neither submitted/completion-Time is defined.
   *
   * @return the WallClock duration of the stage in milliseconds if defined, or 0L otherwise.
   */
  @Calculated
  @WallClock
  def getDuration: Long = {
    duration.getOrElse(0L)
  }

  def getId: Int = {
    stageInfo.stageId
  }

  def getAttemptId: Int = {
    stageInfo.attemptNumber()
  }
}

object StageModel {
  /**
   * Factory method to create a new instance of StageModel.
   * The purpose of this method is to encapsulate the logic of updating the stageModel based on
   * the argument.
   * Note that this encapsulation is added to avoid a bug when the Spark's stageInfo was not
   * updated correctly when an event was triggered. This resulted in the stageInfo pointing to an
   * outdated Spark's StageInfo.
   * 1- For a new StageModel: this could be triggered by either stageSubmitted event; or
   *    stageCompleted event.
   * 2- For an existing StageModel: the stageInfo argument is not the same object captured when the
   *    stageModel was created. In that case, we need to call updateInfo to point to the new Spark's
   *    StageInfo and re-calculate the duration.
   * @param stageInfo Spark's StageInfo captured from StageSubmitted/StageCompleted events
   * @param stageModel Option of StageModel represents the existing instance of StageModel that was
   *                   created when the stage was submitted.
   * @return a new instance of StageModel if it exists, or returns the existing StageModel after
   *         updating its sInfo and duration fields.
   */
  def apply(stageInfo: StageInfo, stageModel: Option[StageModel]): StageModel = {
    val sModel = stageModel match {
      case Some(existingStageModel) =>
        existingStageModel.updateInfo(stageInfo)
        existingStageModel
      case None => new StageModel(stageInfo)
    }
    sModel
  }
}
