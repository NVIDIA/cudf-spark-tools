/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.tool

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

import com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent
import com.nvidia.spark.rapids.tool.profiling.{BlockManagerRemovedCase, DriverAccumCase, JobInfoClass, ProfileUtils, ResourceProfileInfoCase, SQLExecutionInfoClass, SQLPlanMetricsCase}

import org.apache.spark.internal.Logging
import org.apache.spark.scheduler._
import org.apache.spark.sql.execution.ui._
import org.apache.spark.sql.rapids.tool.util.{ConnectEventHandler, EventUtils, StringUtils}
import org.apache.spark.sql.streaming.StreamingQueryListener

abstract class EventProcessorBase[T <: AppBase](app: T) extends SparkListener with Logging {

  def processAnyEvent(event: SparkListenerEvent): Unit = {
    event match {
      case _: SparkListenerLogStart =>
        doSparkListenerLogStart(app, event.asInstanceOf[SparkListenerLogStart])
      case _: SparkListenerBlockManagerAdded =>
        doSparkListenerBlockManagerAdded(app,
          event.asInstanceOf[SparkListenerBlockManagerAdded])
      case _: SparkListenerBlockManagerRemoved =>
        doSparkListenerBlockManagerRemoved(app,
          event.asInstanceOf[SparkListenerBlockManagerRemoved])
      case _: SparkListenerEnvironmentUpdate =>
        doSparkListenerEnvironmentUpdate(app,
          event.asInstanceOf[SparkListenerEnvironmentUpdate])
      case _: SparkListenerApplicationStart =>
        doSparkListenerApplicationStart(app,
          event.asInstanceOf[SparkListenerApplicationStart])
      case _: SparkListenerApplicationEnd =>
        doSparkListenerApplicationEnd(app,
          event.asInstanceOf[SparkListenerApplicationEnd])
      case _: SparkListenerExecutorAdded =>
        doSparkListenerExecutorAdded(app,
          event.asInstanceOf[SparkListenerExecutorAdded])
      case _: SparkListenerExecutorRemoved =>
        doSparkListenerExecutorRemoved(app,
          event.asInstanceOf[SparkListenerExecutorRemoved])
      case _: SparkListenerTaskStart =>
        doSparkListenerTaskStart(app,
          event.asInstanceOf[SparkListenerTaskStart])
      case _: SparkListenerTaskEnd =>
        doSparkListenerTaskEnd(app,
          event.asInstanceOf[SparkListenerTaskEnd])
      case _: SparkListenerSQLExecutionStart =>
        doSparkListenerSQLExecutionStart(app,
          event.asInstanceOf[SparkListenerSQLExecutionStart])
      case _: SparkListenerSQLExecutionEnd =>
        doSparkListenerSQLExecutionEnd(app,
          event.asInstanceOf[SparkListenerSQLExecutionEnd])
      case _: SparkListenerDriverAccumUpdates =>
        doSparkListenerDriverAccumUpdates(app,
          event.asInstanceOf[SparkListenerDriverAccumUpdates])
      case _: SparkListenerJobStart =>
        doSparkListenerJobStart(app,
          event.asInstanceOf[SparkListenerJobStart])
      case _: SparkListenerJobEnd =>
        doSparkListenerJobEnd(app,
          event.asInstanceOf[SparkListenerJobEnd])
      case _: SparkListenerStageSubmitted =>
        doSparkListenerStageSubmitted(app,
          event.asInstanceOf[SparkListenerStageSubmitted])
      case _: SparkListenerStageCompleted =>
        doSparkListenerStageCompleted(app,
          event.asInstanceOf[SparkListenerStageCompleted])
      case _: SparkListenerStageExecutorMetrics =>
        doSparkListenerStageExecutorMetrics(app,
          event.asInstanceOf[SparkListenerStageExecutorMetrics])
      case _: SparkListenerTaskGettingResult =>
        doSparkListenerTaskGettingResult(app,
          event.asInstanceOf[SparkListenerTaskGettingResult])
      case _: SparkListenerSQLAdaptiveExecutionUpdate =>
        doSparkListenerSQLAdaptiveExecutionUpdate(app,
          event.asInstanceOf[SparkListenerSQLAdaptiveExecutionUpdate])
      case _: SparkListenerSQLAdaptiveSQLMetricUpdates =>
        doSparkListenerSQLAdaptiveSQLMetricUpdates(app,
          event.asInstanceOf[SparkListenerSQLAdaptiveSQLMetricUpdates])
      case _: StreamingQueryListener.QueryStartedEvent =>
        doSparkListenerStreamingQuery(app,
          event.asInstanceOf[StreamingQueryListener.QueryStartedEvent])
      case _: StreamingQueryListener.QueryTerminatedEvent =>
        doSparkListenerStreamingQuery(app,
          event.asInstanceOf[StreamingQueryListener.QueryTerminatedEvent])
      case _: SparkRapidsBuildInfoEvent =>
        doSparkRapidsBuildInfoEvent(app,
          event.asInstanceOf[SparkRapidsBuildInfoEvent])
      case _ =>
        val wasResourceProfileAddedEvent = doSparkListenerResourceProfileAddedReflect(app, event)
        if (!wasResourceProfileAddedEvent) {
          if (!ConnectEventHandler.processConnectEvent(app, event)) {
            doOtherEvent(app, event)
          }
        }
    }
  }

  def doSparkListenerResourceProfileAddedReflect(
      app: T,
      event: SparkListenerEvent): Boolean = {
    val rpAddedClass = "org.apache.spark.scheduler.SparkListenerResourceProfileAdded"
    if (event.getClass.getName.equals(rpAddedClass)) {
      try {
        event match {
          case rpAdded: SparkListenerResourceProfileAdded =>
            doSparkListenerResourceProfileAdded(app, rpAdded)
            true
          case _ => false
        }
      } catch {
        case _: ClassNotFoundException =>
          logWarning("Error trying to parse SparkListenerResourceProfileAdded, Spark" +
            " version likely older than 3.1.X, unable to parse it properly.")
          false
      }
    } else {
      false
    }
  }

  def doSparkListenerLogStart(
      app: T,
      event: SparkListenerLogStart): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.handleLogStartForCachedProps(event)
  }

  def doSparkListenerSQLExecutionStart(
      app: T,
      event: SparkListenerSQLExecutionStart): Unit = {
    logDebug("Processing event: " + event.getClass)
    val rootExecutionIdOpt = EventUtils.readRootIDFromSQLStartEvent(event)
    // Extract modifiedConfigs: per-SQL-execution config overrides set via
    // spark.conf.set(). Available in Spark 3.3+ (SPARK-34735).
    // Returns empty map for older Spark versions.
    val modifiedConfigs = EventUtils.readModifiedConfigsFromSQLStartEvent(event)
    val sqlExecution = new SQLExecutionInfoClass(
      event.executionId,
      rootExecutionIdOpt,
      event.description,
      event.details,
      event.time,
      None,
      None,
      hasDatasetOrRDD = false,
      modifiedConfigs = modifiedConfigs
    )
    app.sqlIdToInfo.put(event.executionId, sqlExecution)
    app.sqlManager.addNewExecution(event.executionId, event.sparkPlanInfo,
      event.physicalPlanDescription)

    // Merge runtime config overrides into app-level sparkProperties with
    // redaction and predicate updates (gpuMode, etc.).
    // Last-write-wins if multiple SQL executions have different modifiedConfigs.
    app.mergeModifiedConfigs(modifiedConfigs)

    // Correlate Connect operations to this sqlID via jobTags (Spark 3.5+).
    if (app.isConnectMode) {
      EventUtils.readJobTagsFromSQLStartEvent(event).foreach { tag =>
        app.jobTagToConnectOpId.get(tag).foreach { opId =>
          app.operationIdToSqlIds
            .getOrElseUpdate(opId, mutable.HashSet.empty[Long])
            .add(event.executionId)
        }
      }
    }
  }

  def doSparkListenerSQLExecutionEnd(
      app: T,
      event: SparkListenerSQLExecutionEnd): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.sqlIdToInfo.get(event.executionId).foreach { sql =>
      sql.endTime = Some(event.time)
      sql.duration = ProfileUtils.OptionLongMinusLong(sql.endTime, sql.startTime)
    }
  }

  def doSparkListenerDriverAccumUpdates(
      app: T,
      event: SparkListenerDriverAccumUpdates): Unit = {
    logDebug("Processing event: " + event.getClass)

    val SparkListenerDriverAccumUpdates(sqlID, accumUpdates) = event
    accumUpdates.foreach { accum =>
      val driverAccum = DriverAccumCase(sqlID, accum._1, accum._2)
      val arrBuf = app.driverAccumMap.getOrElseUpdate(accum._1,
        ArrayBuffer[DriverAccumCase]())
      arrBuf += driverAccum
    }
  }

  def doSparkListenerSQLAdaptiveExecutionUpdate(
      app: T,
      event: SparkListenerSQLAdaptiveExecutionUpdate): Unit = {
    // AQE plan can override the ones got from SparkListenerSQLExecutionStart
    app.sqlManager.addAQE(event.executionId, event.sparkPlanInfo,
      event.physicalPlanDescription)
  }

  def doSparkListenerSQLAdaptiveSQLMetricUpdates(
      app: T,
      event: SparkListenerSQLAdaptiveSQLMetricUpdates): Unit = {
    logDebug("Processing event: " + event.getClass)
    val SparkListenerSQLAdaptiveSQLMetricUpdates(sqlID, sqlPlanMetrics) = event
    val metrics = sqlPlanMetrics.map { metric =>
      SQLPlanMetricsCase(sqlID, metric.name,
        metric.accumulatorId, metric.metricType)
    }
    app.sqlPlanMetricsAdaptive ++= metrics
  }

  def doSparkRapidsBuildInfoEvent(
      app: T,
      event: SparkRapidsBuildInfoEvent): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.sparkRapidsBuildInfo = event
  }

  override def onOtherEvent(event: SparkListenerEvent): Unit = event match {
    case e: SparkListenerSQLExecutionStart =>
      doSparkListenerSQLExecutionStart(app, e)
    case e: SparkListenerSQLAdaptiveExecutionUpdate =>
      doSparkListenerSQLAdaptiveExecutionUpdate(app, e)
    case e: SparkListenerSQLAdaptiveSQLMetricUpdates =>
      doSparkListenerSQLAdaptiveSQLMetricUpdates(app, e)
    case e: SparkListenerSQLExecutionEnd =>
      doSparkListenerSQLExecutionEnd(app, e)
    case e: SparkListenerDriverAccumUpdates =>
      doSparkListenerDriverAccumUpdates(app, e)
    case e: SparkListenerLogStart =>
      doSparkListenerLogStart(app, e)
    case _ =>
      val wasResourceProfileAddedEvent = doSparkListenerResourceProfileAddedReflect(app, event)
      if (!wasResourceProfileAddedEvent) {
        if (!ConnectEventHandler.processConnectEvent(app, event)) {
          doOtherEvent(app, event)
        }
      }
  }

  def doSparkListenerResourceProfileAdded(
      app: T,
      event: SparkListenerResourceProfileAdded): Unit = {
    // leave off maxTasks for now
    val rp = ResourceProfileInfoCase(event.resourceProfile.id,
      event.resourceProfile.executorResources, event.resourceProfile.taskResources)
    app.resourceProfIdToInfo(event.resourceProfile.id) = rp
  }

  override def onResourceProfileAdded(event: SparkListenerResourceProfileAdded): Unit = {
    doSparkListenerResourceProfileAdded(app, event)
  }

  def doSparkListenerBlockManagerAdded(
      app: T,
      event: SparkListenerBlockManagerAdded): Unit = {
    logDebug("Processing event: " + event.getClass)
    val execExists = app.executorIdToInfo.get(event.blockManagerId.executorId)
    if (event.blockManagerId.executorId == "driver" && execExists.isEmpty) {
      // means its not in local mode, skip counting as executor
    } else {
      // note that one block manager is for driver as well
      val exec = app.getOrCreateExecutor(event.blockManagerId.executorId, event.time)
      exec.hostPort = event.blockManagerId.hostPort
      event.maxOnHeapMem.foreach { mem =>
        exec.totalOnHeap = mem
      }
      event.maxOffHeapMem.foreach { offHeap =>
        exec.totalOffHeap = offHeap
      }
      exec.isActive = true
      exec.maxMemory = event.maxMem
    }
  }

  def doSparkListenerStreamingQuery(
      app: T,
      event: SparkListenerEvent): Unit = {
    throw StreamingEventLogException()
  }

  override def onBlockManagerAdded(blockManagerAdded: SparkListenerBlockManagerAdded): Unit = {
    doSparkListenerBlockManagerAdded(app, blockManagerAdded)
  }

  def doSparkListenerBlockManagerRemoved(
      app: T,
      event: SparkListenerBlockManagerRemoved): Unit = {
    val thisBlockManagerRemoved = BlockManagerRemovedCase(
      event.blockManagerId.executorId,
      event.blockManagerId.host,
      event.blockManagerId.port,
      event.time
    )
    app.blockManagersRemoved += thisBlockManagerRemoved
  }

  override def onBlockManagerRemoved(
      blockManagerRemoved: SparkListenerBlockManagerRemoved): Unit = {
    doSparkListenerBlockManagerRemoved(app, blockManagerRemoved)
  }

  def doSparkListenerEnvironmentUpdate(
      app: T,
      event: SparkListenerEnvironmentUpdate): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.handleEnvUpdateForCachedProps(event)
    app.registerAttemptId()
  }

  override def onEnvironmentUpdate(environmentUpdate: SparkListenerEnvironmentUpdate): Unit = {
    doSparkListenerEnvironmentUpdate(app, environmentUpdate)
  }

  def doSparkListenerApplicationStart(
      app: T,
      event: SparkListenerApplicationStart): Unit = {
    app.appMetaData = Some(AppMetaData(app.getEventLogPath, event))
  }

  override def onApplicationStart(applicationStart: SparkListenerApplicationStart): Unit = {
    doSparkListenerApplicationStart(app, applicationStart)
  }

  def doSparkListenerApplicationEnd(
      app: T,
      event: SparkListenerApplicationEnd): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.updateEndTime(event.time)
  }

  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
    doSparkListenerApplicationEnd(app, applicationEnd)
  }

  def doSparkListenerExecutorAdded(
      app: T,
      event: SparkListenerExecutorAdded): Unit = {
    logDebug("Processing event: " + event.getClass)
    val exec = app.getOrCreateExecutor(event.executorId, event.time)
    exec.host = event.executorInfo.executorHost
    exec.isActive = true
    exec.totalCores = event.executorInfo.totalCores
    val rpId = event.executorInfo.resourceProfileId
    exec.resources = event.executorInfo.resourcesInfo
    exec.resourceProfileId = rpId
    // make sure updateMaxExecutorsIfNeeded is called after added executor information
    app.updateMaxExecutors()
    app.updateMaxNodes()
  }

  override def onExecutorAdded(executorAdded: SparkListenerExecutorAdded): Unit = {
    doSparkListenerExecutorAdded(app, executorAdded)
  }

  def doSparkListenerExecutorRemoved(
      app: T,
      event: SparkListenerExecutorRemoved): Unit = {
    logDebug("Processing event: " + event.getClass)
    val exec = app.getOrCreateExecutor(event.executorId, event.time)
    exec.isActive = false
    exec.removeTime = event.time
    exec.removeReason =
      StringUtils.renderStr(event.reason, doEscapeMetaCharacters = true, maxLength = 0)
  }

  override def onExecutorRemoved(executorRemoved: SparkListenerExecutorRemoved): Unit = {
    doSparkListenerExecutorRemoved(app, executorRemoved)
  }

  def doSparkListenerTaskStart(
      app: T,
      event: SparkListenerTaskStart): Unit = {}

  override def onTaskStart(taskStart: SparkListenerTaskStart): Unit = {
    doSparkListenerTaskStart(app, taskStart)
  }

  def doSparkListenerTaskEnd(
      app: T,
      event: SparkListenerTaskEnd): Unit = {
    // TODO: this implementation needs to be updated to use attemptID
    // Parse task accumulables
    for (res <- event.taskInfo.accumulables) {
      try {
        app.accumManager.addAccToTask(event.stageId, res)
      } catch {
        case NonFatal(e) =>
          logWarning("Exception when parsing accumulables on task-completed "
            + "stageID=" + event.stageId + ",taskId=" + event.taskInfo.taskId
            + ": ")
          logWarning(e.toString)
          logWarning("The problematic accumulable is: name="
            + res.name + ",value=" + res.value + ",update=" + res.update)
      }
    }
    // Create Task Objects and update the taskEnd dataStructure.
    app.taskManager.addTaskFromEvent(event)
  }

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    doSparkListenerTaskEnd(app, taskEnd)
  }

  def doSparkListenerJobStart(
      app: T,
      event: SparkListenerJobStart): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.handleJobStartForCachedProps(event)
    val sqlIDString = event.properties.getProperty("spark.sql.execution.id")
    val sqlID = StringUtils.stringToLong(sqlIDString)
    if (sqlID.nonEmpty) {
      app.sqlIdToStages.getOrElseUpdate(sqlID.get, ArrayBuffer.empty) ++= event.stageIds
    }
    sqlID.foreach(app.jobIdToSqlID(event.jobId) = _)

    // Correlate Connect operations to this jobID via spark.job.tags.
    if (app.isConnectMode) {
      val tagStr = event.properties.getProperty("spark.job.tags")
      if (tagStr != null && tagStr.nonEmpty) {
        tagStr.split(",").iterator.map(_.trim).filter(_.nonEmpty).foreach { tag =>
          app.jobTagToConnectOpId.get(tag).foreach { opId =>
            app.operationIdToJobIds
              .getOrElseUpdate(opId, mutable.HashSet.empty[Int])
              .add(event.jobId)
          }
        }
      }
    }
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    doSparkListenerJobStart(app, jobStart)
  }

  def doSparkListenerJobEnd(
      app: T,
      event: SparkListenerJobEnd): Unit = {
    logDebug("Processing event: " + event.getClass)

    def jobResult(res: JobResult): String = {
      res match {
        case JobSucceeded => "JobSucceeded"
        case _: JobFailed => "JobFailed"
        case _ => "Unknown"
      }
    }

    def failedReason(res: JobResult): String = {
      res match {
        case JobSucceeded => ""
        case jobFailed: JobFailed => jobFailed.exception.toString
        case _ => ""
      }
    }

    app.jobIdToInfo.get(event.jobId) match {
      case Some(j) =>
        j.endTime = Some(event.time)
        j.duration = ProfileUtils.OptionLongMinusLong(j.endTime, j.startTime)
        val thisJobResult = jobResult(event.jobResult)
        j.jobResult = Some(thisJobResult)
        val thisFailedReason = failedReason(event.jobResult)
        j.failedReason = Some(thisFailedReason)
      case None =>
        val thisJobResult = jobResult(event.jobResult)
        val thisFailedReason = failedReason(event.jobResult)
        val thisJob = new JobInfoClass(
          event.jobId,
          Seq.empty,
          None,
          Map.empty,
          event.time,  // put end time as start time
          Some(event.time),
          Some(thisJobResult),
          Some(thisFailedReason),
          None,
          app.gpuMode
        )
        app.jobIdToInfo.put(event.jobId, thisJob)
    }
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
    doSparkListenerJobEnd(app, jobEnd)
  }

  def doSparkListenerStageSubmitted(
      app: T,
      event: SparkListenerStageSubmitted): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.getOrCreateStage(event.stageInfo)
  }

  override def onStageSubmitted(stageSubmitted: SparkListenerStageSubmitted): Unit = {
    doSparkListenerStageSubmitted(app, stageSubmitted)
  }

  def doSparkListenerStageCompleted(
      app: T,
      event: SparkListenerStageCompleted): Unit = {
    logDebug("Processing event: " + event.getClass)
    app.getOrCreateStage(event.stageInfo)
    // TODO: Should accumulators be added only if the stage is successful?
    // Parse stage accumulables
    for (res <- event.stageInfo.accumulables) {
      try {
        val accumInfo = res._2
        app.accumManager.addAccToStage(event.stageInfo.stageId, accumInfo)
      } catch {
        case NonFatal(e) =>
          logWarning("Exception when parsing accumulables on stage-completed " +
            "stageID=" + event.stageInfo.stageId + ": ")
          logWarning(e.toString)
          logWarning("The problematic accumulable is: name="
            + res._2.name + ",value=" + res._2.value + ",update=" + res._2.update)
      }
    }
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    doSparkListenerStageCompleted(app, stageCompleted)
  }

  def doSparkListenerStageExecutorMetrics(
      app: T,
      event: SparkListenerStageExecutorMetrics): Unit = {
    logDebug("Processing event: " + event.getClass)
    val pythonVMemory = event.executorMetrics.getMetricValue("ProcessTreePythonVMemory")
    app.stageExecutorMetricsManager.addPythonVMemory(
      event.stageId, event.stageAttemptId, event.execId, pythonVMemory)
  }

  override def onStageExecutorMetrics(event: SparkListenerStageExecutorMetrics): Unit = {
    doSparkListenerStageExecutorMetrics(app, event)
  }

  def doSparkListenerTaskGettingResult(
      app: T,
      event: SparkListenerTaskGettingResult): Unit = {}

  override def onTaskGettingResult(taskGettingResult: SparkListenerTaskGettingResult): Unit = {
    doSparkListenerTaskGettingResult(app, taskGettingResult)
  }

  // To process all other unknown events
  def doOtherEvent(
      app: T,
      event: SparkListenerEvent): Unit = {}
}
