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

package com.nvidia.spark.rapids.tool.analysis

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.nvidia.spark.rapids.tool.EventLogPathProcessor
import com.nvidia.spark.rapids.tool.profiling.{AppAggGpuMetricsProfileResult,
  StageAggGpuMetricsProfileResult}
import com.nvidia.spark.rapids.tool.views.RawMetricProfilerView
import org.apache.hadoop.conf.Configuration
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo

/**
 * Report-level tests for the GPU metric tables over a hand-built event log.
 *
 * The stored fixtures carry neither gpuMaxTaskFootprint nor a stage whose accumulable arrives
 * only at completion, so the emitted contract for those two cases cannot be exercised from
 * them.
 */
class GpuMetricReportSuite extends AnyFunSuite {

  private val hadoopConf = new Configuration()

  private def taskAccum(id: Int, name: String, update: Long, value: Long): String =
    s"""{"ID":$id,"Name":"$name","Update":$update,"Value":$value,""" +
      """"Internal":true,"Count Failed Values":true}"""

  private def stageAccum(id: Int, name: String, value: Long): String =
    s"""{"ID":$id,"Name":"$name","Value":$value,""" +
      """"Internal":true,"Count Failed Values":true}"""

  private def stageInfo(stageId: Int, numTasks: Int, accums: Seq[String]): String =
    s"""{"Stage ID":$stageId,"Stage Attempt ID":0,"Stage Name":"stage$stageId",""" +
      s""""Number of Tasks":$numTasks,"RDD Info":[],"Parent IDs":[],"Details":"",""" +
      s""""Submission Time":1000,"Completion Time":2000,"Resource Profile Id":0,""" +
      s""""Accumulables":[${accums.mkString(",")}]}"""

  private def stageSubmitted(stageId: Int, numTasks: Int): String =
    s"""{"Event":"SparkListenerStageSubmitted","Stage Info":""" +
      s"""${stageInfo(stageId, numTasks, Seq.empty)}}"""

  private def stageCompleted(stageId: Int, numTasks: Int, accums: Seq[String]): String =
    s"""{"Event":"SparkListenerStageCompleted","Stage Info":""" +
      s"""${stageInfo(stageId, numTasks, accums)}}"""

  // An ExceptionFailure needs its class, description and stack trace to deserialize.
  private val failureReason =
    """{"Reason":"ExceptionFailure","Class Name":"java.lang.RuntimeException",""" +
      """"Description":"synthetic","Stack Trace":[],"Full Stack Trace":"synthetic",""" +
      """"Accumulator Updates":[]}"""

  private def taskEnd(stageId: Int, taskId: Int, accums: Seq[String],
      failed: Boolean = false): String =
    s"""{"Event":"SparkListenerTaskEnd","Stage ID":$stageId,"Stage Attempt ID":0,""" +
      s""""Task Type":"ResultTask","Task End Reason":""" +
      s"""${if (failed) failureReason else """{"Reason":"Success"}"""},"Task Info":""" +
      s"""{"Task ID":$taskId,"Index":$taskId,"Attempt":0,"Partition ID":$taskId,""" +
      s""""Launch Time":1000,"Executor ID":"1","Host":"h","Locality":"PROCESS_LOCAL",""" +
      s""""Speculative":false,"Getting Result Time":0,"Finish Time":1500,""" +
      s""""Failed":$failed,"Killed":false,"Accumulables":[${accums.mkString(",")}]}}"""

  private def withEventLog(events: Seq[String])(verify: ApplicationInfo => Unit): Unit = {
    TrampolineUtil.withTempDir { dir =>
      val path = Paths.get(dir.getAbsolutePath, "gpu_metric_eventlog")
      Files.write(path, events.mkString("\n").getBytes(StandardCharsets.UTF_8))
      val info = EventLogPathProcessor.getEventLogInfo(path.toString, hadoopConf).head._1
      verify(new ApplicationInfo(hadoopConf, info))
    }
  }

  private val header = Seq(
    """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}""",
    """{"Event":"SparkListenerApplicationStart","App Name":"gpuMetricReport",""" +
      """"App ID":"app-gpu-metric-report","Timestamp":100,"User":"test"}""")

  private def gpuRows(app: ApplicationInfo): Seq[StageAggGpuMetricsProfileResult] =
    RawMetricProfilerView.getAggMetrics(Seq(app)).gpuStageAggs

  private def appRows(app: ApplicationInfo): Seq[AppAggGpuMetricsProfileResult] =
    RawMetricProfilerView.getAggMetrics(Seq(app)).gpuAppAggs

  private def row(rows: Seq[StageAggGpuMetricsProfileResult], stageId: Int, name: String) =
    rows.find(r => r.stageId == stageId && r.metricName == name)

  test("gpuMaxTaskFootprint reports the spread across the attempts that carried it") {
    // Stage 1 runs four tasks; only three report the footprint, one of them on a failed attempt.
    val footprints = Seq(1000000000L, 2000000000L, 6000000000L)
    val events = header ++ Seq(
      stageSubmitted(1, 4),
      taskEnd(1, 0, Seq(taskAccum(10, "gpuMaxTaskFootprint", footprints(0), footprints(0)))),
      taskEnd(1, 1, Seq(taskAccum(10, "gpuMaxTaskFootprint", footprints(1), footprints(1)))),
      taskEnd(1, 2, Seq(taskAccum(10, "gpuMaxTaskFootprint", footprints(2), footprints(2))),
        failed = true),
      taskEnd(1, 3, Seq(taskAccum(11, "gpuTime", 5L, 5L))),
      stageCompleted(1, 4, Seq(stageAccum(10, "gpuMaxTaskFootprint", 6000000000L))))
    withEventLog(events) { app =>
      val r = row(gpuRows(app), 1, "gpuMaxTaskFootprint")
        .getOrElse(fail("no gpuMaxTaskFootprint row"))
      assert(r.numTasks == 4, "numTasks counts every attempt in the stage")
      // The failed attempt reported, so it is one of the samples.
      assert(r.count == 3L, "sampleCount counts the attempts that reported, not the tasks")
      assert(r.min.contains(footprints.min) && r.max.contains(footprints.max))
      assert(r.unit == "bytes", "the catalog declares this metric in bytes")
      val mean = footprints.sum.toDouble / footprints.size
      assert(r.avg.exists(v => Math.abs(v - mean) < 1e-6))
      val ss = footprints.map(v => (v - mean) * (v - mean)).sum
      val expected = Math.sqrt(ss / (footprints.size - 1))
      assert(r.stddev.exists(v => Math.abs(v - expected) < 1e-6),
        s"stddev ${r.stddev} != two-pass reference $expected")
      assert(r.maxOverMean.exists(v => Math.abs(v - footprints.max / mean) < 1e-6))
    }
  }

  test("a stage whose metric arrives only at completion publishes no extrema") {
    // Stage 2 has a task, but the accumulable appears only on the StageCompleted event, which is
    // the shape that leaves a record with a total and no samples behind it.
    val events = header ++ Seq(
      stageSubmitted(2, 1),
      taskEnd(2, 0, Seq(taskAccum(11, "gpuTime", 7L, 7L))),
      stageCompleted(2, 1, Seq(stageAccum(20, "gpuSemaphoreWait", 500L))))
    withEventLog(events) { app =>
      val r = row(gpuRows(app), 2, "gpuSemaphoreWait")
        .getOrElse(fail("no gpuSemaphoreWait row"))
      assert(r.count == 0L, "no task attempt reported this metric")
      assert(r.sum.contains(500L), "the stage completion value is still published")
      // 0 is the placeholder the record was created with, not a value any attempt produced.
      assert(r.max.isEmpty, "a metric no attempt reported has no maximum")
      assert(r.min.isEmpty, "a metric no attempt reported has no minimum")
      assert(r.avg.isEmpty && r.stddev.isEmpty && r.cv.isEmpty && r.maxOverMean.isEmpty)
      val cells = r.convertToCSVSeq().toSeq
      assert(cells.length == r.outputHeaders.length)
      assert(cells(5) == "", "the max cell must be blank, not 0")
      assert(cells(8) == "", "the min cell must be blank, not 0")
      // The rollup must carry the blank through rather than substituting the placeholder. The
      // SQL table is empty for this log, which has no job event to populate sqlIdToStages.
      val appRow = appRows(app).find(_.metricName == "gpuSemaphoreWait")
        .getOrElse(fail("no application row"))
      assert(appRow.count == 0L && appRow.sum.contains(500L))
      assert(appRow.max.isEmpty, "a rolled up maximum over no sample stays empty")
      assert(appRow.min.isEmpty, "a rolled up minimum over no sample stays empty")
      val appCells = appRow.convertToCSVSeq().toSeq
      assert(appCells(4) == "" && appCells(7) == "", "blank max and min cells at app level")
    }
  }
}
