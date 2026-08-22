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

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import com.nvidia.spark.rapids.BaseNoSparkSuite
import com.nvidia.spark.rapids.tool.EventLogPathProcessor
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.scheduler.SparkListenerEvent
import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.execution.{SparkPlanInfo => UpstreamSparkPlanInfo}
import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.util.RapidsToolsConfUtil

class SQLPlanModelManagerSuite extends BaseNoSparkSuite {

  private class TestApp extends AppBase(None, None) {
    override def processEvent(event: SparkListenerEvent): Boolean = false
  }

  private def plan(name: String, children: UpstreamSparkPlanInfo*): UpstreamSparkPlanInfo = {
    new UpstreamSparkPlanInfo(name, name, children, Map.empty, Seq.empty)
  }

  private def withProfilingApp(f: ApplicationInfo => Unit): Unit = {
    val events = Seq(
      """{"Event":"SparkListenerLogStart","Spark Version":"3.5.0"}""",
      """{"Event":"SparkListenerApplicationStart","App Name":"PlanTest",""" +
        """"App ID":"local-plan-test","Timestamp":100000,"User":"test"}""",
      """{"Event":"SparkListenerApplicationEnd","Timestamp":200000}""")
    TrampolineUtil.withTempDir { tempDir =>
      val eventLog = Paths.get(tempDir.getAbsolutePath, "eventlog")
      Files.write(eventLog, events.mkString("\n").getBytes(StandardCharsets.UTF_8))
      val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
      val eventLogInfo = EventLogPathProcessor
        .getEventLogInfo(eventLog.toString, hadoopConf).head._1
      f(new ApplicationInfo(hadoopConf, eventLogInfo))
    }
  }

  test("SQL cache evidence includes every plan version and survives cleanup") {
    val app = new TestApp

    assert(!app.hasSqlCacheEvidence)

    app.sqlManager.addNewExecution(1L, plan("Project"), "primary")
    app.sqlManager.addAQE(1L,
      plan("AdaptiveSparkPlan", plan("Filter", plan("InMemoryTableScan"))), "aqe-cache")
    app.sqlManager.addAQE(1L, plan("Project"), "aqe-final")
    app.sqlManager.addNewExecution(2L,
      plan("Project", plan("InMemoryRelation"), plan("InMemoryRelation")), "second-sql")

    app.hasSqlCacheEvidence shouldBe true

    app.cleanupSQL(1L)
    app.cleanupSQL(2L)

    app.hasSqlCacheEvidence shouldBe true
  }

  test("SQL cache evidence uses normalized Spark names for platform-aware plans") {
    withProfilingApp { app =>
      app.gpuMode = true

      app.sqlManager.addNewExecution(1L, plan("GpuInMemoryTableScan"), "gpu-plan")

      app.sqlManager.getPlanInfoById(1L).get.platformName shouldBe "GpuInMemoryTableScan"
      app.hasSqlCacheEvidence shouldBe true
    }
  }

  Seq(
    "InMemoryRelation",
    "InMemoryTableScan",
    "TableCacheQueryStage",
    "GpuInMemoryTableScan",
    "Scan In-memory table hot_orders").foreach { cacheNodeName =>
    test(s"SQL cache evidence recognizes '$cacheNodeName'") {
      val app = new TestApp

      app.sqlManager.addNewExecution(1L,
        plan("Project", plan(cacheNodeName)), "cache-plan")

      app.hasSqlCacheEvidence shouldBe true
    }
  }

  test("non-cache scan names do not produce SQL cache evidence") {
    val app = new TestApp

    app.sqlManager.addNewExecution(1L,
      plan("Project", plan("Scan parquet orders")), "non-cache-plan")

    app.hasSqlCacheEvidence shouldBe false
  }
}
