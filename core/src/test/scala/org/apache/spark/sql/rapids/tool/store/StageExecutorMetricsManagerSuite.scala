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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.executor.ExecutorMetrics
import org.apache.spark.scheduler.{SparkListenerEvent, SparkListenerStageExecutorMetrics}
import org.apache.spark.sql.rapids.tool.{AppBase, EventProcessorBase}

class StageExecutorMetricsManagerSuite extends AnyFunSuite {

  private class TestApp extends AppBase(None, None) {
    override def processEvent(event: SparkListenerEvent): Boolean = false
  }

  test("retain maximum positive Python virtual memory by exact stage attempt and executor") {
    val manager = new StageExecutorMetricsManager()

    manager.addPythonVMemory(1, 0, "1", 100L)
    manager.addPythonVMemory(1, 0, "1", 90L)
    manager.addPythonVMemory(1, 0, "1", 120L)
    manager.addPythonVMemory(1, 0, "2", 80L)
    manager.addPythonVMemory(1, 1, "1", 70L)
    manager.addPythonVMemory(2, 0, "1", 60L)

    assert(manager.getPythonVMemory(1, 0) === Map("1" -> 120L, "2" -> 80L))
    assert(manager.getPythonVMemory(1, 1) === Map("1" -> 70L))
    assert(manager.getPythonVMemory(2, 0) === Map("1" -> 60L))
  }

  test("exclude driver and non-positive Python virtual memory") {
    val manager = new StageExecutorMetricsManager()

    manager.addPythonVMemory(1, 0, "driver", 200L)
    manager.addPythonVMemory(1, 0, "1", 0L)
    manager.addPythonVMemory(1, 0, "2", -1L)

    assert(manager.getPythonVMemory(1, 0).isEmpty)
  }

  test("remove all attempts for cleaned stage IDs") {
    val manager = new StageExecutorMetricsManager()
    manager.addPythonVMemory(1, 0, "1", 100L)
    manager.addPythonVMemory(1, 1, "1", 110L)
    manager.addPythonVMemory(2, 0, "1", 120L)

    manager.removeStages(Set(1))

    assert(manager.getPythonVMemory(1, 0).isEmpty)
    assert(manager.getPythonVMemory(1, 1).isEmpty)
    assert(manager.getPythonVMemory(2, 0) === Map("1" -> 120L))
  }

  test("common event processor ingests stage executor metrics for all AppBase tools") {
    val app = new TestApp()
    val processor = new EventProcessorBase[TestApp](app) {}

    def metrics(pythonVMemory: Long): ExecutorMetrics = {
      new ExecutorMetrics(Map("ProcessTreePythonVMemory" -> pythonVMemory))
    }

    processor.processAnyEvent(SparkListenerStageExecutorMetrics("1", 3, 0, metrics(100L)))
    processor.processAnyEvent(SparkListenerStageExecutorMetrics("1", 3, 0, metrics(120L)))
    processor.processAnyEvent(SparkListenerStageExecutorMetrics("2", 3, 1, metrics(80L)))
    processor.processAnyEvent(SparkListenerStageExecutorMetrics("driver", 3, 0, metrics(200L)))

    assert(app.stageExecutorMetricsManager.getPythonVMemory(3, 0) === Map("1" -> 120L))
    assert(app.stageExecutorMetricsManager.getPythonVMemory(3, 1) === Map("2" -> 80L))

    app.cleanupStages(Set(3))
    assert(app.stageExecutorMetricsManager.getPythonVMemory(3, 0).isEmpty)
    assert(app.stageExecutorMetricsManager.getPythonVMemory(3, 1).isEmpty)
  }
}
