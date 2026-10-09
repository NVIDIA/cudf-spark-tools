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


package com.nvidia.spark.rapids.tool.qualification.checkers

import java.io.File

import com.nvidia.spark.rapids.tool.{EventlogProviderTrait, PlatformNames, StaticEventlogProvider}

import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.rapids.tool.util.RapidsToolsConfUtil


/**
 * A builder that helps to create a QToolTestCtxt.
 * If the outputDirectory is not provided by the caller, then the builder will create a
 * temporary workDir. If needed, the workDir can be set to outlive the unit-tests by setting the
 * build properties `test.tools.cleanup.tmp.dir`.
 *
 * @param eventlogs the eventlogs to test against
 * @param toolArgs the arguments to pass to the tool
 * @param checkBuilders the builders for the different checks
 */
case class QToolTestCtxtBuilder(
    var eventlogs: Array[String] = Array(),
    var toolArgs: Array[String] = Array(),
    var checkBuilders: Seq[QToolResultCheckerTrait] = Seq.empty) {

  private var _platform: String = PlatformNames.DEFAULT
  private var _selfManagedWorkDir: Option[File] = None

  def platform: String = _platform
  def platform_=(value: String): Unit = {
    _platform = value
  }

  private var _workDir: Option[File] = None
  def workDir: Option[File] = _workDir
  def workDir_=(value: Option[File]): Unit = {
    _workDir = value
  }
  private var reloadEvProvider: Boolean = false

  private var _evLogProvider: EventlogProviderTrait = StaticEventlogProvider(eventlogs)
  def evLogProvider: EventlogProviderTrait = _evLogProvider
  def evLogProvider_=(value: EventlogProviderTrait): Unit = {
    _evLogProvider = value
  }

  def withEvLogProvider(
      eventlogProvider: EventlogProviderTrait): QToolTestCtxtBuilder = {
    evLogProvider = eventlogProvider
    reloadEvProvider = true
    this
  }

  def withToolArgs(tArgs: Array[String], descr: String = ""): QToolTestCtxtBuilder = {
    toolArgs ++= tArgs
    this
  }

  def withMultipleThreads(numThreads: Int): QToolTestCtxtBuilder = {
    toolArgs ++= Array("--num-threads", s"$numThreads")
    this
  }

  def withWorkDirectory(outPath: File): QToolTestCtxtBuilder = {
    workDir = Option(outPath)
    this
  }

  def withWorkDirectory(outPath: String): QToolTestCtxtBuilder = {
    workDir = Option(new File(outPath))
    this
  }

  def withPlatform(plat: String): QToolTestCtxtBuilder = {
    _platform = plat
    this
  }

  def withPerSQL() : QToolTestCtxtBuilder = {
    if (!toolArgs.contains("--per-sql")) {
      toolArgs ++= Array("--per-sql")
    }
    this
  }

  def withChecker(b: QToolResultCheckerTrait): QToolTestCtxtBuilder = {
    checkBuilders = checkBuilders :+ b
    this
  }

  def buildArgs(): Array[String] = {
    toolArgs ++ Array("--platform", platform)
  }

  def setUpWorkDir(): Unit = {
    // we need to create the tempDirectory to run the tests. if it is not set.
    if (workDir.isEmpty) {
      // set the cleanup flag depending on the build properties
      val cleanupFlag =
        RapidsToolsConfUtil.toolsBuildProperties
          .getProperty("test.tools.cleanup.tmp.dir", "false")
          .toBoolean
      workDir = Some(TrampolineUtil.createTempDir())
      if (cleanupFlag) {
        _selfManagedWorkDir = workDir
      }
    }
  }

  def build(): Unit = {
    // set the work directory to a temp dir if not set
    setUpWorkDir()
    try {
      // set the output directory as root/output
      val outputDirectory = new File(workDir.get, "output")
      if (!outputDirectory.exists()) {
        outputDirectory.mkdirs()
      }
      val absPath = outputDirectory.getAbsolutePath
      // check the evntlogProvider
      if (reloadEvProvider) {
        evLogProvider = evLogProvider.build(workDir.get)
      }
      // create the test context passing the output dire and the eventlogs
      val testCtxt = QToolTestCtxt(evLogProvider.eventlogs, absPath, buildArgs())
      val defaultCheckers = Seq(QToolExecsInvariantChecker())
      testCtxt.fire((defaultCheckers ++ checkBuilders).map(_.build()))
    } finally {
      // cleanup workDirectories if needed
      _selfManagedWorkDir.foreach { _ =>
        TrampolineUtil.deleteRecursively(_)
      }
    }
  }
}
