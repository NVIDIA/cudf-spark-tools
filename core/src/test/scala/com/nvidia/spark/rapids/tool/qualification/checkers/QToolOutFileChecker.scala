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
import java.nio.file.{Files, Paths, StandardCopyOption}

import com.nvidia.spark.rapids.tool.views.qualification.{QualOutputTableDefinition, QualYamlConfigLoader}
import com.nvidia.spark.rapids.tool.views.qualification.QualReportGenConfProvider.TABLES_CONFIG_PATH
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.sql.rapids.tool.util.RapidsToolsConfUtil


/**
 * Abstract class for checkers that verify the content of a file.
 *
 * In a typical usage, the checker is created in the test code, and the
 * properties of the checker are set by calling the withXXX methods.
 * The checker is then passed to the test framework, which calls the
 * doFire method.
 *
 * The checker is responsible for loading the file and calling the
 * contentVisitors to verify the content of the file.
 */
abstract class QToolOutFileChecker
    extends QToolResultCheckerTrait {
  // the label of the table definition in the yaml file.
  var qTableLabel: String = "qualCoreCSVSummary"
  // defines the expected file path to compare the actual file.
  var expectedLocPath: Option[Seq[String]] = None
  // defines a list of checks to apply on the content of the file.
  var contentVisitors: Seq[QToolCSVVisitorTrait] = Seq.empty
  // if true, there should be a check that no files are generated.
  // This typically needed to  test that certain configs or arguments would disable a specific file.
  var noFileGenerated: Boolean = false

  // the output directory of the qual tool
  private var _qOutDir: String = ""
  def qOutDir: String = _qOutDir
  def qOutDir_=(value: String): Unit = {
    _qOutDir = value
  }

  lazy val generateGoldenSetEnabled: Boolean = {
    // check if the context should regenerate the golden set.
    RapidsToolsConfUtil.toolsBuildProperties
      .getProperty("test.qual.generate.golden.enable", "false")
      .toBoolean
  }

  lazy val goldenSetPath: File = {
    // where the golden set is stored
    val goldenRootDir = RapidsToolsConfUtil.toolsBuildProperties
      .getProperty("test.qual.generate.golden.dir", "./golden-sets/qual")
    new File(goldenRootDir)
  }

  // loads the qual table definition from the configuration.
  lazy val qTableDefn: QualOutputTableDefinition = {
    val tableSelector: QualOutputTableDefinition => Boolean = { defn =>
      defn.label == qTableLabel
    }
    QualYamlConfigLoader.loadConfig(TABLES_CONFIG_PATH)
      .qualTableDefinitions
      .filter(tableSelector).head
  }

  // the actual containers of the output files
  lazy val actualCSVContainers: Map[String, QToolCSVFileContainer] = {
    val tempContainers = QToolCSVFileContainer.loadFromDir(qOutDir, qTableDefn)
    tempContainers.map( c =>
      c.getUUID -> c
    ).toMap
  }

  def withTableLabel(label: String): QToolOutFileChecker = {
    qTableLabel = label
    this
  }

  def withNoGeneratedFile(): QToolOutFileChecker = {
    noFileGenerated = true
    this
  }

  def withExpectedLoc(expectedPath: String): QToolOutFileChecker = {
    expectedLocPath = Some(expectedLocPath.getOrElse(Seq()) :+ expectedPath)
    this
  }

  def withExpectedRows(desc: String, expectedCount: Int): QToolOutFileChecker = {
    contentVisitors :+= QToolCSVFileContentVisitor.withRowCount(desc, expectedCount)
    this
  }

  def withContentVisitor(v: QToolCSVVisitorTrait): QToolOutFileChecker = {
    contentVisitors :+= v
    this
  }

  def withContentVisitor(descr: String, f: QToolCSVFileContainer => Unit): QToolOutFileChecker = {
    val v = QToolCSVFileContentVisitor(descr, f)
    contentVisitors :+= v
    this
  }

  def verifyFilesExist(): Unit = {
    actualCSVContainers should not be empty
  }

  def verifyHeaders(): Unit = {
    actualCSVContainers.values.foreach(c => c.verifyHeaders())
  }

  def verifyContent(): Unit = {
    actualCSVContainers.values.foreach { csvContainer =>
      contentVisitors.foreach { v =>
        csvContainer.accept(v)
      }
    }
  }

  def copyFileUTF8(srcPath: String, destPath: String): Unit = {
    // copy the file from src to dest
    val source = Paths.get(srcPath)
    val destination = Paths.get(destPath)

    Files.copy(source, destination, StandardCopyOption.REPLACE_EXISTING)
  }

  // generate the golden set
  def generateGoldenSet(): Unit = {
    expectedLocPath.foreach { expPaths =>
      expPaths.foreach { onePath =>
        val goldenLeafDir = new File(onePath).getName
        actualCSVContainers.foreach { case (uuid, realContainer) =>
          val subFolderPrefix = s"${realContainer.tableDefn.label}."
          val goldenSubDir = new File(goldenSetPath, goldenLeafDir)
          val destinationFolder = if (uuid.startsWith(subFolderPrefix)) {
            // this is a table should be in subdirectory
            val cleanedUuid = uuid.stripPrefix(subFolderPrefix)
            new File(goldenSubDir, cleanedUuid)
          } else {
            // this is a table should be in the root directory
            goldenSubDir
          }
          destinationFolder.mkdirs()
          val destinationPath = new File(destinationFolder, realContainer.tableDefn.fileName)
          copyFileUTF8(realContainer.fileLoc, destinationPath.getAbsolutePath)
        }
      }
    }
  }

  // compares the actual output to the expected files if provided.
  def verifyExpectedFiles(): Unit = {
    expectedLocPath.foreach { expPaths =>
      val expectedContainers =
        expPaths.flatMap { d =>
          QToolCSVFileContainer.loadFromDir(d, qTableDefn).map(c => c.getUUID -> c)
        }.toMap
      // lets compare the actual and expected containers
      actualCSVContainers.keySet should contain theSameElementsAs expectedContainers.keySet
      // lets compare the content of the files
      actualCSVContainers.foreach { case (cUUID, actualContainer) =>
        expectedContainers.get(cUUID) match {
          case Some(expectedContainer) =>
            actualContainer.compareFileContent(expectedContainer)
          case None =>
            fail(s"Expected CSV with UUID $cUUID not found in expected paths.")
        }
      }
    }
  }
}

/**
 * The concrete implementation of QToolOutFileChecker.
 * @param description a description of the test behavior.
 */
case class QToolOutFileCheckerImpl(description: String) extends QToolOutFileChecker {
  override def doFire(qTestCtxt: QToolTestCtxt): Unit = {
    qOutDir = qTestCtxt.outputDirectory
    super.doFire(qTestCtxt)
    if (noFileGenerated) {
      // check the actual file does not exist.
      actualCSVContainers shouldBe empty
    } else {
      verifyFilesExist()
      verifyHeaders()
      verifyContent()
      if (generateGoldenSetEnabled) {
        // generate the golden set then exit without comparing the expected files
        generateGoldenSet()
      } else {
        // compare the actual output to the expected files
        // only verify the files if they were generated by the same spark runtime.
        verifyExpectedFiles()
      }
    }
  }

  override def build(): QToolResultCheckerTrait = {
    this
  }
}
