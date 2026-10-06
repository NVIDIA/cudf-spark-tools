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

import com.github.tototoshi.csv.CSVReader
import com.nvidia.spark.rapids.tool.ToolTestUtils.findFilesRecursively
import com.nvidia.spark.rapids.tool.views.qualification.QualOutputTableDefinition
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

trait QToolCSVVisitorTrait {
  def visit(csvContainer: QToolCSVFileContainer): Unit
  var accept: QToolCSVFileContainer => Boolean = {
    _ => true
  }
}

trait CSVContentTrait {
  def accept(visitor: QToolCSVVisitorTrait): Unit
}

/**
 * A visitor that loads Qual output file and can dyncamically be
 * initialized with different code checks depending on the test.
 * @param desc explanation of the test behavior
 * @param fn the function to be applied on the container.
 */
case class QToolCSVFileContentVisitor(
    desc: String,
    fn: QToolCSVFileContainer => Unit) extends QToolCSVVisitorTrait {
  override def visit(csvContainer: QToolCSVFileContainer): Unit = {
    if (accept(csvContainer)) {
      fn(csvContainer)
    }
  }
  def withPredicate(p: QToolCSVFileContainer => Boolean): QToolCSVFileContentVisitor = {
    accept = p
    this
  }
}

// USed to create a checker that verifies the number of rows in a specific file.
object QToolCSVFileContentVisitor {
  def withRowCount(desc: String, expectedRowCount: Int): QToolCSVFileContentVisitor = {
    QToolCSVFileContentVisitor(desc, c => {
      c.csvRows.size shouldBe expectedRowCount
    })
  }
}

/**
 * A trait to load Qual output file
 * @tparam T the type of content loaded from the file. For now, we can load CSV file, and later
 *           we can specify different content for JSOn/Text files.
 */
trait QToolOutFileContainer[T] extends CSVContentTrait {
  val fileLoc: String
  val tableDefn: QualOutputTableDefinition
  def loadContent: T
  // Gives a UUID to every loaded file. This helpful in case when we are loading per-app files.
  // In that case, there will be multiple sql reports and we might need to run a checks on
  // specific ones.
  def getUUID: String = {
    tableDefn.scope match {
      case "per-app" =>
        // tableUUID = label_<parent-folder>
        // get parent folder name
        val parentFolder = new File(fileLoc)
        val parentFolderName = parentFolder.getParentFile.getName
        s"${tableDefn.label}.$parentFolderName"
      case "global" =>
        // get the table label as UUID
        tableDefn.label
      case _ =>
        throw new IllegalArgumentException(s"Unknown scope: ${tableDefn.scope}")
    }
  }
}

/**
 * A checker implementation that loads CSV file and verifies its contents.
 * @param fileLoc the path of the CSV file
 * @param tableDefn the table definition created by the yaml configurations.
 */
case class QToolCSVFileContainer private(
    fileLoc: String,
    tableDefn: QualOutputTableDefinition)
    extends QToolOutFileContainer[(List[String], List[Map[String, String]])] {

  lazy val csvRows: List[Map[String, String]] = csvContent._2
  lazy val csvHeaders: List[String] = csvContent._1
  lazy val csvContent: (List[String], List[Map[String, String]]) = loadContent

  def verifyHeaders(): Unit = {
    val expectedHeaders = tableDefn.columns.map(_.name)
    csvHeaders shouldBe expectedHeaders
  }

  // Used to compare between actual and expected files.
  def compareFileContent(expectedContainer: QToolCSVFileContainer): Unit = {
    val expectedRows = expectedContainer.csvRows
    val actualRows = csvRows
    actualRows shouldBe expectedRows
  }

  // return the list of values in the given column
  def getColumn(columnHeader: String): Seq[String] = {
    csvRows.map(_(columnHeader))
  }

  override def accept(visitor: QToolCSVVisitorTrait): Unit = {
    visitor.visit(this)
  }

  // Initialize the CSV content
  override def loadContent: (List[String], List[Map[String, String]]) = {
    val reader = CSVReader.open(fileLoc)
    val content = reader.allWithOrderedHeaders()
    reader.close()
    content
  }
}

object QToolCSVFileContainer {
  def loadFromDir(rootDir: String,
      tableDefn: QualOutputTableDefinition): Seq[QToolCSVFileContainer] = {
    val filesMatched = findFilesRecursively(new File(rootDir), tableDefn.fileName)
    filesMatched.map { csvFile =>
      QToolCSVFileContainer(csvFile.getAbsolutePath, tableDefn)
    }
  }
}
