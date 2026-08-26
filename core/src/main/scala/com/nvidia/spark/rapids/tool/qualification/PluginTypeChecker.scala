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

package com.nvidia.spark.rapids.tool.qualification

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.io.BufferedSource
import scala.util.control.NonFatal

import com.nvidia.spark.rapids.tool.{Platform, PlatformFactory}
import com.nvidia.spark.rapids.tool.planparser.ops.UnsupportedExprOpRef
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.util.UTF8Source

object OpSuppLevel extends Enumeration {
  case class OpSuppLevelVal(label: String, support: Boolean,
      description: String = "", requiresNotes: Boolean = false) extends super.Val {
    def isSupported: Boolean = support
    def getNotesForDisabledOp(notesFromFile: String): String = {
      if (notesFromFile != "None") {
        notesFromFile
      } else {
        description
      }
    }

    def shouldLoadNotesFromFile(notesFromFile: String): Boolean = {
      if (requiresNotes) {
        notesFromFile != "None"
      } else {
        true
      }
    }
  }

  import scala.language.implicitConversions
  implicit def valueToOpSupportLevelsVal(x: Value): OpSuppLevelVal = x.asInstanceOf[OpSuppLevelVal]

  val S = OpSuppLevelVal("S", true, "Enabled by the Plugin")
  val TON = OpSuppLevelVal("TON", true, "Force enabled by the Tools")
  // we consider PS supported for now
  val PS = OpSuppLevelVal("PS", true, "Partially supported by the Plugin")
  val TNEW = OpSuppLevelVal("TNEW", false, "Recently added and not tested yet in Tools")
  val TOFF = OpSuppLevelVal("TOFF", false, "Force disabled by the Tools")
  val NS = OpSuppLevelVal("NS", false, "Not supported by the plugin", requiresNotes = true)
  val CO = OpSuppLevelVal("CO", false, "Configured OFF by the Plugin", requiresNotes = true)
  val NA = OpSuppLevelVal("NA", false, "Not available by the Plugin")
  val Unknown = OpSuppLevelVal("Unknown", false, "Anything that is not defined")

  // Those are the support level that propagate the notes to clarify why a certain op is disabled.
  val suppLevelsWithPropagation: Set[OpSuppLevelVal] = Set(TOFF, NS, CO)

  def fromString(str: String): OpSuppLevelVal = {
    values.find(_.toString.toLowerCase() == str.toLowerCase()).getOrElse(Unknown)
  }

  def isLevelSupported(level: String): Boolean = {
    val opSupLevel = fromString(level)
    opSupLevel.isSupported
  }
}

/**
 * This class is used to check what the cuDF plugin
 * supports for data formats and data types.
 * By default it relies on a csv file included in the jar which is generated
 * by the plugin which lists the formats and types supported.
 * The class also supports a custom speedup factor file as input.
 */
class PluginTypeChecker(platform: Platform = PlatformFactory.createInstance(),
                        speedupFactorFile: Option[String] = None) extends Logging {
  private val NONE = "None"

  private val DEFAULT_DS_FILE = "supportedDataSource.csv"
  private val SUPPORTED_EXECS_FILE = "supportedExecs.csv"
  private val SUPPORTED_EXPRS_FILE = "supportedExprs.csv"

  // map of file format => Map[support category => Seq[Datatypes for that category]]
  // contains the details of formats to which ones have datatypes not supported.
  // Write formats contains only the formats that are supported. Types cannot be determined
  // from event logs for write formats.
  // var for testing purposes
  private var (readFormatsAndTypes, writeFormats) = readSupportedTypesForPlugin

  private var supportedOperatorsScore = readOperatorsScore

  private val supportedExecs = readSupportedExecs.map { x => (x._1, OpSuppLevel.fromString(x._2))}

  private val supportedExprs = readSupportedExprs.map { x => (x._1, OpSuppLevel.fromString(x._2))}

  private val unsupportedOpsReasons = readUnsupportedOpsByDefaultReasons

  // for testing purposes only
  def setPluginDataSourceFile(filePath: String): Unit = {
    val source = UTF8Source.fromFile(filePath)
    val (readFormatsAndTypesTest, writeFormatsTest) = readSupportedTypesForPlugin(source)
    readFormatsAndTypes = readFormatsAndTypesTest
    writeFormats = writeFormatsTest
  }

  def setOperatorScore(filePath: String): Unit = {
    val source = UTF8Source.fromFile(filePath)
    supportedOperatorsScore = readOperators(source, "score", true).map(x => (x._1, x._2.toDouble))
  }

  def getSupportedExprs: Map[String, OpSuppLevel.OpSuppLevelVal] = supportedExprs

  private def readOperatorsScore: Map[String, Double] = {
    speedupFactorFile match {
      case None =>
        logDebug(s"Trying to read operators scores with platform: $platform")
        val file = platform.getOperatorScoreFile
        try {
          val source = UTF8Source.fromResource(file)
          readOperators(source, "score", true).map(x => (x._1, x._2.toDouble))
        } catch {
          case NonFatal(_) =>
            val defaultFile = platform.getDefaultOperatorScoreFile
            logWarning(s"Unable to read operator scores from file: $file. " +
                s"Using default operator scores file: $defaultFile.")
            val source = UTF8Source.fromResource(defaultFile)
            readOperators(source, "score", true).map(x => (x._1, x._2.toDouble))
        }
      case Some(file) =>
        logDebug(s"Reading operators scores from custom speedup factor file: $file")
        try {
          val path = new Path(file)
          val fs = FileSystem.get(path.toUri, new Configuration())
          val source = new BufferedSource(fs.open(path))
          readOperators(source, "score", true).map(x => (x._1, x._2.toDouble))
        } catch {
          case NonFatal(e) =>
            logError(s"Exception processing operators scores with file: $file", e)
          Map.empty[String, Double]
        }
    }
  }

  private def readSupportedExecs: Map[String, String] = {
    val source = UTF8Source.fromResource(SUPPORTED_EXECS_FILE)
    readOperators(source, "execs", true)
  }

  private def readSupportedExprs: Map[String, String] = {
    val source = UTF8Source.fromResource(SUPPORTED_EXPRS_FILE)
    // Some SQL function names have backquotes(`) around their names,
    // so we remove them before saving.
    readOperators(source, "exprs", true).map(
      x => (x._1.toLowerCase.replaceAll("\\`", "").replaceAll(" ", ""), x._2))
  }

  def readUnsupportedOpsByDefaultReasons: Map[String, String] = {
    val execsSource = UTF8Source.fromResource(SUPPORTED_EXECS_FILE)
    val unsupportedExecsBydefault = readOperators(execsSource, "execs", false)
    val exprsSource = UTF8Source.fromResource(SUPPORTED_EXPRS_FILE)
    val unsupportedExprsByDefault = readOperators(exprsSource, "exprs", false).map(
      x => (x._1.toLowerCase.replaceAll("\\`", "").replaceAll(" ", ""), x._2))
    unsupportedExecsBydefault ++ unsupportedExprsByDefault
  }

  private def readSupportedTypesForPlugin: (
      Map[String, Map[String, Seq[String]]], ArrayBuffer[String]) = {
    val source = UTF8Source.fromResource(DEFAULT_DS_FILE)
    readSupportedTypesForPlugin(source)
  }

  /**
   * Reads operators from a source and processes them based on the provided logic.
   *
   * @param source The source from which to read the operators.
   * @param operatorType The type of operators being read ("execs", "exprs" or "score").
   * @param isSupported Flag to determine if we are reading supported or unsupported operators.
   * @param processLine A function that takes a line, it's operator type and isSupported flag,
   *                    and returns a sequence of key-value pairs to add to the result map.
   * @return A Map containing the processed operators.
   */
  def readOperators(source: BufferedSource,
      operatorType: String, isSupported: Boolean): Map[String, String] = {
    val operatorsMap = HashMap.empty[String, String]
    try {
      val fileContents = source.getLines().toSeq
      if (fileContents.size < 2) {
        throw new IllegalStateException(s"${source.toString} file appears corrupt, " +
            s"must have at least the header and one line")
      }
      val header = fileContents.head.split(",").map(_.trim.toLowerCase)
      fileContents.tail.foreach { line =>
        val cols = line.split(",").map(_.trim)
        if (header.size != cols.size) {
          throw new IllegalStateException(s"${source.toString} file appears corrupt, " +
              s"header length doesn't match rows length. " +
              s"Row that doesn't match is ${cols.mkString(",")}")
        }
        processOperatorLine(cols, operatorType, isSupported).foreach { case (key, value) =>
          operatorsMap.put(key, value)
        }
      }
    } finally {
      source.close()
    }
    operatorsMap.toMap
  }

  // Custom logic for processing lines for supported or unsupported operators
  // Here unsupported operators mean the ones that are not supported by default.
  // In the notes section of the supported csv files, it specifies reason for why it is not
  // supported by default. We use this information to propagate it unsupported operators
  // csv file.
  private def processOperatorLine(cols: Array[String], operatorType: String,
      isSupported: Boolean): Seq[(String, String)] = {
    operatorType match {
      case "exprs" if isSupported =>
        // Logic for supported expressions
        val exprName = Seq((cols(0), cols(1)))
        val sqlFuncNames = if (cols(2).nonEmpty && cols(2) != NONE) {
          // There are addidtional checks for Expressions. In physical plan, SQL function name is
          // printed instead of expression name. We have to save both expression name and
          // SQL function name(if there is one) so that we don't miss the expression while
          // parsing the execs.
          // Ex: Expression name = Substring, SQL function= `substr`; `substring`
          // Ex: Expression name = Average, SQL function name = `avg`
          cols(2).split(";").map(_.trim).toSeq
        } else {
          Seq.empty
        }
        exprName ++ sqlFuncNames.map(name => (name, cols(1)))
      case "exprs" =>
        // Logic for unsupported expressions
        val opSupVal = OpSuppLevel.fromString(cols(1))
        if (OpSuppLevel.suppLevelsWithPropagation.contains(opSupVal) &&
            opSupVal.shouldLoadNotesFromFile(cols(3))) {
          val loadedReason = opSupVal.getNotesForDisabledOp(cols(3))
          val exprName = Seq((cols(0), loadedReason))
          val sqlFuncNames = if (cols(2).nonEmpty && cols(2) != NONE) {
            cols(2).split(";").map(_.trim).toSeq
          } else {
            Seq.empty
          }
          exprName ++ sqlFuncNames.map(name => (name, loadedReason))
        } else {
          Seq.empty
        }
      case _ if isSupported =>
        // Logic for supported execs
        Seq((cols(0), cols(1)))
      case _ =>
        // Logic for unsupported execs
        val opSupVal = OpSuppLevel.fromString(cols(1))
        if (OpSuppLevel.suppLevelsWithPropagation.contains(opSupVal) &&
            opSupVal.shouldLoadNotesFromFile(cols(2))) {
          val loadedReason = opSupVal.getNotesForDisabledOp(cols(2))
          Seq((cols(0), loadedReason))
        } else {
          Seq.empty
        }
    }
  }

  // file format should be like this:
  // Format,Direction,BOOLEAN,BYTE,SHORT,INT,LONG,FLOAT,DOUBLE,DATE,...
  // CSV,read,S,S,S,S,S,S,S,S,S*,S,NS,NA,NS,NA,NA,NA,NA,NA
  // Parquet,write,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA
  private def readSupportedTypesForPlugin(
      source: BufferedSource): (Map[String, Map[String, Seq[String]]], ArrayBuffer[String]) = {
    // get the types the Rapids Plugin supports
    val allSupportedReadSources = HashMap.empty[String, Map[String, Seq[String]]]
    val allSupportedWriteFormats = ArrayBuffer[String]()
    try {
      val fileContents = source.getLines().toSeq
      if (fileContents.size < 2) {
        throw new IllegalStateException("supportedDataSource file appears corrupt," +
          " must have at least the header and one line")
      }
      // first line is header
      val header = fileContents.head.split(",").map(_.toLowerCase)
      // the rest of the rows are file formats with type supported info
      fileContents.tail.foreach { line =>
        val cols = line.split(",")
        if (header.size != cols.size) {
          throw new IllegalStateException("supportedDataSource file appears corrupt," +
            " header length doesn't match rows length")
        }
        val format = cols(0).toLowerCase
        val direction = cols(1).toLowerCase()
        if (direction.equals("read")) {
          val dataTypesToSup = header.drop(2).zip(cols.drop(2)).toMap
          val nsTypes = dataTypesToSup.filter { case (_, sup) =>
            !OpSuppLevel.isLevelSupported(sup)
          }.keys.toSeq.map(_.toLowerCase)
          val allNsTypes = nsTypes.flatMap(t => getOtherTypes(t) :+ t)
          val allBySup = HashMap(OpSuppLevel.NS.label -> allNsTypes)
          allSupportedReadSources.put(format, allBySup.toMap)
        } else if (direction.equals("write")) {
          allSupportedWriteFormats += format
        }
      }
    } finally {
      source.close()
    }
    (allSupportedReadSources.toMap, allSupportedWriteFormats)
  }

  private def getOtherTypes(typeRead: String): Seq[String] = {
    typeRead match {
      case "binary" => Seq("blob")
      case "byte" => Seq("tinyint")
      case "calendar" => Seq("interval")
      case "decimal" => Seq("dec", "numeric")
      case "float" => Seq("real")
      case "int" => Seq("integer")
      case "long" => Seq("bigint")
      case "short" => Seq("smallint")
      case "string" => Seq("varchar", "char")
      case _ => Seq.empty[String]
    }
  }

  // Parsing the schema string is very complex when you get into nested types, so for now
  // we do the simpler thing of checking to see if the schema string contains types we
  // don't support.
  // NOTE, UDT doesn't show up in the event log, when its written, it gets written as
  // other types since parquet/orc has to know about it
  def scoreReadDataTypes(format: String, schema: String): (Double, Set[String]) = {
    val schemaLower = schema.toLowerCase
    val formatInLower = format.toLowerCase
    val typesBySup = readFormatsAndTypes.get(formatInLower)
    val score = typesBySup match {
      case Some(dtSupMap) =>
        // check if any of the not supported types are in the schema
        val nsFiltered =
          dtSupMap(OpSuppLevel.NS.label).filter(t => schemaLower.contains(t.toLowerCase()))
        if (nsFiltered.nonEmpty) {
          (0.0, nsFiltered.toSet)
        } else {
          // Started out giving different weights based on partial support and so forth
          // but decided to be optimistic and not penalize if we don't know, perhaps
          // make it smarter later.
          // Schema could also be incomplete, but similarly don't penalize since we don't
          // know.
          (1.0, Set.empty[String])
        }
      case None =>
        // assume we don't support that format
        (0.0, Set("*"))
    }
    score
  }

  def isWriteFormatSupported(writeFormat: String): Boolean = {
    val format = writeFormat.toLowerCase.trim
    writeFormats.map(x => x.trim).contains(format)
  }

  def getUnsupportedWriteFormat(writeFormat: Iterable[String]): Iterable[String] = {
    writeFormat.map(x => x.toLowerCase.trim).filterNot(
      writeFormats.map(x => x.trim).contains(_))
  }

  def getSpeedupFactor(execOrExpr: String): Double = {
    supportedOperatorsScore.get(execOrExpr).getOrElse(1.0)
  }

  def isExecSupported(exec: String): Boolean = {
    val opSupLevel = supportedExecs.getOrElse(exec, OpSuppLevel.NS)
    opSupLevel.isSupported
  }

  def isExprSupported(expr: String): Boolean = {
    val exprLowercase = expr.toLowerCase
    val exprSupported = supportedExprs.getOrElse(exprLowercase, OpSuppLevel.NS)
    exprSupported.isSupported
  }

  def getNotSupportedExprs(exprs: Seq[String]): Seq[UnsupportedExprOpRef] = {
    exprs.distinct.collect {
      case expr if !isExprSupported(expr) =>
        val reason = unsupportedOpsReasons.getOrElse(expr, "")
        UnsupportedExprOpRef(expr, reason)
    }
  }

  def getNotSupportedExecsReason(exec: String): String = {
    unsupportedOpsReasons.getOrElse(exec, "")
  }
}
