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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.-=
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.tool.planparser

import scala.util.Try
import scala.util.matching.Regex

import com.nvidia.spark.rapids.tool.planparser.hive.HiveParseHelper

import org.apache.spark.sql.rapids.tool.store.{WriteOperationMetaBuilder, WriteOperationMetadataTrait}
import org.apache.spark.sql.rapids.tool.util.StringUtils

/**
 * Trait to extract metadata from write operation log entries.
 *
 * Provides helper functions to break down node description strings into components,
 * extract execution name, data format, location, catalog information, output columns,
 * write mode, and partition details.
 *
 * For example on how the insertCmds are formated in the eventlog, see WriteOperationParserSuite.
 */
trait WriteOpMetaExtractorTrait {
  /**
   * Given a string, this helper functions will break down the string into components.
   * The string is formed of a sequence of substrings separated by ",".
   * A single component can contain arrays surrounded by "[]/()".
   * The helper iterates character‐by‐character, tracking bracket depth so that commas
   * inside [ and ] or ( and ) are not used as delimiters. Each extracted component is trimmed.
   * @param input a sequence of strings separated by comma.
   * @return a sequence of trimmed components.
   */
  def parseComponents(input: String): Seq[String] = {
    val result = scala.collection.mutable.ListBuffer[String]()
    val current = new StringBuilder
    var bracketDepth = 0
    var parenDepth = 0
    var i = 0
    while (i < input.length) {
      val ch = input.charAt(i)
      if (ch == ',' && bracketDepth == 0 && parenDepth == 0) {
        // Found a delimiter outside brackets and parentheses.
        result += current.toString.trim
        current.clear()
        i += 1  // Skip the delimiter.
      } else {
        if (ch == '[') {
          bracketDepth += 1
        } else if (ch == ']') {
          bracketDepth = math.max(bracketDepth - 1, 0)
        } else if (ch == '(') {
          parenDepth += 1
        } else if (ch == ')') {
          parenDepth = math.max(parenDepth - 1, 0)
        }
        current.append(ch)
        i += 1
      }
    }
    if (current.nonEmpty) {
      result += current.toString.trim
    }
    result.toSeq
  }

  /**
   * Extracts the name of the execution from the components. It removes the "Execute" prefix.
   * @param components a sequence of strings.
   * @return the name of the execution if found, or UNKNOWN_EXTRACT otherwise.
   */
  def extractExecName(components: Seq[String]): String = {
    // The exec name is usually in the first component
    // This method extracts the first word after "Execute" if any.
    components.headOption.flatMap { p =>
      p.split("\\s+") match {
        case Array(pref, cmd, _*) if pref.startsWith("Execute") => Some(cmd)
        case Array(cmd, _*) => Some(cmd) // No Execute prefix
        case _ => None
      }
    }.getOrElse(StringUtils.UNKNOWN_EXTRACT)
  }
  // Where does the format exist in the components?
  var formatIndex: Int
  // where the output columns exist in the components?
  var outColumnsIndex: Int = -1
  def extractFormat(components: Seq[String]): String
  def extractLocation(components: Seq[String]): Option[String] = None
  def extractCatalog(components: Seq[String]): (String, String)
  def extractOutputColumns(components: Seq[String]): Option[String]
  def extractWriteMode(components: Seq[String]): String
  def extractPartitions(components: Seq[String]): Option[String] = None
  def extractOptions(components: Seq[String]): Map[String, String] = Map.empty
}

// This abstract class extracts metadata for Hadoop write operations from log nodes.
abstract class InsertIntoHadoopExtract(val nodeDescr: String) extends WriteOpMetaExtractorTrait {
  // Captures the different components of the node description. This assumes that the node
  // description is a sequence of strings separated by commas.
  private val components: Seq[String] = parseComponents(nodeDescr)
  // The index of the component that might contain the database and table name.
  private val catalogIndex = components.indexWhere(c => catalogPredicate(c))
  // The string component that might contain the database and table name.
  val catalogStr: String = getCatalogComponent
  // The name of the exec to be stored in the writeOp.
  val execName: String = extractExecName(components)

  /**
   * Returns the string that might contain information about the DataBase and tableName if any.
   * @return a string if it is found, or UNKNOWN_EXTRACT otherwise.
   */
  private def getCatalogComponent: String = {
    if (hasCatalog) {
      return components(catalogIndex)
    }
    StringUtils.UNKNOWN_EXTRACT
  }

  // Returns true if the execName starts with "Gpu"
  private def isGPU: Boolean = {
    execName.startsWith("Gpu")
  }

  // Returns true if there is a component that might contain the database/table.
  protected def hasCatalog: Boolean = {
    catalogIndex >= 0
  }

  // Returns true if the component string might contain the database/table.
  protected def catalogPredicate(compStr: String): Boolean = false

  // Extracts the file format from a class object string, such as
  // "com.nvidia.spark.rapids.GpuParquetFileFormat@9f5022c".
  //
  // This function is designed to handle cases where the cuDF plugin logs raw object names
  // instead of a user-friendly file format name. For example, it extracts "Parquet" from
  // "com.nvidia.spark.rapids.GpuParquetFileFormat@9f5022c".
  // Refer: https://github.com/NVIDIA/cudf-spark-tools/issues/1561
  //
  // If the input string does not match the expected pattern, the function returns the original
  // string as a fallback.
  //
  // @param formatStr The raw format string, typically containing the class name of the file
  //                  format.
  // @return A user-friendly file format name (e.g., "Parquet") or the original string if no
  //         match is found.
  protected def processGPUFormat(extractValue: String): String = {
    if (isGPU) {
      // Extracting file format from the full object string
      // 1. `.*\.` - Matches sequence of character between literal dots
      // 2. `([a-zA-Z]+)FileFormat` - Captures fileFormat from the class name
      // 3. `(@.*)` - Group capturing @ followed by any character
      val formatRegex = """.*\.Gpu([a-zA-Z]+)FileFormat(@.*)?""".r
      extractValue match {
        case formatRegex(fileFormat, _) => fileFormat
        case _ => extractValue // Return original if no match
      }
    } else { // Return the original string if not a GPU format
      extractValue
    }
  }

  override def extractOutputColumns(components: Seq[String]): Option[String] = {
    // Output Columns are defined as the last component surrounded by brackets.
    // It many cases, it is the very last element; but in some GPU eventlogs it may come as 3rd to
    // last.
    // This helper iterates on all the components backward and returns the first elements that
    // satisfies the condition.
    if (components.isEmpty) {
      return None
    }
    for (i <- components.indices.reverse) {
      if (components(i).startsWith("[") && components(i).endsWith("]")) {
        // this is the output columns
        outColumnsIndex = i
        val cols = components(i).substring(1, components(i).length - 1)
        return Some(cols.replaceAll(",\\s*", ";"))
      }
    }
    // if we reach here, we have not found the output columns
    None
  }

  override def extractWriteMode(components: Seq[String]): String = {
    // Extract the writeMode which can be any argument starting from 4th argument.
    // We need to define a map in case we find some different modes and we need to map them to
    // the standard definitions.
    val knownModes = Map(
      "Append" -> "Append",
      "Overwrite" -> "Overwrite",
      "ErrorIfExists" -> "ErrorIfExists",
      "Ignore" -> "Ignore"
    )

    if (components.isEmpty || components.length < 4) {
      // No need to check if the components are empty or the arguments are less than expected.
      return StringUtils.UNKNOWN_EXTRACT
    }
    for (i <- 3 until components.length) {
      // Starting from component in index(3) to the end, try to find the write mode.
      val mode = components(i)
      if (knownModes.contains(mode)) {
        return knownModes(mode)
      }
    }
    StringUtils.UNKNOWN_EXTRACT
  }

  override def extractOptions(components: Seq[String]): Map[String, String] = {
    // Options are available following the format argument.
    // If the format is the 3rd argument, then options are in the 4th
    components.lift(formatIndex + 1).flatMap { optsStr =>
      if (!optsStr.startsWith("[")) {
        // Not an options string.
        None
      } else {
        // remove the brackets
        val optsContent = optsStr.substring(1, optsStr.length - 1).trim
        // The options are in the form of Map(key = value, key2 = value2)
        // Split the options content by comma and trim each entry
        Some(
          optsContent.split(",").flatMap { entry =>
            entry.split("=", 2) match {
              case Array(key, value) => Some(key.trim -> value.trim)
              case _ => None
            }
          }.toMap
        )
      }
    }.getOrElse(Map.empty[String, String])
  }

  /**
   * Builds the WriteOperationMetadataTrait object from the extracted components.
   * @return a WriteOperationMetadataTrait object.
   */
  def build(): WriteOperationMetadataTrait = {
    val (databaseName, tableName) = extractCatalog(components)
    WriteOperationMetaBuilder.build(
      execName = execName,
      dataFormat = extractFormat(components),
      outputPath = extractLocation(components),
      outputColumns = extractOutputColumns(components),
      writeMode = extractWriteMode(components),
      tableName = tableName,
      dataBaseName = databaseName,
      partitionCols = extractPartitions(components),
      opOptions = Some(extractOptions(components)),
      fullDescr = Some(nodeDescr)
    )
  }
}

/**
 * This class extracts metadata from InsertIntoHadoop command when the node description contains
 * a CatalogTable.
 * The CatalogTable is container that contains "key: value" pairs separated by new lines.
 * For exact details on the format of the catalogTable, see the unit-test WriteOperationParserSuite.
 * The catalogTable requires specific implementation to extract the values.
 * @param nodeDescr the node description string.
 */
case class InsertIntoHadoopExtractWithCatalog(
    override val nodeDescr: String) extends InsertIntoHadoopExtract(nodeDescr) {

  override var formatIndex: Int = 0

  // Determines if a component string contains catalog information.
  // It returns true if the string starts with "CatalogTable(".
  override def catalogPredicate(compStr: String): Boolean = {
    compStr.startsWith("CatalogTable(")
  }

  // Retrieves a property value from the catalog information.
  // The regex uses multiline mode to match the property key at the beginning of a line.
  // If found, it returns the trimmed value from that match.
  private def getPropertyFromCatalog(key: String): Option[String] = {
    // The (?m) flag enables multiline mode so that ^ matches the beginning of a line.
    val regex = s"""(?m)^$key:\\s*(.*)$$""".r
    regex.findFirstMatchIn(catalogStr).map(_.group(1).trim)
  }

  // Extracts the database and table information from the catalog.
  // It retrieves the values for "Database" and "Table".
  // If either property is missing, it returns a default unknown value.
  override def extractCatalog(components: Seq[String]): (String, String) = {
    val database = getPropertyFromCatalog("Database")
    val table = getPropertyFromCatalog("Table")
    (database.getOrElse(StringUtils.UNKNOWN_EXTRACT), table.getOrElse(StringUtils.UNKNOWN_EXTRACT))
  }

  // Extracts the file format from the SerDe information in the catalog.
  // It retrieves the value for "Serde Library", then optionally maps it to a user-friendly format.
  // If no valid format is found, it returns a default unknown value.
  override def extractFormat(components: Seq[String]): String = {
    val serdeDescr = getPropertyFromCatalog("Serde Library")
    // TODO: check if the provider is Hive. Otherwise, the format might be different.
    //       If native Spark is used, then we should map HiveParquet to Parquet.
    serdeDescr.flatMap(
      x => HiveParseHelper.getOptionalHiveFormat(x).map(processGPUFormat)) match {
      case Some(f) =>
        // update the format index to point to the correct index
        val actualFormat =
          if (f == "HiveParquet" || f == "HiveORC") {
            f.replaceFirst("Hive", "")
          } else {
            f
          }
        for (i <- components.indices) {
          if (components(i) == actualFormat) {
            formatIndex = i
          }
        }
        f
      case _ => StringUtils.UNKNOWN_EXTRACT
    }
  }

  // Extracts the location information from the catalog information.
  // It retrieves the value for "Location", if available.
  override def extractLocation(components: Seq[String]): Option[String] = {
    getPropertyFromCatalog("Location")
  }

  // Extracts and formats the partition columns from the catalog information.
  // Retrieves the value for "Partition Columns", removes any backticks,
  // strips square brackets from the beginning and end, and replaces commas with semicolons.
  override def extractPartitions(components: Seq[String]): Option[String] = {
    getPropertyFromCatalog("Partition Columns")
      .map(_.replaceAll("`", "").stripPrefix("[").stripSuffix("]").replaceAll(",\\s*", ";"))
  }
}

/**
 * Class for extracting metadata from Hadoop write operations when no CatalogTable information is
 * provided.
 *
 * This class extends InsertIntoHadoopExtract and implements methods to extract the file format,
 * location, catalog details (database and table names), and partition columns from log entries
 * that do not include a dedicated catalog section. It uses a regular expression to extract strings
 * enclosed by backticks in the node description to identify database and table names.
 * When the required metadata is missing, default
 * values are returned.
 *
 * @param nodeDescr the node description string from which metadata is extracted.
 */
class InsertIntoHadoopExtractNoCatalog(
    override val nodeDescr: String) extends InsertIntoHadoopExtract(nodeDescr) {

  import InsertIntoHadoopExtract._

  // The format exists as the 3rd argument in most cases.
  // However, if the 3rd argument is an array (partitions), then the format is in the 4th argument.
  // This index is used as a reference in extract methods.
  // Note that this index is not always valid, so it should be used with caution.
  private var _formatIndex = 2

  override def formatIndex: Int = _formatIndex

  override def formatIndex_=(value: Int): Unit = {
    _formatIndex = value
  }

  // Determines if a string component contains catalog information based on backticks.
  // It finds all strings between backticks and returns true if there are two or more,
  // which indicates both database and table are present.
  override def catalogPredicate(compStr: String): Boolean = {
    val matches = BACKTICKS_CATALOG_REG_EX.findAllMatchIn(compStr).map(_.group(1)).toList
    matches.length >= 2
  }

  // Extracts the file format from the components.
  // It checks the third argument; if it starts with "[" then it looks at the fourth argument.
  // The selected raw format string is processed using processGPUFormat.
  override def extractFormat(components: Seq[String]): String = {
    // Attempt to get the third argument; if not available, default to an empty string.
    val thirdArg = components.lift(formatIndex).getOrElse("")
    // Determine which argument likely contains the format.
    val rawFormat = if (thirdArg.startsWith("[")) {
      // If the third argument is an array, the format might be in the following argument.
      formatIndex += 1
      components.lift(formatIndex).getOrElse("")
    } else {
      // Otherwise, use the third argument.
      thirdArg
    }
    // Process the raw format string to avoid a bug in the cuDF plugin.
    processGPUFormat(rawFormat)
  }

  // Extracts the location from the components.
  // It splits the first component (which starts with "Execute") by whitespace,
  // and returns the third word as the location if the pattern matches.
  override def extractLocation(components: Seq[String]): Option[String] = {
    components.headOption.flatMap { p =>
      p.split("\\s+") match {
        case Array(pref, _, loc, _*) if pref.startsWith("Execute") => Some(loc)
        case _ => None
      }
    }
  }

  // Extracts catalog information (database and table) from the components.
  // It uses a regular expression to extract strings within backticks from the catalogStr.
  // If at least two matches are found, the last two items are returned as (database, table).
  // Otherwise, returns inapplicable extracts.
  override def extractCatalog(components: Seq[String]): (String, String) = {
    // Note that in some cases the database.table are prefixed by catalog.
    // iterate on the components to find the database and table
    if (hasCatalog) {
      val matches = BACKTICKS_CATALOG_REG_EX.findAllMatchIn(catalogStr).map(_.group(1)).toList
      if (matches.length >= 2) {
        // Last two occurrences are database and table
        return (matches(matches.length - 2), matches.last)
      }
    }
    // Fall back if no catalog is available. It might be an insert into file.
    (StringUtils.INAPPLICABLE_EXTRACT, StringUtils.INAPPLICABLE_EXTRACT)
  }

  // Extracts partition columns from the components.
  // It looks at the third argument, and if it is enclosed in square brackets,
  // returns the content inside as the partition columns.
  override def extractPartitions(components: Seq[String]): Option[String] = {
    components.lift(2).flatMap { partitions =>
      partitions match {
        case s if s.startsWith("[") && s.endsWith("]") => Some(s.substring(1, s.length - 1))
        case _ => None
      }
    }
  }
}

/**
 * Class for extracting metadata from Hive table insert operations.
 *
 * This class extends InsertIntoHadoopExtractNoCatalog and provides specialized logic
 * for parsing Hive table write log entries. It extracts the Hive SerDe format from the second
 * component of the log, interprets a boolean flag to determine the write mode
 * (either "Append" or "Overwrite"), and handles extraction of catalog information
 * (database and table names) where available. If a field cannot be properly parsed,
 * a default value (such as UNKNOWN_EXTRACT) is returned.
 *
 * The extracted metadata includes:
 *   - Execution name (with the "Execute" prefix removed)
 *   - Data format (converted from the Hive SerDe class to a user-friendly format)
 *   - Write mode (interpreted from the command parameters)
 *   - Catalog details (database and table names) if present
 *   - Output columns as defined in the log entry
 *
 * This consistent extraction of metadata facilitates downstream processing and reporting.
 */
case class InsertIntoHiveExtractor(
    override val nodeDescr: String) extends InsertIntoHadoopExtractNoCatalog(nodeDescr) {

  import InsertIntoHadoopExtract._

  // The format exists as the 2nd argument in most cases.
  private var _formatIndex = 1
  override def formatIndex: Int = _formatIndex
  override def formatIndex_=(value: Int): Unit = { _formatIndex = value }

  // Overrides extractFormat to obtain the Hive SerDe format.
  // The second element in the components sequence is expected to hold the Hive SerDe class string.
  // HiveParseHelper.getOptionalHiveFormat extracts the user-friendly format,
  // returning UNKNOWN_EXTRACT if not found.
  override def extractFormat(components: Seq[String]): String = {
    components.lift(formatIndex).flatMap { comp =>
      HiveParseHelper.getOptionalHiveFormat(comp)
    }.getOrElse(StringUtils.UNKNOWN_EXTRACT)
  }

  // Overrides extractLocation.
  // For InsertIntoHiveExtractor the location is not applicable, so it always returns Inapplicable.
  override def extractLocation(components: Seq[String]): Option[String] = {
    INAPPLICABLE_EXTRACT_OPTION
  }

  // Overrides extractWriteMode to determine the write operation mode.
  // It interprets the 3rd-to-last component (if present and if the last component starts with "[")
  // as a boolean flag indicating whether the query overwrites existing data.
  // A value of true returns "Overwrite" and false returns "Append".
  // If any of these conditions are not met or conversion to boolean fails, UNKNOWN_EXTRACT
  // is returned.
  override def extractWriteMode(components: Seq[String]): String = {
    // The write mode in the cmd exists as boolean as the 3rd to last argument.
    // The value is interpreted as a boolean whether the query overwrites existing data
    // (false means it's an append operation).
    if (components.length >= 3 && outColumnsIndex != -1) {
      // the last component is the output columns. Then we need the 3rd to last.
      components.lift(outColumnsIndex - 2)
        .flatMap(flag => Try(flag.toBoolean).toOption)
        .map { writeModeFlag =>
          if (writeModeFlag) {
            "Overwrite"
          } else {
            "Append"
          }
        }
        .getOrElse(StringUtils.UNKNOWN_EXTRACT)
    } else {
      StringUtils.UNKNOWN_EXTRACT
    }
  }

  override def extractOptions(components: Seq[String]): Map[String, String] = {
    val rawOptions = super.extractOptions(components)
    // Hive options do not show the compression. So, we inject unknown compression by default.
    rawOptions + ("compression" -> StringUtils.UNKNOWN_EXTRACT)
  }
}

/**
 * Trait defining the contract for extracting write operation metadata.
 * It specifies two methods: one to build the write operation record and another to check
 * if the extractor accepts a specific operation name.
 */
trait InsertCmdExtractorTrait {
  // Builds the write operation metadata given a node description string.
  def buildWriteOp(nodeDescr: String): WriteOperationMetadataTrait
  // Checks if the given operation name is accepted by this extractor.
  def accepts(opName: String): Boolean
}

/**
 * Object for extracting metadata from Hive insert operations.
 * It implements the InsertCmdExtractorTrait, delegating the build of metadata
 * to the InsertIntoHiveExtractor class, and verifying the operation using HiveParseHelper.
 */
object InsertIntoHiveExtract extends InsertCmdExtractorTrait {
  // Constructs the write operation metadata by instantiating InsertIntoHiveExtractor
  // and invoking its build method.
  def buildWriteOp(nodeDescr: String): WriteOperationMetadataTrait = {
    InsertIntoHiveExtractor(nodeDescr).build()
  }

  // Determines if the given operation name corresponds to a Hive table insert operation.
  def accepts(opName: String): Boolean = {
    HiveParseHelper.isHiveTableInsertNode(opName)
  }
}

/**
 * Object for extracting metadata from Hadoop insert operations.
 * It implements InsertCmdExtractorTrait and decides which extractor to use based on
 * whether the node description contains a CatalogTable indicator.
 */
object InsertIntoHadoopExtract extends InsertCmdExtractorTrait {
  // Regular expression to capture text enclosed in backticks. This is used to
  // extract potential catalog details like database and table names.
  val BACKTICKS_CATALOG_REG_EX: Regex = """`([^`]+)`""".r
  // Capture the reference to inapplicable extracts instead of locating a new one for each call.
  val INAPPLICABLE_EXTRACT_OPTION: Option[String] = Some(StringUtils.INAPPLICABLE_EXTRACT)

  // Builds the write operation metadata based on the node description.
  // If the description contains a CatalogTable entry, it uses InsertIntoHadoopExtractWithCatalog;
  // otherwise, it falls back to using InsertIntoHadoopExtractNoCatalog.
  def buildWriteOp(nodeDescr: String): WriteOperationMetadataTrait = {
    if (",\\s*CatalogTable\\(".r.findFirstIn(nodeDescr).isDefined) {
      InsertIntoHadoopExtractWithCatalog(nodeDescr).build()
    } else {
      new InsertIntoHadoopExtractNoCatalog(nodeDescr).build()
    }
  }

  // Determines if the operation name corresponds to a Hadoop insert command.
  // It checks if the opName contains the specific marker defined in DataWritingCommandExecParser.
  def accepts(opName: String): Boolean = {
    opName.contains(DataWritingCommandExecParser.insertIntoHadoopCMD)
  }
}
