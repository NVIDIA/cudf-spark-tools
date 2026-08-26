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

package com.nvidia.spark.rapids.tool

import java.io.FileNotFoundException
import java.time.LocalDateTime
import java.util.zip.ZipOutputStream

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, LinkedHashMap}
import scala.util.control.NonFatal

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileContext, FileStatus, FileSystem, Path, PathFilter}

import org.apache.spark.deploy.history.{EventLogFileReader, EventLogFileWriter}
import org.apache.spark.internal.Logging
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.rapids.tool.AppFilterImpl
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.util.FSUtils
import org.apache.spark.sql.rapids.tool.util.StringUtils

sealed trait EventLogInfo {
  def eventLog: Path
}

case class EventLogFileSystemInfo(timestamp: Long, size: Long)

case class ApacheSparkEventLog(override val eventLog: Path) extends EventLogInfo
case class DatabricksEventLog(override val eventLog: Path) extends EventLogInfo

case class FailedEventLog(override val eventLog: Path,
                          private val reason: String) extends EventLogInfo {
  def getReason: String = {
    StringUtils.renderStr(reason, doEscapeMetaCharacters = true, maxLength = 0)
  }
}

object EventLogPathProcessor extends Logging {
  // Apache Spark event log prefixes
  private val EVENT_LOG_DIR_NAME_PREFIX = "eventlog_v2_"
  private val DB_EVENT_LOG_FILE_NAME_PREFIX = "eventlog"

  private def isEventLogDir(status: FileStatus): Boolean = {
    status.isDirectory && isEventLogDir(status.getPath.getName)
  }

  // This only checks the name of the path
  private def isEventLogDir(path: String): Boolean = {
    path.startsWith(EVENT_LOG_DIR_NAME_PREFIX)
  }

  private def isDBEventLogFile(fileName: String): Boolean = {
    fileName.startsWith(DB_EVENT_LOG_FILE_NAME_PREFIX)
  }

  def isDBEventLogFile(status: FileStatus): Boolean = {
    status.isFile && isDBEventLogFile(status.getPath.getName)
  }

  // scalastyle:off line.size.limit
  // https://github.com/apache/spark/blob/0494dc90af48ce7da0625485a4dc6917a244d580/core/src/main/scala/org/apache/spark/io/CompressionCodec.scala#L67
  // scalastyle:on line.size.limit
  private val SPARK_SHORT_COMPRESSION_CODEC_NAMES = Set("lz4", "lzf", "snappy", "zstd")
  // Apache Spark ones plus gzip
  private val SPARK_SHORT_COMPRESSION_CODEC_NAMES_FOR_FILTER =
    SPARK_SHORT_COMPRESSION_CODEC_NAMES ++ Set("gz")

  // Files having these keywords are not considered as event logs
  private val EXCLUDED_EVENTLOG_NAME_KEYWORDS = Set("stdout", "stderr", "log4j", ".log.")

  /**
   * Expand event log inputs from text files containing paths or return direct event log paths.
   * If the path ends with .txt, it reads and expands the file to get event log paths.
   * Otherwise, it treats the path as a direct event log path.
   * Each non-empty line in txt files can optionally be a comma-separated list of paths.
   */
  private def expandEventLogsList(
     eventLogsPaths: List[String], hadoopConf: Configuration): List[String] = {
    val expandedPaths = eventLogsPaths.flatMap { pathString =>
      if (pathString.toLowerCase.endsWith(".txt")) {
        // The paths coming from the txt file have to normalised again
        // to ensure compatibility with Hadoop FS
        normalizeS3Paths(expandTextFile(pathString, hadoopConf))
      } else {
        // In case of a non-txt file, we assume it is a valid event log path
        // This can later be expanded to support other file types
        List(pathString)
      }
    }

    if (expandedPaths.isEmpty && eventLogsPaths.nonEmpty) {
      logWarning("No valid event log paths found. Ensure .txt files contain valid event log paths.")
    }

    expandedPaths
  }

  /**
   * Reads a text file and returns a flat list of all parsed paths in the order they appear.
   * This function splits any sequence of event log paths by newlines, whitespace, comma
   * or semicolon. If the file is readable but yields no valid tokens, a warning is logged
   * and an empty list is returned.
   */
  private def expandTextFile(filePath: String, hadoopConf: Configuration): List[String] = {
    try {
      // Load the file content (remote or local)
      val content = FSUtils.readFileContentAsUTF8(
        filePath = filePath,
        hadoopConf = Some(hadoopConf))
      // Split on new lines, whitespace, comma, or semicolon in one pass
      val tokens = content
        .split("[\\s,;]+")
        .toList
        .filter(_.nonEmpty)
      if (tokens.isEmpty) {
        logWarning(s"No valid event log paths found in file: $filePath")
      }
      tokens
    } catch {
      case NonFatal(e) =>
        logWarning(s"Failed to read event log list file: $filePath. Skipping file.", e)
        List.empty
    }
  }

  /**
   * Normalize S3-style URIs to Hadoop's s3a scheme.
   * Converts prefixes `s3://` and `s3n://` to `s3a://`.
   */
  private def normalizeS3Paths(paths: List[String]): List[String] = {
    paths.map { p =>
      // Going with stripPrefix to add safety and less regex computation time
      if (p.startsWith("s3://")) {
        "s3a://" + p.stripPrefix("s3://")
      } else if (p.startsWith("s3n://")) {
        "s3a://" + p.stripPrefix("s3n://")
      } else {
        p
      }
    }
  }

  /**
   * Filter to identify valid event log files based on the criteria:
   *  - File should either not have any suffix or have a supported compression codec suffix
   *  - File should not contain any of the EXCLUDED_EVENTLOG_NAME_KEYWORDS keywords in its name
   * @param logFile File to be filtered.
   * @return        True if the file is a valid event log, false otherwise.
   */
  private def eventLogNameFilter(logFile: Path): Boolean = {
    val hasValidSuffix = EventLogFileWriter.codecName(logFile)
      .forall(suffix => SPARK_SHORT_COMPRESSION_CODEC_NAMES_FOR_FILTER.contains(suffix))
    val hasExcludedKeyword = EXCLUDED_EVENTLOG_NAME_KEYWORDS.exists(logFile.getName.contains)
    hasValidSuffix && !hasExcludedKeyword
  }

  // Databricks has the latest events in file named eventlog and then any rolled in format
  // eventlog-2021-06-14--20-00.gz, here we assume that if any files start with eventlog
  // then the directory is a Databricks event log directory.
  private def isDatabricksEventLogDir(dir: FileStatus, fs: FileSystem): Boolean = {
    lazy val dbLogFiles = fs.listStatus(dir.getPath, new PathFilter {
      override def accept(path: Path): Boolean = {
        isDBEventLogFile(path.getName)
      }
    })
    dir.isDirectory && dbLogFiles.size > 1
  }

  /**
   * Identifies if the input file or directory is a valid event log.
   *
   * TODO - Need to handle size of files in directory, for now document its not supported.
   *        Reference: https://github.com/NVIDIA/cudf-spark-tools/pull/1275
   *
   * @param s   FileStatus to be identified.
   * @param fs  FileSystem instance for file system operations.
   * @return    Option[EventLogInfo] if valid, None otherwise.
   */
  private def identifyEventLog(s: FileStatus, fs: FileSystem): Option[EventLogInfo] = {
    if (s.isFile && eventLogNameFilter(s.getPath)) {
      // Regular event log file. See function `eventLogNameFilter` for criteria.
      Some(ApacheSparkEventLog(s.getPath))
    } else if (isEventLogDir(s)) {
      // Apache Spark event log directory (starting with "eventlog_v2_").
      Some(ApacheSparkEventLog(s.getPath))
    } else if (isDatabricksEventLogDir(s, fs)) {
      // Databricks event log directory (files starting with "eventlog").
      Some(DatabricksEventLog(s.getPath))
    } else {
      // Ignore other types of files.
      None
    }
  }

  /**
   * Retrieves all event logs from the input path.
   *
   * Note:
   *   - If the input path is a directory and recursive search is enabled, this function will
   *     search for event logs in all subdirectories.
   *   - This is done using a queue to avoid stack overflow.
   *
   * @param inputPath               Root path to search for event logs.
   * @param fs                      FileSystem instance for file system operations.
   * @param recursiveSearchEnabled  If enabled, search for event logs in all subdirectories.
   * @return                        List of (EventLogInfo, EventLogFileSystemInfo) tuples.
   */
  private def getEventLogInfoInternal(inputPath: Path, fs: FileSystem,
      recursiveSearchEnabled: Boolean): List[(EventLogInfo, EventLogFileSystemInfo)] = {
    val results = ArrayBuffer[(EventLogInfo, EventLogFileSystemInfo)]()
    val queue = mutable.Queue[FileStatus]()
    queue.enqueue(fs.getFileStatus(inputPath))
    while (queue.nonEmpty) {
      // Note: currentEntry can be a file, directory, or a symlink.
      val currentEntry = queue.dequeue()
      identifyEventLog(currentEntry, fs) match {
        case Some(eventLogInfo) =>
          // If currentEntry is an event log (either a file or a directory), add it to the results
          // with its modification time and size.
          results += eventLogInfo -> EventLogFileSystemInfo(
            currentEntry.getModificationTime, currentEntry.getLen)
        case None if currentEntry.isDirectory =>
          // If currentEntry is a directory (but not an event log) enqueue all contained files for
          // further processing. Include subdirectories only if recursive search is enabled.
          fs.listStatus(currentEntry.getPath)
            .filter(child => child.isFile || (child.isDirectory && recursiveSearchEnabled))
            .foreach(queue.enqueue(_))
        case _ =>
          // Using a debug log (instead of warn) to avoid excessive logging due to recursive nature.
          val supportedTypes = SPARK_SHORT_COMPRESSION_CODEC_NAMES_FOR_FILTER.mkString(", ")
          logDebug(s"File: ${currentEntry.getPath} is not a supported file type. " +
            s"Supported compression types are: $supportedTypes. Skipping this file.")
      }
    }
    results.toList
  }

  /**
   * If the user provides wildcard in the eventlogs, then the path processor needs
   * to process the path as a pattern. Otherwise, HDFS throws an exception mistakenly that
   * no such file exists.
   * Once the glob path is checked, the list of eventlogs is the result of the flattenMap
   * of all the processed files.
   *
   * @return List of (rawPath, List[processedPaths])
   */
  private def processWildcardsLogs(eventLogsPaths: List[String],
      hadoopConf: Configuration): List[(String, List[String])] = {
    eventLogsPaths.map { rawPath =>
      if (!rawPath.contains("*")) {
        (rawPath, List(rawPath))
      } else {
        try {
          val globPath = new Path(rawPath)
          val fileContext = FileContext.getFileContext(globPath.toUri, hadoopConf)
          val fileStatuses = fileContext.util().globStatus(globPath)
          val paths = fileStatuses.map(_.getPath.toString).toList
          (rawPath, paths)
        } catch {
          case _ : Throwable =>
            // Do not fail in this block.
            // Instead, ignore the error and add the file as is; then the caller should fail when
            // processing the file.
            // This will make handling errors more consistent during the processing of the analysis
            logWarning(s"Processing pathLog with wildCard has failed: $rawPath")
            (rawPath, List.empty)
        }
      }
    }
  }

  def getEventLogInfo(pathString: String, hadoopConf: Configuration,
      recursiveSearchEnabled: Boolean = true): Map[EventLogInfo, Option[EventLogFileSystemInfo]] = {
    val inputPath = new Path(pathString)
    try {
      // Note that some cloud storage APIs may throw FileNotFoundException when the pathPrefix
      // (i.e., bucketName) is not visible to the current configuration instance.
      val fs = FSUtils.getFSForPath(inputPath, hadoopConf)
      val eventLogInfos = getEventLogInfoInternal(inputPath, fs, recursiveSearchEnabled)
      if (eventLogInfos.isEmpty) {
        val message = "No valid event logs found in the path. Check the path or set the " +
          "log level to DEBUG for more details."
        Map(FailedEventLog(inputPath, message) -> None)
      } else {
        eventLogInfos.toMap.iterator.map { case (k, v) => k -> Some(v) }.toMap
      }
    } catch {
      case fnfEx: FileNotFoundException =>
        logWarning(s"$pathString not found, skipping!")
        Map(FailedEventLog(new Path(pathString), fnfEx.getMessage) -> None)
      case NonFatal(e) =>
        logWarning(s"Unexpected exception occurred reading $pathString, skipping!", e)
        Map(FailedEventLog(new Path(pathString), e.getMessage) -> None)
    }
  }

  /**
   * Function to process event log paths passed in and evaluate which ones are really event
   * logs and filter based on user options.
   *
   * @param filterNLogs             Number of event logs to be selected
   * @param matchlogs               Keyword to match file names in the directory
   * @param eventLogsPaths          Array of event log paths
   * @param hadoopConf              Hadoop Configuration
   * @param recursiveSearchEnabled  If enabled, search for event logs in all subdirectories.
   *                                Enabled by default.
   * @param maxEventLogSize         Optional maximum event log size as a string
   * @param minEventLogSize         Optional minimum event log size as a string
   * @param fsStartTime             Filesystem date and time for filtering event logs based on
   *                                start time
   * @param fsEndTime               Filesystem date and time for filtering event logs based on
   *                                end time
   * @return (Seq[EventLogInfo], Seq[EventLogInfo]) - Tuple indicating paths of event logs in
   *         filesystem. First element contains paths of event logs after applying filters and
   *         second element contains paths of all event logs.
   */
  def processAllPaths(
      filterNLogs: Option[String],
      matchlogs: Option[String],
      eventLogsPaths: List[String],
      hadoopConf: Configuration,
      recursiveSearchEnabled: Boolean = true,
      maxEventLogSize: Option[String] = None,
      minEventLogSize: Option[String] = None,
      fsStartTime: Option[String] = None,
      fsEndTime: Option[String] = None): (Seq[EventLogInfo], Seq[EventLogInfo]) = {
    val normalizedPaths = normalizeS3Paths(eventLogsPaths)
    val expandedPaths = expandEventLogsList(normalizedPaths, hadoopConf)
    val logsPathNoWildCards = processWildcardsLogs(expandedPaths, hadoopConf)
    val logsWithTimestamp = logsPathNoWildCards.flatMap {
      case (rawPath, processedPaths) if processedPaths.isEmpty =>
        // If no event logs are found in the path after wildcard expansion, return a failed event
        Map(FailedEventLog(new Path(rawPath), s"No event logs found in $rawPath") -> None)
      case (_, processedPaths) =>
        processedPaths.flatMap(getEventLogInfo(_, hadoopConf, recursiveSearchEnabled))
    }.toMap

    logDebug("Paths after stringToPath: " + logsWithTimestamp)
    // Filter the event logs to be processed based on the criteria. If it is not provided in the
    // command line, then return all the event logs processed above.
    val matchedLogs = matchlogs.map { strMatch =>
      logsWithTimestamp.iterator
        .filter { case (k, _) => k.eventLog.getName.contains(strMatch) }.toMap
    }.getOrElse(logsWithTimestamp)

    val filteredLogs = if ((filterNLogs.nonEmpty && !filterByAppCriteria(filterNLogs)) ||
      maxEventLogSize.isDefined || minEventLogSize.isDefined ||
      fsStartTime.isDefined || fsEndTime.isDefined) {
      val validMatchedLogs = matchedLogs.collect {
        case (info, Some(ts)) => info -> ts
      }
      val filteredByMinSize = if (minEventLogSize.isDefined) {
        // size is assumed to be mb (see QualificationArgs)
        val minSizeInBytes = StringUtils.convertMemorySizeToBytes(minEventLogSize.get,
          Some(ByteUnit.MiB))
        val (matched, filtered) = validMatchedLogs.partition(info => info._2.size >= minSizeInBytes)
        logInfo(s"Filtering eventlogs by size, minimum size is ${minSizeInBytes}b. " +
          s"The logs filtered out include: " +
          s"${filtered.iterator.map { case (info, _) => info.eventLog.toString }.mkString(",")}")
        matched
      } else {
        validMatchedLogs
      }
      val filteredByMaxSize = if (maxEventLogSize.isDefined) {
        // size is assumed to be mb (see QualificationArgs)
        val maxSizeInBytes = StringUtils.convertMemorySizeToBytes(maxEventLogSize.get,
          Some(ByteUnit.MiB))
        val (matched, filtered) =
          filteredByMinSize.partition(info => info._2.size <= maxSizeInBytes)
        logInfo(s"Filtering eventlogs by size, max size is ${maxSizeInBytes}b. " +
          s"The logs filtered out include: " +
          s"${filtered.iterator.map { case (info, _) => info.eventLog.toString }.mkString(",")}")
        matched
      } else {
        filteredByMinSize
      }
      // filter by date time range first
      val filteredByDateTime = if (fsStartTime.isDefined || fsEndTime.isDefined) {
        if (fsStartTime.isDefined && fsEndTime.isDefined) {
          val startTimeMs = AppFilterImpl.parseDateTimePeriod(fsStartTime.get).get
          val endTimeMs = AppFilterImpl.parseDateTimePeriod(fsEndTime.get).get
          val (matched, filtered) = filteredByMaxSize.partition { case (_, v) =>
            (v.timestamp >= startTimeMs) && (v.timestamp <= endTimeMs)
          }
          logInfo(s"Filtered out event logs based on both fs start time: ${fsStartTime.get} " +
            s"and fs end time: ${fsEndTime.get}. The logs filtered out include: " +
            s"${filtered.iterator.map { case (info, _) => info.eventLog.toString }.mkString(",")}")
          matched
        } else if (fsStartTime.isDefined) {
          val startTimeMs = AppFilterImpl.parseDateTimePeriod(fsStartTime.get).get
          val (matched, filtered) =
            filteredByMaxSize.partition { case (_, v) => v.timestamp >= startTimeMs }
          logInfo(s"Filtered out event logs based on fs start time: ${fsStartTime.get} " +
            s"The logs filtered out include: " +
            s"${filtered.iterator.map { case (info, _) => info.eventLog.toString }.mkString(",")}")
          matched
        } else {
          val endTimeMs = AppFilterImpl.parseDateTimePeriod(fsEndTime.get).get
          val (matched, filtered) =
            filteredByMaxSize.partition { case (_, v) => v.timestamp <= endTimeMs }
          logInfo(s"Filtered out event logs based on fs end time: ${fsEndTime.get} " +
            s"The logs filtered out include: " +
            s"${filtered.iterator.map { case (info, _) => info.eventLog.toString }.mkString(",")}")
          matched
        }
      } else {
        filteredByMaxSize
      }
      if (filterNLogs.nonEmpty && !filterByAppCriteria(filterNLogs)) {
        val filteredInfo = filterNLogs.get.split("-")
        val numberofEventLogs = filteredInfo(0).toInt
        val criteria = filteredInfo(1)
        // Before filtering based on user criteria, remove the failed event logs
        // (i.e. logs without timestamp) from the list.
        val matched = if (criteria.equals("newest")) {
          LinkedHashMap(filteredByDateTime.toSeq.sortWith(_._2.timestamp > _._2.timestamp): _*)
        } else if (criteria.equals("oldest")) {
          LinkedHashMap(filteredByDateTime.toSeq.sortWith(_._2.timestamp < _._2.timestamp): _*)
        } else {
          logError("Criteria should be either newest-filesystem or oldest-filesystem")
          Map.empty[EventLogInfo, Long]
        }
        matched.take(numberofEventLogs)
      } else {
        filteredByDateTime
      }
    } else {
      matchedLogs
    }
    (filteredLogs.iterator.map(_._1).toSeq, logsWithTimestamp.iterator.map(_._1).toSeq)
  }

  def filterByAppCriteria(filterNLogs: Option[String]): Boolean = {
    filterNLogs.get.endsWith("-oldest") || filterNLogs.get.endsWith("-newest") ||
        filterNLogs.get.endsWith("per-app-name")
  }

  def logApplicationInfo(app: ApplicationInfo) = {
    logDebug(s"==============  ${app.appId} ==============")
  }

  def getDBEventLogFileDate(eventLogFileName: String): LocalDateTime = {
    if (!isDBEventLogFile(eventLogFileName)) {
      logError(s"$eventLogFileName Not an event log file!")
    }
    val fileParts = eventLogFileName.split("--")
    if (fileParts.size < 2) {
      // assume this is the current log and we want that one to be read last
      LocalDateTime.now()
    } else {
      val date = fileParts(0).split("-")
      val day = Integer.parseInt(date(3))
      val month = Integer.parseInt(date(2))
      val year = Integer.parseInt(date(1))
      val time = fileParts(1).split("-")
      val minParse = time(1).split('.')
      val hour = Integer.parseInt(time(0))
      val min = Integer.parseInt(minParse(0))
      LocalDateTime.of(year, month, day, hour, min)
    }
  }
}

/**
 * The reader which will read the information of Databricks rolled multiple event log files.
 */
class DatabricksRollingEventLogFilesFileReader(
    fs: FileSystem,
    path: Path) extends EventLogFileReader(fs, path) with Logging {

  private lazy val files: Seq[FileStatus] = {
    val ret = fs.listStatus(rootPath).toSeq
    if (!ret.exists(EventLogPathProcessor.isDBEventLogFile)) {
      Seq.empty[FileStatus]
    } else {
      ret
    }
  }

  private lazy val eventLogFiles: Seq[FileStatus] = {
    files.filter(EventLogPathProcessor.isDBEventLogFile).sortWith { (status1, status2) =>
      val dateTime = EventLogPathProcessor.getDBEventLogFileDate(status1.getPath.getName)
      val dateTime2 = EventLogPathProcessor.getDBEventLogFileDate(status2.getPath.getName)
      dateTime.isBefore(dateTime2)
    }
  }

  override def completed: Boolean = true
  override def modificationTime: Long = lastEventLogFile.getModificationTime
  private def lastEventLogFile: FileStatus = eventLogFiles.last
  override def listEventLogFiles: Seq[FileStatus] = eventLogFiles

  // unused functions
  override def compressionCodec: Option[String] = None
  override def totalSize: Long = 0
  override def zipEventLogFiles(zipStream: ZipOutputStream): Unit = {}
  override def fileSizeForLastIndexForDFS: Option[Long] = None
  override def fileSizeForLastIndex: Long = lastEventLogFile.getLen
  override def lastIndex: Option[Long] = None
}
