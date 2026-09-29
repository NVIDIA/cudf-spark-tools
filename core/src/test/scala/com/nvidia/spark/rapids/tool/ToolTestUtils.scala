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

import java.io.{File, FilenameFilter, FileNotFoundException}

import scala.collection.mutable.ArrayBuffer

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.nvidia.spark.rapids.tool.profiling.ProfileArgs
import com.nvidia.spark.rapids.tool.qualification.QualOutputWriter
import com.nvidia.spark.rapids.tool.tuning._
import com.nvidia.spark.rapids.tool.tuning.config.{TuningConfigEntry, TuningConfiguration, TuningEntryDefinition}
import org.apache.hadoop.fs.Path
import org.yaml.snakeyaml.{DumperOptions, Yaml}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession, TrampolineUtil}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.rapids.tool.ClusterSummary
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.util.RapidsToolsConfUtil
import org.apache.spark.sql.types._

object ToolTestUtils extends Logging {
  val RAPIDS_MVN_BASE_URL =
    "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/"

  // Scheme fields for Status Report file
  private val csvStatusFields = Seq(
    (QualOutputWriter.STATUS_REPORT_PATH_STR, StringType),
    (QualOutputWriter.STATUS_REPORT_STATUS_STR, StringType),
    (QualOutputWriter.STATUS_REPORT_APP_ID, StringType),
    (QualOutputWriter.STATUS_REPORT_DESC_STR, StringType))

  val statusReportSchema =
    new StructType(csvStatusFields.map(f => StructField(f._1, f._2)).toArray)

  def pluginMvnURL(version: String): String = {
    s"$RAPIDS_MVN_BASE_URL$version"
  }
  def pluginMvnPrefix(version: String): String = {
    s"${pluginMvnURL(version)}/rapids-4-spark_2.12-$version"
  }

  def getTestResourceFile(file: String): File = {
    new File(getClass.getClassLoader.getResource(file).getFile)
  }

  def getTestResourcePath(file: String): String = {
    getTestResourceFile(file).getCanonicalPath
  }

  def runAndCollect(appName: String)
    (fun: SparkSession => DataFrame): String = {

    // we need to close any existing sessions to ensure that we can
    // create a session with a new event log dir
    TrampolineUtil.cleanupAnyExistingSession()

    lazy val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName(appName)
      .getOrCreate()

    // execute the query and generate events
    val df = fun(spark)
    df.collect()

    val appId = spark.sparkContext.applicationId

    // close the event log
    spark.close()
    appId
  }

  def generateEventLog(eventLogDir: File, appName: String,
      confs: Option[Map[String, String]] = None,
      enableHive: Boolean = false)
    (fun: SparkSession => DataFrame): (String, String) = {

    // we need to close any existing sessions to ensure that we can
    // create a session with a new event log dir
    TrampolineUtil.cleanupAnyExistingSession()

    lazy val sparkBuilder = SparkSession
      .builder()
      .master("local[*]")
      .appName(appName)
      .config("spark.eventLog.enabled", "true")
      .config("spark.eventLog.dir", eventLogDir.getAbsolutePath)
    if (enableHive) {
      sparkBuilder.enableHiveSupport()
    }
    confs.foreach(_.foreach {case (k, v) => sparkBuilder.config(k, v)})
    lazy val spark = sparkBuilder.getOrCreate()

    // execute the query and generate events
    val df = fun(spark)
    df.collect()

    val appId = spark.sparkContext.applicationId

    // close the event log
    spark.close()

    // find the event log
    val files = listFilesMatching(eventLogDir, !_.startsWith("."))
    if (files.length != 1) {
      throw new FileNotFoundException(s"Could not find event log in ${eventLogDir.getAbsolutePath}")
    }
    (files.head.getAbsolutePath, appId)
  }

  def listFilesMatching(dir: File, matcher: String => Boolean): Array[File] = {
    dir.listFiles(new FilenameFilter {
      override def accept(file: File, s: String): Boolean = matcher(s)
    })
  }

  def compareDataFrames(df: DataFrame, expectedDf: DataFrame): Unit = {
    val diffCount = df.except(expectedDf).union(expectedDf.except(df)).count
    if (diffCount != 0) {
      logWarning("Diff expected vs actual:")
      expectedDf.show(1000, truncate = false)
      df.show(1000, truncate = false)
    }
    assert(diffCount == 0)
  }

  def readExpectationCSV(sparkSession: SparkSession, path: String,
      schema: Option[StructType] = None, escape: String = "\\"): DataFrame = {
    // make sure to change null value so empty strings don't show up as nulls
    if (schema.isDefined) {
      sparkSession.read.option("header", "true").option("nullValue", "-").option("escape", escape)
        .schema(schema.get).csv(path)
    } else {
      sparkSession.read.option("header", "true").option("nullValue", "-").option("escape", escape)
        .csv(path)
    }
  }

  def processProfileApps(logs: Array[String],
      sparkSession: SparkSession): ArrayBuffer[ApplicationInfo] = {
    val apps: ArrayBuffer[ApplicationInfo] = ArrayBuffer[ApplicationInfo]()
    val appArgs = new ProfileArgs(logs)
    val enableDiagnosticViews = appArgs.enableDiagnosticViews()
    var index: Int = 1
    val platform = PlatformFactory.createInstance(appArgs.platform())
    for (path <- appArgs.eventlog()) {
      val eventLogInfo = EventLogPathProcessor
        .getEventLogInfo(path, RapidsToolsConfUtil.newHadoopConf())
      assert(eventLogInfo.nonEmpty, s"event log not parsed as expected $path")
      apps += new ApplicationInfo(RapidsToolsConfUtil.newHadoopConf(),
        eventLogInfo.head._1, platform, enableDiagnosticViews)
      index += 1
    }
    apps
  }

  /**
   * Performs an expected status report dataframe to validate the counts
   * of different status categories (SUCCESS, FAILURE, UNKNOWN, SKIPPED).
   */
  def compareStatusReport(sparkSession: SparkSession, expStatusReportCount: StatusReportCounts,
      filePath: String): Unit = {
    val csvFile = new File(filePath)

    // If the status report file does not exist, all applications are expected to be failures.
    if(!csvFile.exists()) {
      assert(expStatusReportCount.success == 0 &&
        expStatusReportCount.unknown == 0 &&
        expStatusReportCount.failure > 0)
    } else {
      val expectedStatusDf = readExpectationCSV(sparkSession, filePath, Some(statusReportSchema))

      // Function to count the occurrences of a specific status in the DataFrame
      def countStatus(status: String): Long =
        expectedStatusDf.filter(col("STATUS") === lit(status)).count()

      val actualStatusReportCount = StatusReportCounts(
        countStatus("SUCCESS"),
        countStatus("FAILURE"),
        countStatus("SKIPPED"),
        countStatus("UNKNOWN"))
      assert(actualStatusReportCount == expStatusReportCount,
        s"Expected status report counts: $expStatusReportCount, " +
          s"but got: $actualStatusReportCount")
    }
  }

  /**
   * Load a JSON file containing ClusterSummary objects.
   * @param jsonFile The JSON file to load.
   * @return ClusterSummary object.
   */
  def loadClusterSummaryFromJson(jsonFile: File): ClusterSummary = {
    val mapper = new ObjectMapper().registerModule(DefaultScalaModule)
    mapper.readValue(jsonFile, classOf[ClusterSummary])
  }

  /**
   * Compares two JSON strings for equality, ignoring field ordering within objects.
   * This is useful for testing JSON output where field order may vary between
   * different JSON library versions or configurations.
   *
   * @param actualJson   The actual JSON string
   * @param expectedJson The expected JSON string
   * @return true if the JSON structures are equivalent, false otherwise
   */
  def compareJsonIgnoringFieldOrder(actualJson: String, expectedJson: String): Boolean = {
    try {
      val mapper = new ObjectMapper().registerModule(DefaultScalaModule)

      // Parse both JSON strings into JsonNode objects
      val actualNode = mapper.readTree(actualJson)
      val expectedNode = mapper.readTree(expectedJson)

      // JsonNode.equals() compares structure and values but ignores field order
      actualNode.equals(expectedNode)
    } catch {
      case _: Exception =>
        // If JSON parsing fails, fall back to string comparison
        actualJson.trim == expectedJson.trim
    }
  }

  def buildTargetClusterInfo(
      driverNodeInstanceType: Option[String] = None,
      workerNodeInstanceType: Option[String] = None,
      cpuCores: Option[Int] = None,
      memoryGB: Option[Long] = None,
      gpuCount: Option[Int] = None,
      gpuMemory: Option[String] = None,
      gpuDevice: Option[String] = None,
      enforcedSparkProperties: Map[String, String] = Map.empty,
      preserveSparkProperties: List[String] = List.empty,
      excludeSparkProperties: List[String] = List.empty,
      tuningDefinitions: java.util.List[TuningEntryDefinition] =
        new java.util.ArrayList[TuningEntryDefinition]()): TargetClusterProps = {
    val driverProps = new DriverInfo(driverNodeInstanceType.getOrElse(""))
    import scala.jdk.CollectionConverters._
    val gpuWorkerProps = new GpuWorkerProps(
      gpuMemory.getOrElse(""), gpuCount.getOrElse(0), gpuDevice.getOrElse(""))
   val workerProps = new WorkerInfo(workerNodeInstanceType.getOrElse(""), cpuCores.getOrElse(0),
      memoryGB.getOrElse(0L), gpuWorkerProps)
    val sparkProps = new SparkProperties()
    sparkProps.getEnforced.putAll(enforcedSparkProperties.asJava)
    sparkProps.getPreserve.addAll(preserveSparkProperties.asJava)
    sparkProps.getExclude.addAll(excludeSparkProperties.asJava)
    sparkProps.setTuningDefinitions(tuningDefinitions)
    new TargetClusterProps(driverProps, workerProps, sparkProps)
  }

  def buildTargetClusterInfoAsString(
      driverNodeInstanceType: Option[String] = None,
      workerNodeInstanceType: Option[String] = None,
      cpuCores: Option[Int] = None,
      memoryGB: Option[Long] = None,
      gpuCount: Option[Int] = None,
      gpuMemory: Option[String] = None,
      gpuDevice: Option[String] = None,
      enforcedSparkProperties: Map[String, String] = Map.empty,
      preserveSparkProperties: List[String] = List.empty,
      excludeSparkProperties: List[String] = List.empty,
      tuningDefinitions: java.util.List[TuningEntryDefinition] =
        new java.util.ArrayList[TuningEntryDefinition]()): String = {
    val targetCluster = buildTargetClusterInfo(driverNodeInstanceType, workerNodeInstanceType,
      cpuCores, memoryGB, gpuCount, gpuMemory, gpuDevice, enforcedSparkProperties,
      preserveSparkProperties, excludeSparkProperties, tuningDefinitions)
    // set the options to convert the object into formatted yaml content
    val options = new DumperOptions()
    options.setIndent(2)
    options.setPrettyFlow(true)
    options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
    val yaml = new Yaml(options)
    val rawString = yaml.dump(targetCluster)
    // Skip the first line as it contains "the name of the class"
    rawString.split("\n").drop(1).mkString("\n")
  }

  def createTargetClusterInfoFile(
       outputDirectory: String,
       driverNodeInstanceType: Option[String] = None,
       workerNodeInstanceType: Option[String] = None,
       cpuCores: Option[Int] = None,
       memoryGB: Option[Long] = None,
       gpuCount: Option[Int] = None,
       gpuMemory: Option[String] = None,
       gpuDevice: Option[String] = None,
       enforcedSparkProperties: Map[String, String] = Map.empty,
       preserveSparkProperties: List[String] = List.empty,
       excludeSparkProperties: List[String] = List.empty,
       tuningDefinitions: java.util.List[TuningEntryDefinition] =
         new java.util.ArrayList[TuningEntryDefinition]()): Path = {
    val fileWriter = new ToolTextFileWriter(outputDirectory, "targetClusterInfo.yaml",
      "Target Cluster Info")
    try {
      val targetClusterInfoString = buildTargetClusterInfoAsString(driverNodeInstanceType,
        workerNodeInstanceType, cpuCores, memoryGB, gpuCount, gpuMemory, gpuDevice,
        enforcedSparkProperties, preserveSparkProperties, excludeSparkProperties, tuningDefinitions)
      fileWriter.write(targetClusterInfoString)
      fileWriter.getFileOutputPath
    } finally {
      fileWriter.close()
    }
  }

  def buildTuningConfigs(
      default: List[TuningConfigEntry] = List.empty,
      qualification: List[TuningConfigEntry] = List.empty,
      profiling: List[TuningConfigEntry] = List.empty): TuningConfiguration = {
    import scala.jdk.CollectionConverters._
    new TuningConfiguration(
        default = default.asJava,
        qualification = qualification.asJava,
        profiling = profiling.asJava)
  }

  /**
   * Given a directory, find all files recursively in the directory with the given file name.
   * Example usage is to crate a list of all output files in a directory.
   * @param dir The directory root
   * @param fileName the file name to be loaded
   * @return a list of files with the given file name
   */
  def findFilesRecursively(dir: File, fileName: String): Seq[File] = {
    val files = Option(dir.listFiles).getOrElse(Array.empty[File])
    val matchedFiles = files.filter(f => f.isFile && f.getName == fileName)
    val subDirs = files.filter(_.isDirectory).flatMap(d => findFilesRecursively(d, fileName))
    matchedFiles ++ subDirs
  }
}

/**
 * Case class representing the counts of different status categories in a status report.
 */
case class StatusReportCounts(success: Long, failure: Long, skipped: Long, unknown: Long)
