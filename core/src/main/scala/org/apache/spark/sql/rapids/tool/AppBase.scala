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

package org.apache.spark.sql.rapids.tool

import java.io.InputStream
import java.util.zip.GZIPInputStream

import scala.collection.{immutable, mutable}
import scala.collection.mutable.{ArrayBuffer, HashMap, HashSet, LinkedHashSet, Map}

import com.nvidia.spark.rapids.SparkRapidsBuildInfoEvent
import com.nvidia.spark.rapids.tool.{DatabricksEventLog, DatabricksRollingEventLogFilesFileReader, EventLogInfo, Identifiable, Platform}
import com.nvidia.spark.rapids.tool.planparser.{BatchScanExecParser, ReadParser}
import com.nvidia.spark.rapids.tool.planparser.hive.HiveParseHelper
import com.nvidia.spark.rapids.tool.profiling.{BlockManagerRemovedCase, ConnectOperationInfo, ConnectSessionInfo, DriverAccumCase, JobInfoClass, ResourceProfileInfoCase, SQLExecutionInfoClass, SQLPlanMetricsCase}
import com.nvidia.spark.rapids.tool.qualification.AppSubscriber
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.deploy.history.{EventLogFileReader, EventLogFileWriter}
import org.apache.spark.internal.Logging
import org.apache.spark.rapids.tool.benchmarks.RuntimeInjector
import org.apache.spark.scheduler.{SparkListenerEvent, StageInfo}
import org.apache.spark.sql.rapids.tool.plangraph.{SparkPlanGraphNode, ToolsPlanGraph}
import org.apache.spark.sql.rapids.tool.store.{AccumManager, DataSourceRecord, SparkPlanInfoTruncated, SQLPlanModel, SQLPlanModelManager, StageModel, StageModelManager, TaskModelManager, WriteOperationRecord}
import org.apache.spark.sql.rapids.tool.util.{CacheablePropsHandler, EventUtils, RapidsToolsConfUtil, StringUtils, SuccessAppResult, UTF8Source}
import org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanExtensions.SparkPlanInfoOps
import org.apache.spark.sql.rapids.tool.util.stubs.SparkPlanInfo
import org.apache.spark.util.Utils

abstract class AppBase(
    val eventLogInfo: Option[EventLogInfo],
    val hadoopConf: Option[Configuration],
    val platform: Option[Platform] = None) extends Logging
  with CacheablePropsHandler
  with AccumToStageRetriever
  with Identifiable[String]
  with EventLogParserTrait {

  /**
   * The event log path is used as the unique identifier for the application.
   * It is retrieved from the `eventLogInfo` if available, otherwise a default value is used.
   * Later, we can implement a more sophisticated way to generate a unique identifier for each app.
   * For example, hashing a composite struct of metadata fields (i.e., SQLPlan hash).
   */
  private val _id: String = getEventLogPath

  var appMetaData: Option[AppMetaData] = None

  /**
   * Retrieves the unique identifier for the application.
   * If `appMetaData` is defined, the identifier is retrieved from the metadata.
   * Otherwise, a default identifier (`_id`) is returned.
   *
   * @return A `String` representing the unique identifier of the application.
   */
  override def id: String = {
    appMetaData match {
      case Some(meta) => meta.id
      case _ => _id
    }
  }

  // appId is string is stored as a field in the AppMetaData class
  def appId: String = {
    appMetaData match {
      case Some(meta) => meta.appId.getOrElse("")
      case _ => ""
    }
  }

  lazy val attemptId: Int = appMetaData.map(_.attemptId).getOrElse(1)

  // This is to keep track of the high water mark for maximum number of executors
  // active at any point in time.
  // Dynamic allocation and failures could change the number of
  // executors over time so we are tracking what was the maximum used.
  var maxNumExecutorsRunning = 0
  // high water mark for maximum number of nodes in use at any point in time
  var maxNumNodesRunning = 0
  // Store map of executorId to executor info
  val executorIdToInfo = new HashMap[String, ExecutorInfoClass]()
  // resourceProfile id to resource profile info
  val resourceProfIdToInfo = new HashMap[Int, ResourceProfileInfoCase]()
  var blockManagersRemoved: ArrayBuffer[BlockManagerRemovedCase] =
    ArrayBuffer[BlockManagerRemovedCase]()
  // The data source information
  val dataSourceInfo: ArrayBuffer[DataSourceRecord] = ArrayBuffer[DataSourceRecord]()

  // jobId to job info
  val jobIdToInfo = new HashMap[Int, JobInfoClass]()
  val jobIdToSqlID: HashMap[Int, Long] = HashMap.empty[Int, Long]

  lazy val sqlManager = new SQLPlanModelManager(this)

  // SQL containing any Dataset operation or RDD to DataSet/DataFrame operation
  val sqlIDToDataSetOrRDDCase: HashSet[Long] = HashSet[Long]()
  // Map (sqlID <-> String(problematic issues))
  // Use LinkedHashSet of Strings to preserve the order of insertion.
  val sqlIDtoProblematic: HashMap[Long, LinkedHashSet[String]] =
    HashMap[Long, LinkedHashSet[String]]()
  // sqlId to sql info
  val sqlIdToInfo = new HashMap[Long, SQLExecutionInfoClass]()
  val sqlIdToStages = new HashMap[Long, ArrayBuffer[Int]]()
  var sqlPlanMetricsAdaptive: ArrayBuffer[SQLPlanMetricsCase] = ArrayBuffer[SQLPlanMetricsCase]()

  // accum id to task stage accum info
  lazy val accumManager: AccumManager = new AccumManager()

  lazy val stageManager: StageModelManager = new StageModelManager()
  // Container that manages TaskIno including SparkMetrics.
  // A task is added during a TaskEnd eventLog
  lazy val taskManager: TaskModelManager = new TaskModelManager()

  // The sum over executor core seconds.
  // The duration of an executor is the upTime which is the difference between removalTime and
  // addTime.
  // Caches the total core seconds to be used between both Profiler and Qualification output.
  lazy val totalCoreSeconds: Long = calculateTotalCoreSec()

  var driverAccumMap: HashMap[Long, ArrayBuffer[DriverAccumCase]] =
    HashMap[Long, ArrayBuffer[DriverAccumCase]]()

  var sparkRapidsBuildInfo: SparkRapidsBuildInfoEvent = SparkRapidsBuildInfoEvent(immutable.Map(),
    immutable.Map(), immutable.Map(), immutable.Map())

  // Spark Connect metadata (Spark 3.5+). Empty for non-Connect event logs.
  val connectSessions: HashMap[String, ConnectSessionInfo] = HashMap.empty
  val connectOperations: HashMap[String, ConnectOperationInfo] = HashMap.empty
  // jobTag -> operationId index for correlation with SQL executions and jobs.
  val jobTagToConnectOpId: HashMap[String, String] = HashMap.empty
  // operationId -> sqlIDs discovered via SparkListenerSQLExecutionStart.jobTags.
  val operationIdToSqlIds: HashMap[String, HashSet[Long]] = HashMap.empty
  // operationId -> jobIDs discovered via SparkListenerJobStart.properties["spark.job.tags"].
  val operationIdToJobIds: HashMap[String, HashSet[Int]] = HashMap.empty
  def isConnectMode: Boolean = connectSessions.nonEmpty || connectOperations.nonEmpty

  def sqlPlans: immutable.Map[Long, SparkPlanInfo] = sqlManager.getPlanInfos

  def hasSqlCacheEvidence: Boolean = sqlManager.hasSqlCacheEvidence

  def getStageIDsFromAccumIds(accumIds: Seq[Long]): Set[Int] = {
    accumIds.flatMap(accumManager.getAccStageIds).toSet
  }

  // Returns the String value of the eventlog or empty if it is not defined. Note that the eventlog
  // won't be defined for running applications
  def getEventLogPath: String = {
    eventLogInfo.map(_.eventLog).getOrElse(new Path(StringUtils.UNKNOWN_EXTRACT)).toString
  }

  // Update the endTime of the application and calculate the duration.
  // This is called while handling ApplicationEnd event
  def updateEndTime(newEndTime: Long): Unit = {
    appMetaData.foreach(_.setEndTime(newEndTime))
  }

  // Returns a boolean flag to indicate whether the endTime was estimated.
  def isAppDurationEstimated: Boolean = {
    appMetaData.map(_.isDurationEstimated).getOrElse(false)
  }

  // Returns the AppName
  def getAppName: String = {
    appMetaData.map(_.appName).getOrElse("")
  }

  // Returns optional endTime in ms.
  def getAppEndTime: Option[Long] = {
    appMetaData.flatMap(_.endTime)
  }

  // Returns optional wallClock duration of the Application
  def getAppDuration: Option[Long] = {
    appMetaData.flatMap(_.duration)
  }

  def getWriteOperationRecords(): Iterable[WriteOperationRecord] = {
    sqlManager.getWriteOperationRecords()
  }

  def getWriteDataFormats(): Set[String] = {
    sqlManager.getWriteFormats()
  }

  def getPrimarySQLPlanInfo(): immutable.Map[Long, SparkPlanInfoTruncated] = {
    sqlManager.getTruncatedPrimarySQLPlanInfo
  }

  // Returns a boolean true/false. This is used to check whether processing an eventlog was
  // successful.
  def isAppMetaDefined: Boolean = appMetaData.isDefined

  /**
   * Sets an estimated endTime to the application based on the function passed as an argument.
   * First it checks if the endTime is already defined or not.
   * This method is a temporary refactor because both QualAppInfo and ProfAppInfo have different
   * ways of estimating the endTime.
   *
   * @param callBack function to estimate the endTime
   */
  def estimateAppEndTime(callBack: () => Option[Long]): Unit = {
    if (getAppEndTime.isEmpty) {
      val estimatedResult = callBack()
      estimatedResult.foreach(eT => appMetaData.foreach(_.setEndTime(eT, estimated = true)))
    }
  }

  def guestimateAppEndTimeCB(): () => Option[Long] = {
    () => None
  }

  // time in ms
  def calculateAppDuration(): Option[Long] = {
    if (appMetaData.isDefined) {
      val appMeta = appMetaData.get
      val startTime = appMeta.startTime
      if (startTime > 0) {
        estimateAppEndTime(guestimateAppEndTimeCB())
      }
      getAppDuration
    } else {
      None
    }
  }

  def getOrCreateExecutor(executorId: String, addTime: Long): ExecutorInfoClass = {
    executorIdToInfo.getOrElseUpdate(executorId, {
      new ExecutorInfoClass(executorId, addTime)
    })
  }

  // this is to keep track of the high water mark for number of executors
  // active at anyone instant in time
  def updateMaxExecutors(): Unit = {
    val numActiveExecutors = executorIdToInfo.values.filter(_.isActive).size
    if (numActiveExecutors > maxNumExecutorsRunning) {
      maxNumExecutorsRunning = numActiveExecutors
    }
  }

  // this is to keep track of the high water mark for number of nodes
  // active at anyone instant in time
  def updateMaxNodes(): Unit = {
    // make this a set to make it dedup nodes
    val numActiveNodes = executorIdToInfo.values.filter(_.isActive).map(_.host).toSet.size
    if (numActiveNodes > maxNumNodesRunning) {
      maxNumNodesRunning = numActiveNodes
    }
  }

  def getOrCreateStage(info: StageInfo): StageModel = {
    sqlManager.collectObservedRDDScopes(info.rddInfos)
    val stage = stageManager.addStageInfo(info)
    stage
  }

  def getAllStagesForJobsInSqlQuery(sqlID: Long): Seq[Int] = {
    val jobsIdsInSQLQuery = jobIdToSqlID.filter { case (_, sqlIdForJob) =>
      sqlIdForJob == sqlID
    }.keys.toSeq
    jobsIdsInSQLQuery.flatMap { jId =>
      jobIdToInfo.get(jId).map(_.stageIds)
    }.flatten
  }

  def cleanupAccumId(accId: Long): Unit = {
    accumManager.removeAccumInfo(accId)
    driverAccumMap.remove(accId)
  }

  def cleanupStages(stageIds: Set[Int]): Unit = {
    // stageIdToInfo can have multiple stage attempts, remove all of them
    stageManager.removeStages(stageIds)
  }

  def cleanupSQL(sqlID: Long): Unit = {
    sqlIDToDataSetOrRDDCase.remove(sqlID)
    sqlIDtoProblematic.remove(sqlID)
    sqlIdToInfo.remove(sqlID)
    sqlManager.remove(sqlID)
    val dsToRemove = dataSourceInfo.filter(_.sqlID == sqlID)
    dsToRemove.foreach(dataSourceInfo -= _)

    val jobsInSql = jobIdToSqlID.filter { case (_, sqlIdForJob) =>
      sqlIdForJob == sqlID
    }.keys
    jobsInSql.foreach { jobId =>
      // must call cleanupStage first
      // clean when no other jobs need that stage
      // not sure about races here but lets check the jobs and assume we can clean
      // when none of them reference this stage
      val stagesNotInOtherJobs = jobIdToInfo.get(jobId).map { jInfo =>
        val stagesInJobToRemove = jInfo.stageIds.toSet
        // check to make sure no other jobs reference the same stage
        val allOtherJobs = jobIdToInfo - jobId
        val allOtherStageIds = allOtherJobs.values.flatMap(_.stageIds).toSet
        stagesInJobToRemove.filter(!allOtherStageIds.contains(_))
      }
      stagesNotInOtherJobs.foreach(cleanupStages)
      jobIdToSqlID.remove(jobId)
      jobIdToInfo.remove(jobId)
    }
  }

  def processEvent(event: SparkListenerEvent): Boolean

  private def openEventLogInternal(log: Path, fs: FileSystem): InputStream = {
    EventLogFileWriter.codecName(log) match {
      case c if c.isDefined && c.get.equals("gz") =>
        val in = fs.open(log)
        try {
          new GZIPInputStream(in)
        } catch {
          case e: Throwable =>
            in.close()
            throw e
        }
      case _ => EventLogFileReader.openEventLog(log, fs)
    }
  }

  /**
   * Internal function to process all the events
   */
  private def processEventsInternal(): Unit = {
    eventLogInfo match {
      case Some(eventLog) =>
        val eventLogPath = eventLog.eventLog
        logInfo("Start Parsing Event Log: " + eventLogPath.toString)

        // at this point all paths should be valid event logs or event log dirs
        val hconf = hadoopConf.getOrElse(RapidsToolsConfUtil.newHadoopConf)
        val fs = eventLogPath.getFileSystem(hconf)
        var totalNumEvents = 0L
        val readerOpt = eventLog match {
          case _: DatabricksEventLog =>
            Some(new DatabricksRollingEventLogFilesFileReader(fs, eventLogPath))
          case _ => EventLogFileReader(fs, eventLogPath)
        }

        if (readerOpt.isDefined) {
          val reader = readerOpt.get
          val runtimeGetFromJsonMethod = EventUtils.getEventFromJsonMethod
          reader.listEventLogFiles.foreach { file =>
            Utils.tryWithResource(openEventLogInternal(file.getPath, fs)) { in =>
              UTF8Source.fromInputStream(in).getLines().filter(acceptLine).find { line =>
                // Using find as foreach with conditional to exit early if we are done.
                // Do NOT use a while loop as it is much much slower.
                totalNumEvents += 1
                runtimeGetFromJsonMethod.apply(line) match {
                  case Some(e) => processEvent(e)
                  case None => false
                }
              }
            }
          }
        } else {
          logError(s"Error getting reader for ${eventLogPath.getName}")
        }
        val totalLines = getTotalLines
        val processedLines = getProcessedLinesCount
        val ratio = 100.0 * processedLines / totalLines
        logInfo(
          s"Events stats of ${eventLogPath.toString} (Total/Parsed/Skipped/Process-Percentage): " +
            f"($totalLines%d/$processedLines%d/${getSkippedLinesCount}%d/$ratio%2.2f)")
      case None => logInfo("Streaming events to application")
    }
  }

  private val UDFRegex = ".*UDF.*"

  private val potentialIssuesRegexMap = Map(
    UDFRegex -> "UDF",
    ".*current_timestamp\\(.*\\).*" -> "TIMEZONE current_timestamp()",
    ".*to_timestamp\\(.*\\).*" -> "TIMEZONE to_timestamp()",
    ".*hour\\(.*\\).*" -> "TIMEZONE hour()",
    ".*minute\\(.*\\).*" -> "TIMEZONE minute()",
    ".*second\\(.*\\).*" -> "TIMEZONE second()"
  )

  def findPotentialIssues(desc: String): Set[String] = {
    val potentialIssuesRegexs = potentialIssuesRegexMap
    val issues = potentialIssuesRegexs.iterator.filter { case (k, _) => desc.matches(k) }.toMap
    issues.values.toSet
  }

  /**
   * Builds cluster information based on executor nodes and sets it in the
   * platform so that it can be used later.
   * If executor nodes exist, calculates the number of hosts and total cores,
   * and extracts executor and driver instance types (databricks only)
   */
  def buildClusterInfo(): Unit = {
    // try to figure out number of executors per node based on the executor info
    // Group by host name, find max executors per host
    val execsPerNodeList = executorIdToInfo.values.groupBy(_.host).iterator
      .map { case (k, v) => k -> v.size }.toMap.values
    // if we have different number of execs per node, then we blank it out to indicate
    // not applicable (like when dynamic allocation is on in multi-tenant cluster)
    // Since with dynamic allocation you could end up with more executors on a node then it
    // has slots, just always set numExecsPerNode to -1 when its enabled.
    val dynamicAllocEnabled = Platform.isDynamicAllocationEnabled(sparkProperties)
    val numExecsPerNode = if (!dynamicAllocEnabled && execsPerNodeList.nonEmpty &&
      execsPerNodeList.forall(_ == execsPerNodeList.head)) {
      execsPerNodeList.head
    } else {
      -1
    }
    val execCoreCounts = executorIdToInfo.values.map(_.totalCores)
    if (execCoreCounts.nonEmpty) {
      if (execCoreCounts.toSet.size != 1) {
        logWarning(s"Application $appId: Cluster with variable executor cores detected. " +
          s"Using maximum value.")
      }
      // Create cluster information based on platform type
      platform.foreach(_.configureClusterInfoFromEventLog(execCoreCounts.max,
        numExecsPerNode, maxNumExecutorsRunning, maxNumNodesRunning,
        sparkProperties, systemProperties))
    } else {
      logWarning("Could not determine if any executors were allocated or the number of cores " +
        "used per executor. Can't build existing cluster information!")
    }
  }

  // The ReadSchema metadata is only in the eventlog for DataSource V1 readers
  def checkMetadataForReadSchema(
      sqlPlanInfoGraph: SQLPlanModel): ArrayBuffer[DataSourceRecord] = {
    // check if planInfo has ReadSchema
    val allMetaWithSchema = sqlPlanInfoGraph.planInfo.getPlansWithSchema
    val allNodes = sqlPlanInfoGraph.getToolsPlanGraph.allNodes
    // use linked hashSet to preserve order of insertion and avoid duplicate nodes. This guarantees
    // that each node is matched only once.
    val scanNodes =
      mutable.LinkedHashSet[SparkPlanGraphNode]().++(allNodes.filter(ReadParser.isScanNode))

    val results = ArrayBuffer[DataSourceRecord]()

    allMetaWithSchema.foreach { plan =>
      val meta = plan.metadata
      // TODO: we should find a different way to match the PlanInfo to the GraphNode because
      //       there are some cases where the schema is empty. i.e, schema: struct<>, which implies
      //       that an empty string matches on everything.
      val readSchema = ReadParser.formatSchemaStr(meta.getOrElse("ReadSchema", ""))
      val scanNode = scanNodes.find(node => {
        // Get ReadSchema of each Node and sanitize it for comparison
        val trimmedNode = AppBase.trimSchema(ReadParser.parseReadNode(node).schema)
        if (readSchema.isEmpty && trimmedNode.isEmpty) {
          true
        } else {
          readSchema.contains(trimmedNode)
        }
      })

      // If the ReadSchema is empty or if it is PhotonScan, then we don't need to
      // add it to the dataSourceInfo
      // Processing Photon eventlogs issue: https://github.com/NVIDIA/cudf-spark-tools/issues/251
      scanNode.foreach { sNode =>
        scanNodes.remove(sNode)
        results += DataSourceRecord(
          sqlPlanInfoGraph.id,
          sqlPlanInfoGraph.plan.version,
          sNode.id,
          ReadParser.extractTagFromV1ReadMeta("Format", meta),
          ReadParser.extractTagFromV1ReadMeta("Location", meta),
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_PUSHED_FILTERS, meta),
          readSchema,
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_DATA_FILTERS, meta),
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_PARTITION_FILTERS, meta)
        )
      }
    }
    // "scan hive" has no "ReadSchema" defined. So, we need to look explicitly for nodes
    // that are scan hive and add them one by one to the dataSource
    if (isHiveEnabled) { // only scan for hive when the CatalogImplementation is using hive
      // TODO: this needs to be refactored to follow the same fix in the generic case.
      //       We should not be checking the hive scans explicitly in the caller.
      val allPlanWithHiveScan = sqlPlanInfoGraph.planInfo.getPlansWithHiveScan
      allPlanWithHiveScan.foreach { hiveReadPlan =>
        val sqlGraph = ToolsPlanGraph(hiveReadPlan)
        val hiveScanNode = sqlGraph.allNodes.head
        val scanHiveMeta = HiveParseHelper.parseReadNode(hiveScanNode)
        results += DataSourceRecord(
          sqlPlanInfoGraph.id,
          sqlPlanInfoGraph.plan.version,
          hiveScanNode.id,
          scanHiveMeta.format,
          scanHiveMeta.location,
          scanHiveMeta.pushedFilters,
          scanHiveMeta.schema,
          scanHiveMeta.dataFilters,
          scanHiveMeta.partitionFilters
        )
      }
    }
    dataSourceInfo ++= results
    results
  }

  // This will find scans for DataSource V2, if the schema is very large it
  // will likely be incomplete and have ... at the end.
  def checkGraphNodeForReads(
      sqlID: Long, node: SparkPlanGraphNode): Option[DataSourceRecord] = {
    if (ReadParser.isDataSourceV2Node(node)) {
      // use batchScan parser if it matches otherwise fall back to generic read parser
      val res = if (BatchScanExecParser.accepts(node, Option(this))) {
        BatchScanExecParser.extractReadMetaData(node, Option(this))
      } else {
        ReadParser.parseReadNode(node)
      }
      val dsCase = DataSourceRecord(
        sqlID,
        sqlManager.getPlanById(sqlID).get.plan.version,
        node.id,
        res.format,
        res.location,
        res.pushedFilters,
        res.schema,
        res.dataFilters,
        res.partitionFilters)
      dataSourceInfo += dsCase
      Some(dsCase)
    } else {
      None
    }
  }

  protected def reportComplexTypes: (Seq[String], Seq[String]) = {
    if (dataSourceInfo.nonEmpty) {
      val schema = dataSourceInfo.map { ds => ds.schema }
      AppBase.parseReadSchemaForNestedTypes(schema)
    } else {
      (Seq(), Seq())
    }
  }

  protected def probNotDataset: HashMap[Long, LinkedHashSet[String]] = {
    sqlIDtoProblematic.filterNot { case (sqlID, _) => sqlIDToDataSetOrRDDCase.contains(sqlID) }
  }

  protected def getPotentialProblemsForDf: Seq[String] = {
    probNotDataset.values.flatten.toSet.toSeq
  }

  /**
   * Registers the attempt ID for the application and updates the tracker map if the attemptId is
   * greater than the existing attemptId.
   */
  def registerAttemptId(): Unit = {
    if (isAppMetaDefined) {
      val currentAttemptId = sparkProperties.getOrElse("spark.app.attempt.id", "1").toInt
      appMetaData.foreach(_.setAttemptId(currentAttemptId))
      AppSubscriber.subscribeAppAttempt(appId, currentAttemptId)
    }
  }

  private def postCompletion(): Unit = {
    registerAttemptId()
    calculateAppDuration()
    validateSparkRuntime()
    buildClusterInfo()
    buildPlanGraphs()
  }

  /**
   * Build the plan graphs for all the SQL Plans if any.
   * @note This should only be called once.
   */
  protected def buildPlanGraphs(): Unit = {
    sqlManager.buildPlanGraph(this)
  }

  /**
   * Wrapper function to process all the events followed by any
   * post completion tasks.
   */
  def processEvents(): Unit = {
    processEventsInternal()
    postCompletion()
    RuntimeInjector.insertMemoryMarker("Post processing events")
  }

  /**
   * Validates if the spark runtime (parsed from event log) is supported by the platform.
   * If the runtime is not supported, an `UnsupportedSparkRuntimeException`
   * is thrown.
   */
  private def validateSparkRuntime(): Unit = {
    val parsedRuntime = getSparkRuntime
    platform.foreach { p =>
      require(p.isRuntimeSupported(parsedRuntime),
        throw UnsupportedSparkRuntimeException(p, parsedRuntime)
      )
    }
  }

  /**
   * Calculates total core seconds which is the sum over executor core seconds. Executor
   * core seconds is computed as executor duration (s) multiplied by num of cores.
   */
  private def calculateTotalCoreSec(): Long = {
    var totalCoreSec: Double = 0
    executorIdToInfo.foreach { case(_, eInfo) =>
      val eStartTime = eInfo.addTime
      var eEndTime = eInfo.removeTime
      if (eEndTime == 0L) {
        getAppEndTime match {
          case Some(appEndTime) =>
            eEndTime = appEndTime
          case None =>
            logInfo("Unable to find either executor or app end time: " +
              "setting executor duration to 0")
            eEndTime = eStartTime
        }
      }
      totalCoreSec += (eEndTime - eStartTime).toDouble / 1000 * eInfo.totalCores
    }
    // round up for edge case when total core seconds is in range [0, 1)
    math.ceil(totalCoreSec).toLong
  }
}

object AppBase {

  def parseReadSchemaForNestedTypes(
      schema: ArrayBuffer[String]): (Seq[String], Seq[String]) = {
    val tempStringBuilder = new StringBuilder()
    val individualSchema: ArrayBuffer[String] = new ArrayBuffer()
    var angleBracketsCount = 0
    var parenthesesCount = 0
    val distinctSchema = schema.distinct.filter(_.nonEmpty).mkString(",")

    // Get the nested types i.e everything between < >
    for (char <- distinctSchema) {
      char match {
        case '<' => angleBracketsCount += 1
        case '>' => angleBracketsCount -= 1
        // If the schema has decimals, Example decimal(6,2) then we have to make sure it has both
        // opening and closing parentheses(unless the string is incomplete due to V2 reader).
        case '(' => parenthesesCount += 1
        case ')' => parenthesesCount -= 1
        case _ =>
      }
      if (angleBracketsCount == 0 && parenthesesCount == 0 && char.equals(',')) {
        individualSchema += tempStringBuilder.toString
        tempStringBuilder.setLength(0)
      } else {
        tempStringBuilder.append(char)
      }
    }
    if (tempStringBuilder.nonEmpty) {
      individualSchema += tempStringBuilder.toString
    }

    // If DataSource V2 is used, then Schema may be incomplete with ... appended at the end.
    // We determine complex types and nested complex types until ...
    val incompleteSchema = individualSchema.filter(x => x.contains("..."))
    val completeSchema = individualSchema.filterNot(x => x.contains("..."))

    // Check if it has types
    val incompleteTypes = incompleteSchema.map { x =>
      if (x.contains("...") && x.contains(":")) {
        val schemaTypes = x.split(":", 2)
        if (schemaTypes.size == 2) {
          val partialSchema = schemaTypes(1).split("\\.\\.\\.")
          if (partialSchema.size == 1) {
            partialSchema(0)
          } else {
            ""
          }
        } else {
          ""
        }
      } else {
        ""
      }
    }
    // Omit columnName and get only schemas
    val completeTypes = completeSchema.map { x =>
      val schemaTypes = x.split(":", 2)
      if (schemaTypes.size == 2) {
        schemaTypes(1)
      } else {
        ""
      }
    }
    val schemaTypes = (completeTypes ++ incompleteTypes).toSeq

    // Filter only complex types.
    // Example: array<string>, array<struct<string, string>>
    val complexTypes = schemaTypes.filter(x =>
      x.startsWith("array<") || x.startsWith("map<") || x.startsWith("struct<"))

    // Determine nested complex types from complex types
    // Example: array<struct<string, string>> is nested complex type.
    val nestedComplexTypes = complexTypes.filter(complexType => {
      val startIndex = complexType.indexOf('<')
      val closedBracket = complexType.lastIndexOf('>')
      // If String is incomplete due to dsv2, then '>' may not be present. In that case traverse
      // until length of the incomplete string
      val lastIndex = if (closedBracket == -1) {
        complexType.length - 1
      } else {
        closedBracket
      }
      val string = complexType.substring(startIndex, lastIndex + 1)
      string.contains("array<") || string.contains("struct<") || string.contains("map<")
    })

    (complexTypes.filter(_.nonEmpty), nestedComplexTypes.filter(_.nonEmpty))
  }

  def trimSchema(str: String): String = {
    val index = str.lastIndexOf(",")
    if (index != -1 && str.contains("...")) {
      str.substring(0, index)
    } else {
      str
    }
  }

  def handleException(e: Exception, path: EventLogInfo): FailureApp = {
    val (status, message): (String, String) = e match {
      case incorrectStatusEx: IncorrectAppStatusException =>
        (StringUtils.UNKNOWN_EXTRACT, incorrectStatusEx.getMessage)
      case skippedEx: AppEventlogProcessException =>
        ("skipped", skippedEx.getMessage)
      case _: com.fasterxml.jackson.core.JsonParseException =>
        (StringUtils.UNKNOWN_EXTRACT, s"Error parsing JSON: " +
          s"${path.eventLog.toString}. ${e.getMessage}")
      case e: IllegalArgumentException =>
        (StringUtils.UNKNOWN_EXTRACT, s"Error parsing file: " +
          s"${path.eventLog.toString}. ${e.getMessage}")
      case ue: Exception =>
        // catch all exceptions and skip that file
        (StringUtils.UNKNOWN_EXTRACT, s"Got unexpected exception processing file:" +
          s"${path.eventLog.toString}. ${ue.getMessage} ")
    }

    FailureApp(status, s"${e.getClass.getSimpleName}: $message")
  }

  def toSuccessAppResult(app: AppBase): SuccessAppResult = {
    SuccessAppResult(
      path = app.getEventLogPath,
      appId = app.appId,
      attemptId = app.attemptId,
      appName = app.getAppName,
      message = s"appDuration(ms): ${app.calculateAppDuration().getOrElse(0)}")
  }
}
