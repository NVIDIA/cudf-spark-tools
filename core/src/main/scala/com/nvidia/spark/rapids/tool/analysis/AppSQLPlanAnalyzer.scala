/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.analysis

import scala.collection.mutable.{AbstractSet, ArrayBuffer, HashMap, LinkedHashSet}

import com.nvidia.spark.rapids.tool.analysis.util.IOAccumDiagnosticMetrics._
import com.nvidia.spark.rapids.tool.analysis.util.StageAccumDiagnosticMetrics._
import com.nvidia.spark.rapids.tool.profiling.{AccumProfileResults, IODiagnosticResult, ShuffleStageInputAnalysis, SQLAccumProfileResults, SQLMetricInfoCase, SQLStageInfoProfileResult, UnsupportedSQLPlan, WholeStageCodeGenResults}

import org.apache.spark.sql.rapids.tool.{AppBase, RDDCheckHelper}
import org.apache.spark.sql.rapids.tool.plangraph.{SparkPlanGraphCluster, SparkPlanGraphNode, ToolsPlanGraph}
import org.apache.spark.sql.rapids.tool.store.{DataSourceRecord, SQLPlanModel}

/**
 * This class processes SQL plan to build some information such as: metrics, wholeStage nodes, and
 * connecting operators to nodes. The implementation used to be directly under Profiler's
 * ApplicationInfo class. Moving the logic and the data structure to this new class is part of
 * refactor to have a SQLPlan processor that can produce the same analysis for both the Prof/Qual
 * tools.
 * Calling processSQLPlanMetrics() has a side effect on the app object:
 * 1- it updates dataSourceInfo with V2 and V1 data sources
 * 2- it updates sqlIDtoProblematic the map between SQL ID and potential problems
 * 3- it updates sqlIdToInfo.DsOrRdd as boolean to indicate whether a sql is an RDD/DS or not
 * TODO: this class should extend the trait SparkSQLPlanInfoVisitor[T]
 * @param app the Application info objects that contains the SQL plans to be processed
 */
class AppSQLPlanAnalyzer(app: AppBase)
    extends AppAnalysisBase(app) {
  var wholeStage: ArrayBuffer[WholeStageCodeGenResults] = ArrayBuffer[WholeStageCodeGenResults]()
  // A list of UnsupportedSQLPlan that contains the SQL ID, node ID, node name.
  // TODO: for now, unsupportedSQLPlan is kept here for now to match the legacy Profiler's
  //      output but we need to revisit this in the future to see if we can move it to a
  //      different place or fix any inconsistent issues between this implementation and
  //      SQLPlanParser.
  var unsupportedSQLPlan: ArrayBuffer[UnsupportedSQLPlan] = ArrayBuffer[UnsupportedSQLPlan]()
  var allSQLMetrics: ArrayBuffer[SQLMetricInfoCase] = ArrayBuffer[SQLMetricInfoCase]()
  // A map between stage ID and a set of node names
  val stageToNodeNames: HashMap[Long, Seq[String]] = HashMap.empty[Long, Seq[String]]
  // A mapping from stage ID to diagnostic metrics results.
  // Each stage ID maps to another HashMap, where:
  //   - The key is the diagnostic metric name (String).
  //   - The value is an AccumProfileResults object containing the diagnostic data for that metric.
  val stageToDiagnosticMetrics: HashMap[Long, HashMap[String, AccumProfileResults]] =
    HashMap.empty[Long, HashMap[String, AccumProfileResults]]

  // A mapping from a unique combination of SQL execution identifiers to a list of IO diagnostic
  // metrics results.
  // The key is a tuple consisting of:
  //   - sqlID (Long): The unique identifier for the SQL query.
  //   - nodeID (Long): The unique identifier for the node.
  // The value is an ArrayBuffer of SQLAccumProfileResults objects, storing the IO diagnostic
  // metrics for the given key.
  val IODiagnosticMetricsMap: HashMap[(Long, Long), ArrayBuffer[SQLAccumProfileResults]] =
    HashMap.empty[(Long, Long), ArrayBuffer[SQLAccumProfileResults]]

  /**
   * Raw consumer-stage shuffle input inventory used by the AutoTuner's downward shuffle-partition
   * pass. It is computed lazily and cached so that both tools reuse this analyzer's plan graphs
   * instead of traversing the SQL plans a second time.
   */
  lazy val shuffleStageInputAnalysis: ShuffleStageInputAnalysis = ShuffleStageInputAnalyzer(app)

  /**
   * Updates the stageToDiagnosticMetrics mapping with the provided AccumProfileResults.
   * @param accum AccumProfileResults instance containing diagnostic metrics to be added
   *              to stageToDiagnosticMetrics mapping.
   */
  private def updateStageDiagnosticMetrics(accum: AccumProfileResults): Unit = {
    // Initialize an empty mapping for the stage if it doesn't already exist
    if (!stageToDiagnosticMetrics.contains(accum.stageId)) {
      stageToDiagnosticMetrics(accum.stageId) = HashMap.empty[String, AccumProfileResults]
    }

    stageToDiagnosticMetrics(accum.stageId)(accum.accMetaRef.getName()) = accum
  }

  /**
   * Updates the IODiagnosticMetricsMap with the provided SQLAccumProfileResults.
   * @param accum SQLAccumProfileResults instance containing IO diagnostics metrics
   *              to be added to IODiagnosticMetricsMap.
   */
  private def updateIODiagnosticMetricsMap(accum: SQLAccumProfileResults): Unit = {
    val key = (accum.sqlID, accum.nodeID)

    // Initialize an entry if the key does not exist
    if (!IODiagnosticMetricsMap.contains(key)) {
      IODiagnosticMetricsMap(key) = ArrayBuffer[SQLAccumProfileResults]()
    }

    IODiagnosticMetricsMap(key) += accum
  }

  /**
   * helper to get teh tools planGraph object by SqlId
   * @param sqlId the SQLPlan ids
   * @return optional ToolsPlanGraph if the plan exists.
   */
  private def getToolsPlanGraphById(sqlId: Long): Option[ToolsPlanGraph] = {
    app.sqlManager.applyToPlanModel(sqlId) { planModel =>
      planModel.getToolsPlanGraph
    }
  }
  // Return the stages for a given node.
  private def getNodeStages(sqlId: Long, nodeId: Long): Option[Set[Int]] = {
    getToolsPlanGraphById(sqlId).map(_.getNodeStageRawAssignment(nodeId))
  }
  // Return the nodes that belong to a specific stage.
  private def getStageNodes(sqlId: Long, stageId: Int): Option[collection.Set[Long]] = {
    getToolsPlanGraphById(sqlId).map(_.getStageNodesByRawAssignment(stageId))
  }

  /**
   * Update the potential problems in the app object. This is a common function that is used by
   * Both Qual/Prof analysis.
   * For Qual apps, the app.sqlIDtoProblematic won't be set because it is done later during the
   * aggregation phase.
   *
   * @param sqlId             the SQL ID being analyzed
   * @param potentialProblems a set of strings that represent the potential problems found in the
   *                          SQL plan.
   */
  private def updateAppPotentialProblems(sqlId: Long,
      potentialProblems: AbstractSet[String]): Unit = {
    // Append problematic issues to the global variable for that SqlID
    val existingIssues = app.sqlIDtoProblematic.getOrElse(sqlId, LinkedHashSet[String]())
    app.sqlIDtoProblematic(sqlId) = existingIssues ++ potentialProblems
  }

  // A visitor context to hold the state of the SQL plan visitor.
  // The fields are updated and modified by the visitNode function.
  // sqlIsDsOrRDD is initialized to False, and it is set only once to True when a node is detected
  // as RDD or DS.
  protected case class SQLPlanVisitorContext(
      sqlPIGEntry: SQLPlanModel,
      sqlDataSources: ArrayBuffer[DataSourceRecord] = ArrayBuffer[DataSourceRecord](),
      potentialProblems: LinkedHashSet[String] = LinkedHashSet[String](),
      var sqlIsDsOrRDD: Boolean = false)

  /**
   * The method is called for each node in the SparkGraph plan.
   * It visits the node to extract the following information
   * 1- the metrics;
   * 2- potential problems.
   * 3- data sources
   *
   * It updates the following fields defined in AppSQLPlanAnalyzer:
   * 1- allSQLMetrics: a list of SQLMetricInfoCase
   * 2- wholeStage: a list of WholeStageCodeGenResults
   * 3- unsupportedSQLPlan: a list of UnsupportedSQLPlan that contains the SQL ID, node ID,
   * node name.
   * TODO: Consider handling the construction of this list in a different way for the
   * Qualification
   * 4- sqlPlanNodeIdToStageIds: A map between (SQL ID, Node ID) and the set of stage IDs
   *
   * It has the following effect on the visitor object:
   * 1- It updates the sqlIsDsOrRDD argument to True when the visited node is an RDD or Dataset.
   * 2- If the SQLID is an RDD, the potentialProblems is cleared because once SQL is marked as RDD,
   * all the other problems are ignored. Note that we need to set that flag only once to True
   * for the given SQLID.
   * 3- It appends the current node's potential problems to the SQLID problems only if the SQL is
   * visitor.sqlIsDsOrRDD is False. Otherwise, it is kind of redundant to keep checking for
   * potential problems for every node when they get to be ignored.
   *
   * It has the following effect on the app object:
   * 1- it updates dataSourceInfo with V2 and V1 data sources
   * 2- it updates sqlIDtoProblematic the map between SQL ID and potential problems
   *
   * @param visitor the visitor context defined per SQLPlan
   * @param node    the node being currently visited.
   */
  protected def visitNode(visitor: SQLPlanVisitorContext,
      node: SparkPlanGraphNode): Unit = {
    node match {
      case cluster: SparkPlanGraphCluster =>
        val ch = cluster.nodes
        ch.foreach { c =>
          wholeStage += WholeStageCodeGenResults(
            visitor.sqlPIGEntry.id, node.id, node.name, c.name, c.id)
        }
      case _ =>
    }
    // get V2 dataSources for that node
    val nodeV2Reads = app.checkGraphNodeForReads(visitor.sqlPIGEntry.id, node)
    if (nodeV2Reads.isDefined) {
      visitor.sqlDataSources += nodeV2Reads.get
    }

    val nodeIsDsOrRDD = RDDCheckHelper.isDatasetOrRDDPlan(node.name, node.desc).isRDD
    if (nodeIsDsOrRDD) {
      // we want to report every node that is an RDD
      val thisPlan = UnsupportedSQLPlan(visitor.sqlPIGEntry.id, node.id, node.name, node.desc,
        "Contains Dataset or RDD")
      unsupportedSQLPlan += thisPlan
      // If one node is RDD, the Sql should be set too
      if (!visitor.sqlIsDsOrRDD) { // We need to set the flag only once for the given sqlID
        visitor.sqlIsDsOrRDD = true
        app.sqlIdToInfo.get(visitor.sqlPIGEntry.id).foreach { sql =>
          sql.setDsOrRdd(visitor.sqlIsDsOrRDD)
          app.sqlIDToDataSetOrRDDCase += visitor.sqlPIGEntry.id
          // Clear the potential problems since it is an RDD to free memory
          visitor.potentialProblems.clear()
        }
      }
    }
    if (!visitor.sqlIsDsOrRDD) {
      // Append current node's potential problems to the Sql problems only if the SQL is not an
      // RDD. This is an optimization since the potentialProblems won't be used any more.
      visitor.potentialProblems ++= app.findPotentialIssues(node.desc)
    }
    // Then process SQL plan metric type
    val stages =
      getNodeStages(visitor.sqlPIGEntry.id, node.id).getOrElse(ToolsPlanGraph.EMPTY_CLUSTERS)
    for (metric <- node.metrics) {
      val allMetric = SQLMetricInfoCase(visitor.sqlPIGEntry.id, metric.name,
        metric.accumulatorId, metric.metricType, node.id, node.name, node.desc, stages)
      allSQLMetrics += allMetric
    }
  }

  /**
   * Function to process SQL Plan Metrics after all events are processed
   */
  def processSQLPlanMetrics(): Unit = {
    // Iterate on all the SqlPlans
    for (sqlPIGEntry <- app.sqlManager.sqlPlans.values) {
      // store all dataSources of the given SQL in a variable so that we won't have to iterate
      // through the entire list
      // get V1 dataSources for that SQLId
      val visitorContext = SQLPlanVisitorContext(sqlPIGEntry,
        app.checkMetadataForReadSchema(sqlPIGEntry))
      for (node <- sqlPIGEntry.getToolsPlanGraph.allNodes) {
        visitNode(visitorContext, node)
      }
      if (visitorContext.sqlDataSources.nonEmpty) {
        val sqlNestedComplexTypes =
          AppBase.parseReadSchemaForNestedTypes(
            visitorContext.sqlDataSources.map { ds => ds.schema })
        // Append problematic issues to the global variable for that SqlID
        if (sqlNestedComplexTypes._2.nonEmpty) {
          visitorContext.potentialProblems += "NESTED COMPLEX TYPE"
        }
      }
      // Finally, update the potential problems in the app object
      // Note that the implementation depends on teh type of the AppBase
      if (visitorContext.potentialProblems.nonEmpty) {
        updateAppPotentialProblems(sqlPIGEntry.id, visitorContext.potentialProblems)
      }
    }
  }

  def aggregateSQLStageInfo: Seq[SQLStageInfoProfileResult] = {
    val jobsWithSQL = app.jobIdToInfo.filter { case (_, j) =>
      j.sqlID.nonEmpty
    }
    jobsWithSQL.flatMap { case (jobId, j) =>
      // for each stage in a given job:
      // 1. get the stage model
      // 2. get all nodes in that stage
      // 3. build the names of the nodes
      app.stageManager.getStagesByIds(j.stageIds).map { stageModel =>
        // Get all node Ids in the stage and convert them to names.
        // Create an empty sequence if the stage has no matching nodes.
        val nodeNames = getStageNodes(j.sqlID.get, stageModel.getId) match {
          case Some(nIds) =>
            getToolsPlanGraphById(j.sqlID.get).map { g =>
              g.allNodes
                .filter(n => nIds.contains(n.id))
                .map(n => s"${n.name}(${n.id.toString})")
                .toVector
            }.getOrElse(Vector.empty)
          case None =>
            Vector.empty
        }
        // Only update stageToNodeNames if:
        // 1. diagnostics is enabled. This will save memory heap for large eventlogs.
        // 2. it's not already set to avoid overwriting.
        if (isDiagnosticViewsEnabled && !stageToNodeNames.contains(stageModel.getId)) {
          stageToNodeNames(stageModel.getId) = nodeNames
        }
        SQLStageInfoProfileResult(j.sqlID.get, jobId, stageModel.getId,
          stageModel.getAttemptId, stageModel.duration, nodeNames)
      }
    }.toSeq
  }

  def generateSQLAccums(): Seq[SQLAccumProfileResults] = {
    allSQLMetrics.flatMap { metric =>
      val accumTaskStats = app.accumManager.calculateAccStats(metric.accumulatorId)
      // local mode driver gets updates
      val driverAccumsOpt = app.driverAccumMap.get(metric.accumulatorId)
      val driverMax = driverAccumsOpt match {
        case Some(accums) =>
          StatisticsMetrics.createOptionalFromArr(
            accums.iterator.collect { case a if a.sqlID == metric.sqlID => a.value }.toArray)
        case _ => None
      }

      if (accumTaskStats.isDefined || driverMax.isDefined) {
        val taskInfo = accumTaskStats.getOrElse(StatisticsMetrics.ZERO_RECORD)
        val driverInfo = driverMax.getOrElse(StatisticsMetrics.ZERO_RECORD)

        val max = Math.max(taskInfo.max, driverInfo.max)
        val min = Math.max(taskInfo.min, driverInfo.min)
        val med = Math.max(taskInfo.med, driverInfo.med)
        val total = Math.max(taskInfo.total, driverInfo.total)

        val sqlAccumProfileResult = SQLAccumProfileResults(metric.sqlID,
          metric.nodeID, metric.nodeName, metric.accumulatorId, metric.name,
          min, med, max, total, metric.metricType, metric.stageIds)

        if (isDiagnosticViewsEnabled && isIODiagnosticMetricName(metric.name)) {
          updateIODiagnosticMetricsMap(sqlAccumProfileResult)
        }

        Some(sqlAccumProfileResult)
      } else {
        None
      }
    }.toSeq
  }

  /**
   * Generates IO-related diagnostic metrics for the SQL plan. Metrics include:
   * - Output rows
   * - Scan time
   * - Output batches
   * - Buffer time
   * - Shuffle write time
   * - Fetch wait time
   * - GPU decode time
   *
   * This method processes accumulator information for each SQL stage and node and
   * computes statistical results (min, median, max, sum) for IO-related metrics.
   *
   * @return A sequence of `IODiagnosticResult` objects one per SQL stage and node.
   */
  def generateIODiagnosticAccums(): Seq[IODiagnosticResult] = {
    // Transform the diagnostic metrics map into a sequence of results
    IODiagnosticMetricsMap.flatMap { case ((sqlId, nodeId), sqlAccums) =>
      // Process each stage ID and compute diagnostic results
      // TODO: currently if stage IDs is empty, the result is skipped
      val stageIds = sqlAccums.head.stageIds
      stageIds.flatMap { stageId =>
        val nodeName = sqlAccums.head.nodeName

        // Initialize a map to store statistics for each IO metric
        val metricNameToStatistics = HashMap.empty[String, StatisticsMetrics].
          withDefaultValue(StatisticsMetrics.ZERO_RECORD)

        // Process each accumulator for the current SQL stage
        sqlAccums.foreach { sqlAccum =>
          // TODO: check if accumulator ID is in driverAccumMap, currently skipped
          val accumInfoOpt = app.accumManager.accumInfoMap.get(sqlAccum.accumulatorId)

          val metricStats: Option[StatisticsMetrics] = accumInfoOpt.flatMap { accumInfo =>
            if (!accumInfo.containsStage(stageId)) {
              None
            } else if (stageIds.size == 1) {
              // Skip computing statistics when there is only one stage
              Some(StatisticsMetrics(
                min = sqlAccum.min,
                med = sqlAccum.median,
                max = sqlAccum.max,
                count = 0,
                total = sqlAccum.total))
            } else {
              // Retrieve task updates which correspond to the current stage
              accumInfo.calculateAccStatsForStage(stageId)
            }
          }

          // Compute the metric's statistics and store the results if available
          metricStats.map { stat =>
            val metricKey = normalizeToIODiagnosticMetricKey(sqlAccum.name)
            metricNameToStatistics(metricKey) = stat
          }
        }

        if (metricNameToStatistics.isEmpty) {
          // No IO metric statistics were computed for this stage
          None
        } else {
          Some(IODiagnosticResult(
            app.getAppName,
            app.appId,
            sqlId,
            stageId,
            app.stageManager.getDurationById(stageId),
            nodeId,
            nodeName,
            metricNameToStatistics(OUTPUT_ROWS_METRIC_KEY),
            metricNameToStatistics(SCAN_TIME_METRIC_KEY),
            metricNameToStatistics(OUTPUT_BATCHES_METRIC_KEY),
            metricNameToStatistics(BUFFER_TIME_METRIC_KEY),
            metricNameToStatistics(SHUFFLE_WRITE_TIME_METRIC_KEY),
            metricNameToStatistics(FETCH_WAIT_TIME_METRIC_KEY),
            metricNameToStatistics(GPU_DECODE_TIME_METRIC_KEY)))
        }
      }
    }.toSeq
  }

  /**
   * Generate the stage level metrics for the SQL plan including GPU metrics if applicable.
   * Along with Spark defined metrics, below is the list of GPU metrics that are collected if they
   * are present in the eventlog:
   * gpuSemaphoreWait, gpuRetryCount, gpuSplitAndRetryCount, gpuRetryBlockTime,
   * gpuRetryComputationTime, gpuSpillToHostTime, gpuSpillToDiskTime, gpuReadSpillFromHostTime,
   * gpuReadSpillFromDiskTime
   *
   * @return a sequence of AccumProfileResults
   */
  def generateStageLevelAccums(): Seq[AccumProfileResults] = {
    app.accumManager.accumInfoMap.flatMap { accumMapEntry =>
      val accumInfo = accumMapEntry._2
      accumInfo.getStageIds.flatMap( stageId => {
        // Get the task updates that belong to that stage
        accumInfo.calculateAccStatsForStage(stageId) match {
          case Some(stat) =>
            // Reuse AccumProfileResults to avoid generating allocating new objects
            val accumProfileResults = AccumProfileResults(
              stageId,
              accumInfo.infoRef,
              min = stat.min,
              median = stat.med,
              max = stat.max,
              total = stat.total)
            if (isDiagnosticViewsEnabled && isDiagnosticMetrics(accumInfo.infoRef.name.value)) {
              updateStageDiagnosticMetrics(accumProfileResults)
            }
            Some(accumProfileResults)
          case _ => None
        }
      })
    }.toSeq
  }
}

object AppSQLPlanAnalyzer {
  def apply(app: AppBase): AppSQLPlanAnalyzer = {
    val sqlAnalyzer = new AppSQLPlanAnalyzer(app)
    sqlAnalyzer.processSQLPlanMetrics()
    sqlAnalyzer
  }
}
