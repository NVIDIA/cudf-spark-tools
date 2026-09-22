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

package com.nvidia.spark.rapids.tool.qualification.checkers

import scala.collection.mutable

import org.scalatest.Assertions.fail


/** Validates structural invariants in every generated per-application execs.csv file. */
case class QToolExecsInvariantChecker(
    description: String = "generated execs.csv cluster invariants")
    extends QToolOutFileChecker {

  qTableLabel = "execCSVReport"

  override def doFire(qTestCtxt: QToolTestCtxt): Unit = {
    qOutDir = qTestCtxt.outputDirectory
    super.doFire(qTestCtxt)
    // Some qualification tests intentionally produce no per-application exec output.
    actualCSVContainers.values.foreach { csvContainer =>
      csvContainer.verifyHeaders()
      QToolExecsInvariantChecker.verify(csvContainer)
    }
  }

  override def build(): QToolResultCheckerTrait = this
}

object QToolExecsInvariantChecker {
  private val SqlIdColumn = "SQL ID"
  private val ExecNameColumn = "Exec Name"
  private val NodeIdColumn = "SQL Node Id"
  private val StagesColumn = "Exec Stages"
  private val ChildrenColumn = "Exec Children"
  private val ChildrenNodeIdsColumn = "Exec Children Node Ids"

  private type CSVRow = Map[String, String]
  private type NodeKey = (String, String)

  private def splitValues(value: String): Seq[String] = {
    if (value.isEmpty) {
      Seq.empty
    } else {
      value.split(":", -1).toSeq
    }
  }

  private def stageSet(row: CSVRow): Set[String] = {
    splitValues(row(StagesColumn)).filter(_.nonEmpty).toSet
  }

  private def nodeDescription(row: CSVRow): String = {
    s"SQL ${row(SqlIdColumn)} node ${row(NodeIdColumn)} (${row(ExecNameColumn)})"
  }

  private def formatDetails(label: String, violations: Seq[String]): Seq[String] = {
    val summary = s"$label: ${violations.size}"
    summary +: violations.map(detail => s"  - $detail")
  }

  def verify(csvContainer: QToolCSVFileContainer): Unit = {
    verifyRows(csvContainer.csvRows, csvContainer.fileLoc)
  }

  private[checkers] def verifyRows(rows: Seq[Map[String, String]], fileLoc: String): Unit = {
    val rowsByNode = rows.groupBy(row => (row(SqlIdColumn), row(NodeIdColumn)))
    val clusters = rows.filter(_(ExecNameColumn).startsWith("WholeStageCodegen"))
    val childOwners = mutable.Map.empty[NodeKey, mutable.Set[String]]

    val emptyClusters = mutable.ArrayBuffer.empty[String]
    val cardinalityMismatches = mutable.ArrayBuffer.empty[String]
    val repeatedChildIds = mutable.ArrayBuffer.empty[String]
    val unresolvedChildren = mutable.ArrayBuffer.empty[String]
    val stageDisjointRelations = mutable.ArrayBuffer.empty[String]

    clusters.foreach { cluster =>
      val sqlId = cluster(SqlIdColumn)
      val clusterId = cluster(NodeIdColumn)
      val childNames = splitValues(cluster(ChildrenColumn))
      val childIds = splitValues(cluster(ChildrenNodeIdsColumn))

      if (childIds.isEmpty) {
        emptyClusters += nodeDescription(cluster)
      }
      if (childNames.size != childIds.size) {
        cardinalityMismatches +=
          s"${nodeDescription(cluster)} has ${childNames.size} child names and " +
            s"${childIds.size} child IDs"
      }
      childIds.distinct.foreach { childId =>
        val occurrences = childIds.count(_ == childId)
        if (occurrences > 1) {
          repeatedChildIds +=
            s"${nodeDescription(cluster)} repeats child ID $childId $occurrences times"
        }
      }

      childIds.foreach { childId =>
        val childKey = (sqlId, childId)
        childOwners.getOrElseUpdate(childKey, mutable.Set.empty) += clusterId
        rowsByNode.get(childKey) match {
          case Some(Seq(child)) =>
            val clusterStages = stageSet(cluster)
            val childStages = stageSet(child)
            if (clusterStages.nonEmpty && childStages.nonEmpty &&
                clusterStages.intersect(childStages).isEmpty) {
              stageDisjointRelations +=
                s"${nodeDescription(cluster)} has stage-disjoint child " +
                  s"${nodeDescription(child)}; cluster stages " +
                  s"${clusterStages.toSeq.sorted.mkString(":")}, child stages " +
                  childStages.toSeq.sorted.mkString(":")
            }
          case Some(matches) =>
            unresolvedChildren +=
              s"${nodeDescription(cluster)} child $childId resolves to ${matches.size} rows in " +
                s"SQL $sqlId"
          case None =>
            unresolvedChildren +=
              s"${nodeDescription(cluster)} child $childId does not resolve in SQL $sqlId"
        }
      }
    }

    val multipleOwners = childOwners.toSeq.collect {
      case ((sqlId, childId), owners) if owners.size > 1 =>
        s"SQL $sqlId child $childId belongs to clusters ${owners.toSeq.sorted.mkString(":")}"
    }.sorted

    val report = Seq(
      formatDetails("empty cluster child lists", emptyClusters.toSeq),
      formatDetails("child name/ID cardinality mismatches", cardinalityMismatches.toSeq),
      formatDetails("repeated child IDs within one cluster", repeatedChildIds.toSeq),
      formatDetails("unresolved or ambiguous child IDs", unresolvedChildren.toSeq),
      formatDetails("children with multiple cluster owners", multipleOwners),
      formatDetails("stage-disjoint cluster-child relations", stageDisjointRelations.toSeq))

    if (report.exists(_.size > 1)) {
      fail(
        s"execs.csv cluster invariants failed for $fileLoc:\n" +
          report.flatten.mkString("\n"))
    }
  }
}
