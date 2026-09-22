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

import org.scalatest.exceptions.TestFailedException
import org.scalatest.funsuite.AnyFunSuite


class QToolExecsInvariantCheckerSuite extends AnyFunSuite {

  private def execRow(
      execName: String,
      nodeId: String,
      children: String = "",
      childIds: String = "",
      stages: String = "7"): Map[String, String] = {
    Map(
      "SQL ID" -> "1",
      "Exec Name" -> execName,
      "SQL Node Id" -> nodeId,
      "Exec Stages" -> stages,
      "Exec Children" -> children,
      "Exec Children Node Ids" -> childIds)
  }

  test("reject repeated child IDs within one cluster") {
    val rows = Seq(
      execRow("WholeStageCodegen", "1", "Project:Project", "2:2"),
      execRow("Project", "2"))

    val error = intercept[TestFailedException] {
      QToolExecsInvariantChecker.verifyRows(rows, "in-memory execs.csv")
    }

    assert(error.getMessage.contains("repeated child IDs within one cluster: 1"))
    assert(error.getMessage.contains("SQL 1 node 1 (WholeStageCodegen) " +
      "repeats child ID 2 2 times"))
    assert(error.getMessage.contains("children with multiple cluster owners: 0"))
  }

  test("reject a cluster-child relation with disjoint stages") {
    val rows = Seq(
      execRow("WholeStageCodegen", "1", "Project", "2", stages = "7"),
      execRow("Project", "2", stages = "8"))

    val error = intercept[TestFailedException] {
      QToolExecsInvariantChecker.verifyRows(rows, "in-memory execs.csv")
    }

    assert(error.getMessage.contains("stage-disjoint cluster-child relations: 1"))
    assert(error.getMessage.contains("SQL 1 node 1 (WholeStageCodegen) " +
      "has stage-disjoint child SQL 1 node 2 (Project); cluster stages 7, child stages 8"))
  }

  test("accept a cluster-child relation with overlapping unequal stages") {
    val rows = Seq(
      execRow("WholeStageCodegen", "1", "Project", "2", stages = "7:8"),
      execRow("Project", "2", stages = "8:9"))

    QToolExecsInvariantChecker.verifyRows(rows, "in-memory execs.csv")
  }
}
