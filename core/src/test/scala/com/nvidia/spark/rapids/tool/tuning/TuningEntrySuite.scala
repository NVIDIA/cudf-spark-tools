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

package com.nvidia.spark.rapids.tool.tuning

import com.nvidia.spark.rapids.tool.tuning.config.{ConfTypeEnum, TuningEntryDefinition}
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import org.apache.spark.sql.rapids.tool.InvalidMemoryUnitFormatException

class TuningEntrySuite extends AnyFunSuite {
  private def memoryDefinition(
      specialValues: Seq[String]): TuningEntryDefinition = {
    TuningEntryDefinition(
      label = "test.memory",
      confType = ConfTypeEnum.Byte,
      defaultUnit = Some("Byte"),
      specialValues = specialValues)
  }

  test("memory entry preserves a declared special value") {
    val entry = TuningEntry.build(
      "test.memory", Some("-1"), None, Some(memoryDefinition(Seq("-1"))))
    entry.getOriginalValue shouldBe Some("-1")
  }

  test("memory entry normalizes negative one when it is not declared special") {
    val entry = TuningEntry.build(
      "test.memory", Some("-1"), None, Some(memoryDefinition(Seq.empty)))
    entry.getOriginalValue shouldBe Some("-1b")
  }

  test("memory entry rejects an undeclared negative value") {
    an[InvalidMemoryUnitFormatException] should be thrownBy {
      TuningEntry.build(
        "test.memory", Some("-2g"), None, Some(memoryDefinition(Seq("-1"))))
    }
  }

  test("AQE broadcast threshold declares negative one as special") {
    val definition = TuningEntryDefinition
      .getEntryDefinition("spark.sql.adaptive.autoBroadcastJoinThreshold").get
    definition.isSpecialValue("-1") shouldBe true
  }
}
