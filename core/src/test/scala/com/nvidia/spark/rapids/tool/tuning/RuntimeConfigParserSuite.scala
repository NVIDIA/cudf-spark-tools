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

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

class RuntimeConfigParserSuite extends AnyFunSuite {
  import RuntimeConfigParser._

  private val MiB = 1024L * 1024L
  private val GiB = 1024L * MiB

  test("parseBytes follows the Spark integer byte grammar without applying value policy") {
    val validValues = Seq(
      "0" -> 0L,
      "1" -> 1L,
      "1b" -> 1L,
      "1K" -> 1024L,
      "1kb" -> 1024L,
      "1m" -> MiB,
      "1MB" -> MiB,
      " 2G " -> 2L * GiB,
      "1gb" -> GiB,
      "1t" -> 1024L * GiB,
      "1tb" -> 1024L * GiB,
      "1p" -> 1024L * 1024L * GiB,
      "1pb" -> 1024L * 1024L * GiB)

    validValues.foreach { case (raw, expected) =>
      parseBytes(raw) shouldBe Some(ParsedBytes(expected))
    }

    Seq(null, "", "-1g", "1Gi", "1GiB", "1.5g", "garbage", "8192p")
      .foreach(raw => parseBytes(raw) shouldBe None)
  }

  test("byte helpers apply their positive and non-negative value policies") {
    parsePositiveBytes("1m") shouldBe Some(ParsedBytes(MiB))
    parsePositiveBytes("0") shouldBe None
    parsePositiveBytes("-1g") shouldBe None

    parseNonNegativeBytes("1m") shouldBe Some(MiB)
    parseNonNegativeBytes("0") shouldBe Some(0L)
    parseNonNegativeBytes("-1") shouldBe None
  }

  test("remaining parsers retain their distinct runtime grammars") {
    parseFraction(" 0.25 ") shouldBe Some(0.25)
    parseFraction("1.1") shouldBe None
    parseBoolean(" TRUE ") shouldBe Some(true)
    parseBoolean("yes") shouldBe None
    parsePositiveCount(Long.MaxValue.toString) shouldBe Some(Long.MaxValue)
    parseInt(" 0 ") shouldBe Some(0)
    parseInt("-1") shouldBe Some(-1)
    parseInt(Long.MaxValue.toString) shouldBe None
    parsePositiveInt(Long.MaxValue.toString) shouldBe None
    parsePositiveExecutorResourceAmount("1") shouldBe Some(1)
    parsePositiveExecutorResourceAmount(" 1 ") shouldBe None
    parsePositiveDouble(" 0.001 ") shouldBe Some(0.001)
    parsePositiveDouble("NaN") shouldBe None
  }
}
