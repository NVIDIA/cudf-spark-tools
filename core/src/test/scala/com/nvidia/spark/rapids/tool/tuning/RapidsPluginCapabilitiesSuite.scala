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

class RapidsPluginCapabilitiesSuite extends AnyFunSuite {
  private def rightValue[A](result: Either[String, A]): A = result match {
    case Right(value) => value
    case Left(error) => fail(error)
  }

  private def leftMessage[A](result: Either[String, A]): String = result match {
    case Left(error) => error
    case Right(value) => fail(s"expected failure but got $value")
  }

  test("dynamic GPU admission capability starts at plugin 25.06") {
    RapidsPluginCapabilities.supportsAutoConcurrentGpuTasks(Some("25.04.0")) shouldBe false
    RapidsPluginCapabilities.supportsAutoConcurrentGpuTasks(Some("25.06.0")) shouldBe true
    RapidsPluginCapabilities.supportsAutoConcurrentGpuTasks(Some("26.02.0")) shouldBe true
    RapidsPluginCapabilities.supportsAutoConcurrentGpuTasks(None) shouldBe false
  }

  test("maxConcurrentGpuTasks capability starts at plugin 25.10") {
    RapidsPluginCapabilities.supportsMaxConcurrentGpuTasks(Some("25.06.0")) shouldBe false
    RapidsPluginCapabilities.supportsMaxConcurrentGpuTasks(Some("25.08.0")) shouldBe false
    RapidsPluginCapabilities.supportsMaxConcurrentGpuTasks(Some("25.10.0")) shouldBe true
    RapidsPluginCapabilities.supportsMaxConcurrentGpuTasks(Some("26.02.0")) shouldBe true
  }

  test("unknown plugin version does not claim maxConcurrentGpuTasks capability") {
    RapidsPluginCapabilities.supportsMaxConcurrentGpuTasks(None) shouldBe false
  }

  test("current RMM profile is open-ended from plugin 25.06") {
    val versions = Seq("25.06.0", "25.06.1", "26.10.0", "27.02.0")

    versions.foreach { version =>
      val profile = rightValue(RapidsPluginCapabilities.rmmPoolSizingProfile(version))
      profile.info.id shouldBe "discrete-gpu-rmm-v1"
      profile.info.pluginVersions.minPluginVersionInclusive shouldBe "25.06.0"
      profile.info.pluginVersions.maxPluginVersionExclusive shouldBe None
    }
  }

  test("RMM profile selection rejects old and malformed plugin versions") {
    Seq("25.04.0", "25.06", "99.bad", "").foreach { version =>
      leftMessage(RapidsPluginCapabilities.rmmPoolSizingProfile(version)) should
        include("no RMM pool sizing profile matches")
    }
  }

  test("plugin version ranges support a max-exclusive profile handoff") {
    val v1 = PluginVersionRange(
      minPluginVersionInclusive = "25.06.0",
      maxPluginVersionExclusive = Some("26.10.0"))
    val v2 = PluginVersionRange(
      minPluginVersionInclusive = "26.10.0",
      maxPluginVersionExclusive = None)

    v1.contains("25.06.0") shouldBe true
    v1.contains("26.08.0") shouldBe true
    v1.contains("26.10.0") shouldBe false
    v2.contains("26.08.0") shouldBe false
    v2.contains("26.10.0") shouldBe true
    v2.contains("27.02.0") shouldBe true
  }

  test("overlapping RMM profiles fail selection instead of choosing by order") {
    val profile = rightValue(RapidsPluginCapabilities.rmmPoolSizingProfile("25.08.0"))
    val profiles = Seq(profile, profile)

    leftMessage(RapidsRmmPoolModel.selectCompatibilityProfile("25.08.0", profiles)) should
      include("multiple RMM pool sizing profiles match")
  }
}
