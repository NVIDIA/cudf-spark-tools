/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.util

import java.io.File
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.text.SimpleDateFormat
import java.util.Base64
import java.util.Calendar

import scala.concurrent.duration._
import scala.xml.XML

import com.nvidia.spark.rapids.tool.ToolTestUtils
import com.nvidia.spark.rapids.tool.profiling.{ProfileOutputWriter, ProfileResult}
import com.sun.net.httpserver.HttpServer
import org.scalatest.AppendedClues.convertToClueful
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers.{contain, convertToAnyShouldWrapper, equal, not}

import org.apache.spark.internal.Logging
import org.apache.spark.network.util.ByteUnit
import org.apache.spark.sql.TrampolineUtil
import org.apache.spark.sql.rapids.tool.InvalidMemoryUnitFormatException
import org.apache.spark.sql.rapids.tool.util._

class ToolUtilsSuite extends AnyFunSuite with Logging {
  test("get page links of a url") {
    // Tests that getPageLinks return all the [href] in a page.
    // This is done manually by checking against a URL that won't likely change
    // (aka. an old release page should be stable).

    val version = "25.08.0"
    val mvnPrefix = ToolTestUtils.pluginMvnPrefix(version)
    val webURL = ToolTestUtils.pluginMvnURL(version)
    val allLinks = WebCrawlerUtil.getPageLinks(webURL, None)

    val expected = Set[String](
      s"$mvnPrefix.pom.asc.sha1",
      s"$mvnPrefix-javadoc.jar.asc.sha1",
      s"${ToolTestUtils.RAPIDS_MVN_BASE_URL}",
      s"$mvnPrefix-sources.jar.md5",
      s"$mvnPrefix-cuda12.jar.md5",
      s"$mvnPrefix.pom.asc",
      s"$mvnPrefix-sources.jar.asc.sha1",
      s"$mvnPrefix-sources.jar",
      s"$mvnPrefix.jar.asc",
      s"$mvnPrefix.pom.asc.md5",
      s"$mvnPrefix.jar.sha1",
      s"$mvnPrefix.jar.md5",
      s"$mvnPrefix-cuda12.jar",
      s"$mvnPrefix.pom",
      s"$mvnPrefix-sources.jar.asc",
      s"$mvnPrefix-cuda12.jar.asc.sha1",
      s"$mvnPrefix.pom.md5",
      s"$mvnPrefix-sources.jar.asc.md5",
      s"$mvnPrefix.pom.sha1",
      s"$mvnPrefix-javadoc.jar.asc",
      s"$mvnPrefix-cuda12.jar.sha1",
      s"$mvnPrefix-javadoc.jar",
      s"$mvnPrefix-javadoc.jar.asc.md5",
      s"$mvnPrefix.jar",
      s"$mvnPrefix-cuda12.jar.asc.md5",
      s"$mvnPrefix-javadoc.jar.md5",
      s"$mvnPrefix-javadoc.jar.sha1",
      s"$mvnPrefix-sources.jar.sha1",
      s"$mvnPrefix.jar.asc.sha1",
      s"$mvnPrefix.jar.asc.md5",
      s"$mvnPrefix-cuda12.jar.asc"
    )

    // all links should be matching
    allLinks should contain theSameElementsAs expected
  }

  // checks that regex is used correctly to filter the href pulled from a given url
  test("get page links of a url with regex") {
    // see the list of available regex in https://jsoup.org/cookbook/extracting-data/selector-syntax
    val version = "25.08.0"
    val mvnPrefix = ToolTestUtils.pluginMvnPrefix(version)
    val webURL = ToolTestUtils.pluginMvnURL(version)
    val jarFileRegEx = ".*\\.jar$"
    val allLinks = WebCrawlerUtil.getPageLinks(webURL, Some(jarFileRegEx))
    val expected = Set[String](
      s"$mvnPrefix-cuda12.jar",
      s"$mvnPrefix-javadoc.jar",
      s"$mvnPrefix-sources.jar",
      s"$mvnPrefix.jar"
    )
    // all links should end with jar files
    allLinks should contain theSameElementsAs expected
  }

  //
  test("list available mvn releases") {
    // Use Maven Central explicitly because the expected values below read from that URL.
    val artifactID = "rapids-4-spark_2.12"
    val baseURL = ToolTestUtils.RAPIDS_MVN_BASE_URL
    val nvReleases = WebCrawlerUtil.getMvnReleasesForNVPackage(artifactID, Map.empty)
    // get all the links on the page
    val allLinks = WebCrawlerUtil.getPageLinks(baseURL, None).mkString("\n")
    val versionPattern = "(\\d{2}\\.\\d{2}\\.\\d+)/".r
    val actualVersions = versionPattern.findAllMatchIn(allLinks).map(_.group(1)).toSeq
    nvReleases should contain theSameElementsAs actualVersions
  }

  test("get latest release") {
    // Use Maven Central explicitly because the expected value below reads from that URL.
    val artifactID = "rapids-4-spark_2.12"
    val baseURL = ToolTestUtils.RAPIDS_MVN_BASE_URL
    val latestRelease = WebCrawlerUtil.getLatestMvnReleaseForNVPackage(
      artifactID, Map.empty) match {
      case Some(v) => v
      case None => fail("Could not find pull the latest release successfully")
    }
    // get all the links on the page
    val mavenMetaXml = XML.load(s"${baseURL}maven-metadata.xml")
    val allVersions = (mavenMetaXml \\ "metadata" \ "versioning" \ "versions" \ "version").toList
    // get the latest release from the mvn url
    val actualRelease = allVersions.last.text
    actualRelease.matches("\\d{2}\\.\\d{2}\\.\\d+") shouldBe true
    latestRelease shouldBe actualRelease
  }

  test("Maven base URL defaults to Maven Central") {
    WebCrawlerUtil.getMavenBaseUrl(Map.empty) shouldBe "https://repo1.maven.org/maven2"
  }

  test("Maven base URL can be overridden by environment") {
    WebCrawlerUtil.getMavenBaseUrl(
      Map("RAPIDS_TOOLS_MAVEN_BASE_URL" -> "https://mirror.example/maven/")) shouldBe
      "https://mirror.example/maven"
  }

  test("Maven artifact URL uses default Maven Central when environment is unset") {
    assume(sys.env.get("RAPIDS_TOOLS_MAVEN_BASE_URL").forall(_.trim.isEmpty))
    WebCrawlerUtil.getMVNArtifactURL("rapids.plugin") shouldBe
      "https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12"
  }

  test("Maven metadata request uses configured basic auth") {
    val user = "maven-user"
    val password = "maven-password"
    val token = Base64.getEncoder.encodeToString(
      s"$user:$password".getBytes(StandardCharsets.UTF_8))
    val expectedAuth = s"Basic $token"
    val server = HttpServer.create(new InetSocketAddress("localhost", 0), 0)

    try {
      server.createContext("/com/nvidia/test-artifact/maven-metadata.xml", exchange => {
        val response =
          if (exchange.getRequestHeaders.getFirst("Authorization") == expectedAuth) {
            exchange.sendResponseHeaders(200, 0)
            "<metadata><versioning><latest>26.04.0</latest></versioning></metadata>"
          } else {
            exchange.sendResponseHeaders(401, 0)
            "unauthorized"
          }
        val bytes = response.getBytes(StandardCharsets.UTF_8)
        exchange.getResponseBody.write(bytes)
        exchange.close()
      })
      server.start()

      val env = Map(
        "RAPIDS_TOOLS_MAVEN_BASE_URL" -> s"http://localhost:${server.getAddress.getPort}",
        "RAPIDS_TOOLS_MAVEN_USERNAME" -> user,
        "RAPIDS_TOOLS_MAVEN_PASSWORD" -> password)

      WebCrawlerUtil.getLatestMvnReleaseForNVPackage("test-artifact", env) shouldBe
        Some("26.04.0")
    } finally {
      server.stop(0)
    }
  }

  test("Hadoop Configuration should load system properties") {
    // Tests that Hadoop configurations can load the system property passed to the
    // command line. i.e., "-Drapids.tools.hadoop.property.key=value"
    TrampolineUtil.cleanupAnyExistingSession()
    // sets a hadoop property through Rapids-Tools prefix
    System.setProperty("rapids.tools.hadoop.property.key1", "value1")
    lazy val hadoopConf = RapidsToolsConfUtil.newHadoopConf()
    hadoopConf.get("property.key1") shouldBe "value1"
    System.clearProperty("rapids.tools.hadoop.property.key1")
  }

  test("parse timeFormat 'HH:MM:SS.FFF' to Long") {
    val currCalendar = Calendar.getInstance()
    val acceptedTimeFormat = new SimpleDateFormat("HH:mm:ss.SSS")

    val hour = currCalendar.get(Calendar.HOUR_OF_DAY)
    val minute = currCalendar.get(Calendar.MINUTE)
    val seconds = currCalendar.get(Calendar.SECOND)
    val millis = currCalendar.get(Calendar.MILLISECOND)
    val currTimeAsStr = acceptedTimeFormat.format(currCalendar.getTime)
    val duration =
      hour.hours.toMillis + minute.minutes.toMillis + seconds.seconds.toMillis + millis
    // test successful parsing
    StringUtils.parseFromDurationToLongOption(currTimeAsStr).get shouldBe duration
    // test non-successful with milliseconds is not 3 digits
    val timeNoMillisAsStr = f"$hour:$minute%02d:$seconds%02d.1"
    StringUtils.parseFromDurationToLongOption(timeNoMillisAsStr).get shouldBe {
      duration - millis + 100
    }
    // test successful with high number of hours
    StringUtils.parseFromDurationToLongOption("50:30:30.000").get shouldBe {
      50.hours.toMillis + 30.minutes.toMillis + 30.seconds.toMillis
    }
    // test incorrect format with overflowing minutes
    val timeBrokenMinutesAsString = f"$hour:${minute + 60}%02d:$seconds%02d.$millis%03d"
    StringUtils.parseFromDurationToLongOption(timeBrokenMinutesAsString) should not be 'defined
    // test random string won't cause any exceptions
    StringUtils.parseFromDurationToLongOption("Hello Worlds") should not be 'defined
  }

  test("parse memory sizes from RAPIDS metrics") {
    // This unit test is to evaluate the parser used to handle the new GPU metrics introduced in
    // https://github.com/NVIDIA/spark-rapids/pull/12517
    // Note that:
    // - the metrics are in human readable format (e.g., 0.74GB (11534336000 bytes)).
    // - we do not care to evaluate the case when the metric is just "0" because this would be
    //   handled by a different parser EventUtils.parseAccumFieldToLong

    // test default case
    StringUtils.parseFromGPUMemoryMetricToLongOption("41.47GB (44533943056 bytes)") shouldBe
      Some(44533943056L)
    // test no floating point
    StringUtils.parseFromGPUMemoryMetricToLongOption("41GB (44533943056 bytes)") shouldBe
      Some(44533943056L)
    // test lower case
    StringUtils.parseFromGPUMemoryMetricToLongOption("41gb (44533943056 bytes)") shouldBe
      Some(44533943056L)
    // test different formatting
    StringUtils.parseFromGPUMemoryMetricToLongOption("41GiB (44533943056 bytes)") shouldBe
      Some(44533943056L)
    // test bytes as unit
    StringUtils.parseFromGPUMemoryMetricToLongOption(
      "44533943056B (44533943056 bytes)") shouldBe Some(44533943056L)
    // test correct formatting with 0.
    StringUtils.parseFromGPUMemoryMetricToLongOption("0B (0 bytes)") shouldBe Some(0)
    // test incorrect formatting with W/S separating memory units
    StringUtils.parseFromGPUMemoryMetricToLongOption("41 GiB (44533943056 bytes)") shouldBe
      None
  }

  test("output non-english characters") {
    val nonEnglishString = "你好"
    TrampolineUtil.withTempDir { tempDir =>
      val filePrefix = "non-english"
      val tableHeader = "Non-English"
      val textFilePath = s"${tempDir.getAbsolutePath}/$filePrefix.log"
      val csvFilePath = s"${tempDir.getAbsolutePath}/$filePrefix.csv"
      val profOutputWriter =
        new ProfileOutputWriter(tempDir.getAbsolutePath, filePrefix, 1000, outputCSV = true)
      val profResults = Seq(
        MockProfileResults("appID-0", nonEnglishString, Seq(1, 2, 3).mkString(","))
      )
      try {
        profOutputWriter.writeTable(tableHeader, profResults)
      } finally {
        profOutputWriter.close()
      }
      val csvFile = new File(csvFilePath)
      val textFile = new File(textFilePath)
      assert(csvFile.exists())
      assert(textFile.exists())
      val expectedCSVFileContent =
        s"""appID,nonEnglishField,parentIDs
           |appID-0,"你好","1,2,3"""".stripMargin
      val expectedTXTContent =
        s"""
           |Non-English:
           |+-------+---------------+---------+
           ||appID  |nonEnglishField|parentIDs|
           |+-------+---------------+---------+
           ||appID-0|你好           |1,2,3    |
           |+-------+---------------+---------+""".stripMargin
      val actualCSVContent = FSUtils.readFileContentAsUTF8(csvFilePath)
      val actualTXTContent = FSUtils.readFileContentAsUTF8(textFilePath)
      actualCSVContent should equal (expectedCSVFileContent)
      actualTXTContent should equal (expectedTXTContent)
    }
  }

  test("Finding median of arrays") {
    val testSet: Map[String, (Array[Long], Long)] = Map(
      "All same values" -> (Array[Long](5, 5, 5, 5) -> 5L),
      "Odd number of values [9, 7, 5, 3, 1]" -> (Array[Long](9, 7, 5, 3, 1) -> 5L),
      "Even number of values [11, 9, 7, 5, 3, 1]" -> (Array[Long](11, 9, 7, 5, 3, 1) -> 6),
      "Even number of values(2) [15, 13, 11, 9, 7, 5, 3, 1]" ->
        (Array[Long](15, 13, 11, 9, 7, 5, 3, 1) -> 8),
      "Even number of values(3) [3, 13, 11, 9, 7, 5, 15, 1]" ->
        (Array[Long](3, 13, 11, 9, 7, 5, 15, 1) -> 8),
      "Single element" -> (Array[Long](1) -> 1),
      "Two elements" -> (Array[Long](1, 2).reverse -> 1)
    )
    for ((desc, (arr, expectedMedian)) <- testSet) {
      val actualMedian =
        InPlaceMedianArrView.findMedianInPlace(arr)(InPlaceMedianArrView.chooseMidpointPivotInPlace)
      actualMedian shouldBe expectedMedian withClue s"Failed for $desc. " +
        s"Expected: $expectedMedian, " +
        s"Actual: $actualMedian"
    }
  }

  test("convertMemorySizeToBytes should correctly parse memory sizes") {
    // Test basic unit conversions with default ByteUnit.BYTE
    StringUtils.convertMemorySizeToBytes("1024b", None) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1k", None) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1m", None) shouldBe 1024L * 1024
    StringUtils.convertMemorySizeToBytes("1g", None) shouldBe 1024L * 1024 * 1024

    // Test decimal values
    StringUtils.convertMemorySizeToBytes("1.5k", None) shouldBe (1.5 * 1024).toLong
    StringUtils.convertMemorySizeToBytes("2.5g", None) shouldBe (2.5 * 1024 * 1024 * 1024).toLong

    // Test case insensitivity
    StringUtils.convertMemorySizeToBytes("1K", None) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1KB", None) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1KiB", None) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1M", None) shouldBe 1024L * 1024
    StringUtils.convertMemorySizeToBytes("1G", None) shouldBe 1024L * 1024 * 1024

    // Test with different default units
    // When unit is specified in string, defaultUnit should be ignored
    StringUtils.convertMemorySizeToBytes("1024b", Some(ByteUnit.KiB)) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1k", Some(ByteUnit.MiB)) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1m", Some(ByteUnit.GiB)) shouldBe 1024L * 1024

    // When no unit in string, defaultUnit should be used
    StringUtils.convertMemorySizeToBytes("1", Some(ByteUnit.KiB)) shouldBe 1024L
    StringUtils.convertMemorySizeToBytes("1", Some(ByteUnit.MiB)) shouldBe 1024L * 1024
    StringUtils.convertMemorySizeToBytes("1", Some(ByteUnit.GiB)) shouldBe 1024L * 1024 * 1024

    // Test decimal values with different default units
    StringUtils.convertMemorySizeToBytes("1.5", Some(ByteUnit.KiB)) shouldBe (
      1.5 * 1024).toLong
    StringUtils.convertMemorySizeToBytes("2.5", Some(ByteUnit.GiB)) shouldBe (
      2.5 * 1024 * 1024 * 1024).toLong
    StringUtils.convertMemorySizeToBytes("0.5", Some(ByteUnit.TiB)) shouldBe (
      0.5 * 1024 * 1024 * 1024 * 1024).toLong

    // Test zero values
    StringUtils.convertMemorySizeToBytes("0b", None) shouldBe 0L
    StringUtils.convertMemorySizeToBytes("0k", None) shouldBe 0L
    StringUtils.convertMemorySizeToBytes("0", Some(ByteUnit.GiB)) shouldBe 0L
  }

  test("convertMemorySizeToBytes should handle invalid input") {
    intercept[InvalidMemoryUnitFormatException] {
      StringUtils.convertMemorySizeToBytes("invalid", None)
    }
    intercept[InvalidMemoryUnitFormatException] {
      StringUtils.convertMemorySizeToBytes("invalid", Some(ByteUnit.GiB))
    }
    intercept[InvalidMemoryUnitFormatException] {
      StringUtils.convertMemorySizeToBytes("1z", None)
    }
    intercept[InvalidMemoryUnitFormatException] {
      StringUtils.convertMemorySizeToBytes("-1k", None)
    }
    intercept[InvalidMemoryUnitFormatException] {
      StringUtils.convertMemorySizeToBytes("1.5", None)
    }
  }

  test("convertBytesToLargestUnit should convert bytes to appropriate units") {
    // Test exact conversions that result in whole numbers
    StringUtils.convertBytesToLargestUnit(1024) shouldBe "1k"
    StringUtils.convertBytesToLargestUnit(4194304) shouldBe "4m"
    StringUtils.convertBytesToLargestUnit(2147483648L) shouldBe "2g"
    StringUtils.convertBytesToLargestUnit(5497558138880L) shouldBe "5t"

    // Test values that should remain in bytes to avoid fractions
    StringUtils.convertBytesToLargestUnit(1536) shouldBe "1536b"
    StringUtils.convertBytesToLargestUnit(1) shouldBe "1b"
    StringUtils.convertBytesToLargestUnit(0) shouldBe "0b"

    // Test falling back to lower units to avoid fractions
    // 1.5m -> fallback to 1536k
    StringUtils.convertBytesToLargestUnit(1536 * 1024) shouldBe "1536k"
    // 1.5g -> fallback to 1536m
    StringUtils.convertBytesToLargestUnit(1536 * 1024 * 1024) shouldBe "1536m"

    // Test complex fallback cases
    // 1g + 1 byte -> bytes
    StringUtils.convertBytesToLargestUnit(1024 * 1024 * 1024 + 1) shouldBe "1073741825b"
    // 2m + 1k -> k
    StringUtils.convertBytesToLargestUnit(2048 * 1024 + 1024) shouldBe "2049k"
    // 3t + 1m -> m
    StringUtils.convertBytesToLargestUnit(3L * 1024 * 1024 * 1024 * 1024 + 1024 * 1024) shouldBe
      "3145729m"

    // Test large values that don't perfectly align with units
    StringUtils.convertBytesToLargestUnit(1234567) shouldBe "1234567b"
  }

  case class MockProfileResults(appID: String, nonEnglishField: String,
      parentIDs: String) extends ProfileResult {
    override val outputHeaders: Array[String] = Array("appID", "nonEnglishField",
      "parentIDs")

    override def convertToSeq(): Array[String] = {
      Array(appID, nonEnglishField, parentIDs)
    }

    override def convertToCSVSeq(): Array[String] = {
      Array(appID, StringUtils.reformatCSVString(nonEnglishField),
        StringUtils.reformatCSVString(parentIDs))
    }
  }
}
