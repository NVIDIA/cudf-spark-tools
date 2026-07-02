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

package org.apache.spark.sql.rapids.tool.util

import java.io.{InputStream, IOException}
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Base64

import scala.collection.mutable
import scala.jdk.CollectionConverters._
import scala.util.control.NonFatal
import scala.xml.XML

import org.jsoup.Jsoup

import org.apache.spark.internal.Logging


/**
 * Utility to extract information about RAPIDS packages from mvn URLs.
 */
object WebCrawlerUtil extends Logging {
  private val MAX_CRAWLER_DEPTH = 1
  private val MAVEN_CENTRAL_BASE_URL = "https://repo1.maven.org/maven2"
  private val RAPIDS_GROUP_PATH = "com/nvidia"
  private val MAVEN_BASE_URL_ENV = "RAPIDS_TOOLS_MAVEN_BASE_URL"
  private val MAVEN_USERNAME_ENV = "RAPIDS_TOOLS_MAVEN_USERNAME"
  private val MAVEN_PASSWORD_ENV = "RAPIDS_TOOLS_MAVEN_PASSWORD"
  // defines the artifacts of the RAPIDS libraries
  private val NV_ARTIFACTS_LOOKUP = Map(
    "rapids.plugin" -> "rapids-4-spark_2.12",
    "rapids.tools" -> "rapids-4-spark-tools_2.12"
  )
  private val MAVEN_META_FILE = "maven-metadata.xml"
  // regular expression used to extract the version number from
  // the mvn repository url
  private val ARTIFACT_VERSION_REGEX = "\\d{2}\\.\\d{2}\\.\\d+/"

  def normalizeMavenBaseUrl(baseUrl: String): String = {
    baseUrl.stripSuffix("/")
  }

  def getMavenBaseUrl(env: Map[String, String] = sys.env): String = {
    env.get(MAVEN_BASE_URL_ENV)
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(normalizeMavenBaseUrl)
      .getOrElse(MAVEN_CENTRAL_BASE_URL)
  }

  private def getMavenArtifactRootUrl(env: Map[String, String]): String = {
    s"${getMavenBaseUrl(env)}/$RAPIDS_GROUP_PATH"
  }

  private def getMavenBasicAuthHeader(env: Map[String, String]): Option[String] = {
    for {
      user <- env.get(MAVEN_USERNAME_ENV).map(_.trim).filter(_.nonEmpty)
      password <- env.get(MAVEN_PASSWORD_ENV).filter(_.nonEmpty)
    } yield {
      val token = Base64.getEncoder.encodeToString(
        s"$user:$password".getBytes(StandardCharsets.UTF_8))
      s"Basic $token"
    }
  }

  private def openMavenUrlStream(
      mavenURL: String,
      env: Map[String, String]): InputStream = {
    val connection = new URL(mavenURL).openConnection()
    getMavenBasicAuthHeader(env).foreach { authHeader =>
      connection.setRequestProperty("Authorization", authHeader)
    }
    connection.getInputStream
  }

  // given an artifactID returns the full mvn url that lists all the
  // releases
  def getMVNArtifactURL(
      artifactID: String,
      env: Map[String, String] = sys.env) : String = {
    val artifactUrlPart = NV_ARTIFACTS_LOOKUP.getOrElse(artifactID, artifactID)
    s"${getMavenArtifactRootUrl(env)}/$artifactUrlPart"
  }

  def getMVNMetaURL(
      artifactID: String,
      env: Map[String, String] = sys.env) : String = {
    val artifactUrlPart = getMVNArtifactURL(artifactID, env)
    s"$artifactUrlPart/$MAVEN_META_FILE"
  }

  /**
   * Given a valid URL, this method recursively picks all the hrefs defined in the HTML doc.
   * @param webURL a valid URL to the html page to be crawled
   * @param regEx a valid regular expression as defined by the jsoup library.
   *              For more details, check https://jsoup.org/cookbook/extracting-data/selector-syntax
   * @param maxDepth the level of recursion to get the links.
   * @return a set of all links found recursively in the given url
   */
  def getPageLinks(
      webURL: String,
      regEx: Option[String],
      maxDepth: Int = MAX_CRAWLER_DEPTH,
      env: Map[String, String] = sys.env): mutable.Set[String] = {
    def removeDefaultPorts(rawLink: String): String = {
      val jURL: URL = new URL(rawLink)
      jURL.getProtocol match {
        case "http" if jURL.getPort == 80 =>
          rawLink.replace(":80", "")
        case "https" if jURL.getPort == 443 =>
          rawLink.replace(":443", "")
        case _ => rawLink
      }
    }
    def processPageLinksInternal(currURL: String,
        cssQuery: String,
        currDepth: Int,
        allLinks: mutable.Set[String]): Unit = {
      if (currDepth < maxDepth && !allLinks.contains(currURL)) {
        try {
          val connection = Jsoup.connect(currURL)
          getMavenBasicAuthHeader(env).foreach { authHeader =>
            connection.header("Authorization", authHeader)
          }
          val doc = connection.get
          val pageURLs = doc.select(cssQuery).asScala.toList
          val newDepth = currDepth + 1
          for (page <- pageURLs) {
            val newLink = page.attr("abs:href")
            // remove the default port if any
            allLinks.add(removeDefaultPorts(newLink))
            processPageLinksInternal(newLink, cssQuery, newDepth, allLinks)
          }
        } catch {
          case x: IOException =>
            logWarning(s"Exception while visiting webURL $webURL: ${x.toString}")
        }
      }
    }

    val selectorQuery = regEx match {
      case Some(value) => s"a[href~=$value]"
      case None => "a[href]"
    }
    val links = mutable.Set[String]()
    processPageLinksInternal(webURL, selectorQuery, 0, links)
    links
  }

  // given an artifactID, returns a list of strings containing all
  // available releases.
  def getMvnReleasesForNVPackage(
      artifactID: String,
      env: Map[String, String] = sys.env): Seq[String] = {
    val mvnURL = getMVNArtifactURL(artifactID, env)
    val definedLinks = getPageLinks(mvnURL, Some(ARTIFACT_VERSION_REGEX), env = env).toSeq.sorted
    definedLinks.map(_.split("/").last)
  }

  // given an artifactID, will return the latest version if any
  def getLatestMvnReleaseForNVPackage(
      artifactID: String,
      env: Map[String, String] = sys.env): Option[String] = {
    // Reads maven-metadata.xml file to extract the latest version
    val mvnMetaFile = getMVNMetaURL(artifactID, env)
    try {
      val inputStream = openMavenUrlStream(mvnMetaFile, env)
      try {
        val xml = XML.load(inputStream)
        Some((xml \\ "metadata" \ "versioning" \ "latest").text)
      } finally {
        inputStream.close()
      }
    } catch {
      case NonFatal(e) =>
        logWarning(s"Exception loading maven-metadata.xml: ${mvnMetaFile}", e)
      None
    }
  }

  // given artifactID and release, returns the full mvn url to download the jar
  def getMvnDownloadLink(
      artifactID: String,
      release: String,
      env: Map[String, String] = sys.env): String = {
    s"${getMVNArtifactURL(artifactID, env)}/$release/$artifactID-$release.jar"
  }

  def getPluginMvnDownloadLink(
      release: String,
      env: Map[String, String] = sys.env): String = {
    getMvnDownloadLink(NV_ARTIFACTS_LOOKUP("rapids.plugin"), release, env)
  }

  // get the latest version available for rapids plugin
  def getLatestPluginRelease: Option[String] = {
    getLatestMvnReleaseForNVPackage(NV_ARTIFACTS_LOOKUP("rapids.plugin"))
  }

  // get the latest version available for rapids tools
  def getLatestToolsRelease: Option[String] = {
    getLatestMvnReleaseForNVPackage(NV_ARTIFACTS_LOOKUP("rapids.tools"))
  }
}
