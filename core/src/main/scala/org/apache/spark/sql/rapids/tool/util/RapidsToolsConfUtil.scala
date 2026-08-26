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

import java.io.FileNotFoundException
import java.util.Properties

import scala.jdk.CollectionConverters._

import org.apache.hadoop.conf.Configuration

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession

/**
 * Contains util methods to interact with Spark/Hadoop configurations.
 */
object RapidsToolsConfUtil extends Logging {
  private val RAPIDS_TOOLS_HADOOP_CONF_PREFIX = s"${RAPIDS_TOOLS_SYS_PROP_PREFIX}hadoop."
  // Directory name inside resources that hosts all the configurations in plain properties format
  private val CONFIG_DIR = "/configs"
  private val BUILD_PROPS_FILE_NAME = "build.properties"

  lazy val toolsBuildProperties = loadBuildProperties()

  /**
   * Creates a sparkConfiguration object with system properties applied on-top.
   * @return a hadoop configuration object
   */
  def newHadoopConf(): Configuration = {
    newHadoopConf(loadConfFromSystemProperties)
  }

  /**
   * Creates a new Configuration object after applying the properties on top.
   * The configuration map keys should be prefixed with "rapids.tools.hadoop."
   * @param confMap a key value pair of properties to be applied to the default configurations.
   * @return a hadoop configuration object
   */
  def newHadoopConf(confMap: Map[String, String]): Configuration = {
    // Note that we do not want to use SparkHadoopConfUtil.get.newConfiguration because
    // spark applies spark configurations on top of the hadoop configs.
    // A use-case is when the runtime is running as a Java command (not spark),
    // then the default spark configurations override the actual hadoop configurations.
    // For more details, see https://github.com/NVIDIA/cudf-spark-tools/issues/350
    val hadoopConf = SparkSession.getActiveSession match {
      case Some(spark) =>
        // get the hadoop configuration attached to the session
        new Configuration(spark.sparkContext.hadoopConfiguration)
      case _ =>
        new Configuration()
    }
    // append the configuration map
    appendRapidsToolsHadoopConfigs(confMap, hadoopConf)
    hadoopConf
  }

  /**
   * Appends rapids.tools.hadoop.* configurations from a Map to another without
   * the rapids.tools.hadoop. prefix.
   */
  def appendRapidsToolsHadoopConfigs(
      srcMap: Map[String, String],
      destMap: Configuration): Unit = {
    // Copy any "rapids.tools.hadoop.foo=bar" system properties into destMap as "foo=bar"
    for ((key, value) <- srcMap if key.startsWith(RAPIDS_TOOLS_HADOOP_CONF_PREFIX)) {
      val k = key.substring(RAPIDS_TOOLS_HADOOP_CONF_PREFIX.length)
      destMap.set(k, value)
    }
  }

  /**
   * Reads a properties file from resources/configs.
   * If the file cannot be loaded, an error message will show in the log.
   * Note that this should not happen because it is an internal functionality.
   * @param fileName the name of the file in the directory
   * @return a Java properties object
   */
  private def loadPropFile(
      fileName: String,
      filterPredicate: (String, String) => Boolean): Properties = {
    val props: Properties = new SortedJProperties
    val propsFilePath = s"$CONFIG_DIR/$fileName"
    getClass.getResourceAsStream(propsFilePath) match {
      case null => // return empty properties if the file cannot be loaded
        logError("Cannot load properties from file", new FileNotFoundException(fileName))
      case stream =>
        props.load(UTF8Source.fromInputStream(stream).bufferedReader())
        props.stringPropertyNames.asScala
          .filter(k => filterPredicate(k, props.getProperty(k)))
          .foreach(k => props.setProperty(k, props.getProperty(k)))
    }
    props
  }

  def loadBuildProperties(
      filterPredicate: (String, String) => Boolean = (_, _) => true): Properties = {
    loadPropFile(BUILD_PROPS_FILE_NAME, filterPredicate)
  }
}
