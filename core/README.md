# Qualification and Profiling tools

The Qualification tool is used to look at a set of applications to determine if the NVIDIA cuDF plugin for Apache Spark
might be a good fit for those applications.

The Profiling tool generates information which can be used for debugging and profiling applications.
Information such as Spark version, executor information, properties and so on. This runs on either CPU or
GPU generated event logs.

Please refer to [Qualification tool documentation](https://docs.nvidia.com/spark-rapids/user-guide/latest/qualification/overview.html)
and [Profiling tool documentation](https://docs.nvidia.com/spark-rapids/user-guide/latest/profiling/overview.html)
for more details on how to use the tools.

## Build

We use [Maven](https://maven.apache.org) for the build. Simply run as below command:

```shell script
mvn clean package
```

After a successful build, the jar will be in the `target/` directory. By default, builds target Scala 2.12
and the artifact is named like `rapids-4-spark-tools_2.12-*-SNAPSHOT.jar`. To build for Scala 2.13, use the
Maven profile `-Pscala213`, which produces artifacts named like `rapids-4-spark-tools_2.13-*-SNAPSHOT.jar`.
This will build the plugin for a single version of Spark. By default, this is Apache Spark 3.5.7.

Example: build for Scala 2.13 (default Spark 3.5.7)

```bash
mvn clean package -Pscala213
```

For development purpose, you may need to run the tests against different spark versions.
To run the tests against a specific Spark version, you can use the `-Dbuildver=XXX` command line option.
For instance to build Spark 3.5.7 you would use:

```shell script
mvn -Dbuildver=357 clean package
```

Run `mvn help:all-profiles` to list supported Spark versions.

### Building JAR for release

To build a release JAR file, run the profile `release`

```shell script
mvn clean package -P release
```

### Running tests

The unit tests are run by default when building unless they are explicitly skipped by specifying `-DskipTests`.

To run an individual test the `-Dsuites` option can be specified:

```bash
mvn test -Dsuites=com.nvidia.spark.rapids.tool.qualification.QualificationSuite
```

#### Regenerating golden sets

- To regenerate golden sets, you can use the `-Dtools.qual.test.generate.golden.enable` command line option.
- By default, the location of golden sets is `golden-sets/{buildVer}/qual` by the property
  `tools.qual.test.generate.golden.dir` in the pom file.
- For troubleshooting, you can configure the tests to keep the working directory. This is achieved by
  passing the `-Dtools.test.cleanup.tmp.dir=false` command line option.
- Generate the Qual table output by running the class `com.nvidia.spark.rapids.tool.views.qualification.QualYamlConfigLoader`

### Setting up an Integrated Development Environment

Before proceeding with importing cudf-spark-tools into IDEA or switching to a different Spark
profile, execute the installation's phase cmd with the corresponding `buildver`, e.g. for Spark 3.5.7:

##### Manual Maven Install for a target Spark build

```bash
 mvn clean install -Dbuildver=357 -Dmaven.scaladoc.skip -DskipTests
```

##### Importing the project

To start working with the project in IDEA is as easy as importing the project as a Maven project.
Select the profile used in the mvn command above, e.g. `spark357` for Spark 3.5.7.

The tools project follows the same coding style guidelines as the Apache Spark
project.  For IntelliJ IDEA users, an example `idea-code-style-settings.xml` is available in the
`scripts` subdirectory of the root project folder.
