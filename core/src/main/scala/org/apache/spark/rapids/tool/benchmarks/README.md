# Benchmarking Tools

This package contains the relevant classes to write and run benchmarks for `NVIDIA cuDF plugin for Apache Spark tools` for Apache Spark.

## Writing a benchmark -
* Extend `BenchmarkBase` and override the `runBenchmarkSuite` function
* Write logically similar benchmarks inside the `runBenchmark` which will add a header to the output with the name provided.
* Now for each case to be tested, use the Benchmark class `addCase` to write various cases of the same Benchmark.
* Call the run function to run the benchmark
* Refer included example benchmark - [`SingleThreadedQualToolBenchmark`](./SingleThreadedQualToolBenchmark.scala) for implementation details

## Running the benchmark -
Use the java command to run the created benchmark class with the following supported params -
* `-i` : total number of iterations to run to calculate average metrics
* `-w` : total number of warmup iterations to run before calculating the final metrics ( warmup is relevant so that final results are not skewed by the initial java classloading times )
* `-o` : output directory where to store the final result file. Defaults to the directory rapids-tools-benchmark in the root directory
* `-f` : output format of the stored result. Currently supports text. Json to be added in future iterations
* `-a` : input arguments to pass the underlying benchmark classes

#### Running the Benchmark class directly
```shell
java -cp $CLASSPATH:$SPARK_HOME/jars/*:$MAVEN-ARTIFACT-JAR \
org.apache.spark.rapids.tool.benchmarks.SingleThreadedQualToolBenchmark \
-i 3 -w 3 -a " $EVENT_LOGS_DIR"
```
* `$CLASSPATH` : Path to the compiled class directory. Ex - `<SPARK_RAPIDS_TOOLS_REPO_PATH>/core/target/*`
* `$MAVEN-ARTIFACT-JAR` : Path to maven-artifact jar. Download jar from [here](https://mvnrepository.com/artifact/org.apache.maven/maven-artifact/3.9.0)
* `$EVENT_LOGS_DIR` : Path to the event logs directory

#### Running the Benchmark class using tools jar
```shell
java -cp $SPARK_RAPIDS_TOOLS_JAR:$SPARK_HOME/jars/* \
org.apache.spark.rapids.tool.benchmarks.SingleThreadedQualToolBenchmark \
-i 3 -w 3 -a " $EVENT_LOGS_DIR"
```
* `$SPARK_RAPIDS_TOOLS_JAR` : Path to the spark rapids tools jar
* `$EVENT_LOGS_DIR` : Path to the event logs directory

#### NOTES
* `$SPARK_HOME/jars/*` : Include the `SPARK` jars in the classpath in case benchmarking `QUALIFICATION/PROFILING` tool
