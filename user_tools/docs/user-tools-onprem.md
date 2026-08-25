# NVIDIA cuDF plugin for Apache Spark User Tools on OnPrem platform

This is a guide for the tools for the NVIDIA cuDF plugin for Apache Spark clusters that are provisioned manually (a.k.a onPrem). At the end of this guide, the user will be able to run the cuDF plugin tools to analyze the clusters and the applications running on-premises.
Additionally, user can see cost savings and speedup recommendations for a comparable cluster on different cloud platforms by providing target_platform. Currently "`dataproc`" platform is supported.
If the target_platform is not specified, then the qualification recommendation is based on speedup which is the default behavior.
## Assumptions

The tool currently only supports event logs stored on local path. The remote output storage is also expected to be local.

## Prerequisites

### 1.cuDF plugin tools

- Spark event logs:
  - The cuDF plugin tools can process Apache Spark CPU event logs from Spark 2.0 or higher (raw, .lz4, .lzf, .snappy, .zstd)
  - For `qualification` commands, the event logs need to be archived to an accessible local directory

### 2.Install the package

- Install `spark-rapids-user-tools` with python [3.8, 3.11] using:
  - pip:  `pip install spark-rapids-user-tools`
  - wheel-file: `pip install <wheel-file>`
  - from source: `pip install -e .`
- verify the command is installed correctly by running
  ```bash
    spark_rapids_user_tools onprem -- --help
  ```

### 3.Environment variables

Before running any command, you can set environment variables to specify configurations.
- User tools variables use the `RAPIDS_USER_TOOLS_*` naming pattern:
  - `RAPIDS_USER_TOOLS_CACHE_FOLDER`: specifies the location of a local directory that the cuDF plugin CLI uses to store and cache the downloaded resources. The default is `/var/tmp/spark_rapids_user_tools_cache`.  Note that caching the resources locally has an impact on the total execution time of the command.
  - `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY`: specifies the location of a local directory that the cuDF plugin CLI uses to generate the output. The wrapper CLI arguments override that environment variable (`--local_folder` for Qualification).

## Qualification command

### Local deployment

```
spark_rapids_user_tools onprem qualification [options]
spark_rapids_user_tools onprem qualification -- --help
```

The local deployment runs on the local development machine. It requires:
1. Java 1.8+ development environment
2. Internet access to download JAR dependencies from mvn: `spark-*.jar`
3. Dependencies are cached on the local disk to reduce the overhead of the download.

#### Command options

| Option               | Description                                                                                                                                                                                                                                                                           | Default                                                                                                                                              | Required |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
| **cpu_cluster**      | The on-premises cluster on which the Apache Spark applications were executed. Accepted value is valid path to the cluster properties file (json format)                                                                                                                               | N/A                                                                                                                                                  |    N     |
| **eventlogs**        | A comma separated list of urls pointing to event logs in local directory                                                                                                                                                                                                              | Reads the Spark's property `spark.eventLog.dir` defined in `cpu_cluster`.  Note that the wrapper will raise an exception if the property is not set. |    N     |
| **local_folder**     | Local work-directory path to store the output and to be used as root directory for temporary folders/files. The final output will go into a subdirectory named `qual-${EXEC_ID}` where `exec_id` is an auto-generated unique identifier of the execution.                             | If the argument is NONE, the default value is the env variable `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY` if any; or the current working directory.        |    N     |
| **target_platform**  | Cost savings and speedup recommendation for comparable cluster in target_platform based on on-premises cluster configuration. Currently only `dataproc` is supported for target_platform.If not provided, the final report will be limited to GPU speedups only without cost-savings. | N/A                                                                                                                                                  |    N     |
| **jvm_heap_size**    | The maximum heap size of the JVM in gigabytes                                                                                                                                                                                                                                         | 24                                                                                                                                                   |    N     |
| **tools_jar**        | Path to a bundled jar including cuDF plugin tool. The path is a local filesystem                                                                                                                                                                                                           | Downloads the latest rapids-tools_*.jar from mvn repo                                                                                                |    N     |
| **filter_apps**      | Filtering criteria of the applications listed in the final STDOUT table is one of the following (`ALL`, `SPEEDUPS`). "`ALL`" means no filter applied. "`SPEEDUPS`" lists all the apps that are either '_Recommended_', or '_Strongly Recommended_' based on speedups.                 | `SPEEDUPS`                                                                                                                                           |    N     |
| **cpu_discount**     | A percent discount for the cpu cluster cost in the form of an integer value (e.g. 30 for 30% discount)                                                                                                                                                                                | N/A                                                                                                                                                  |    N     |
| **gpu_discount**     | A percent discount for the gpu cluster cost in the form of an integer value (e.g. 30 for 30% discount)                                                                                                                                                                                | N/A                                                                                                                                                  |    N     |
| **global_discount**  | A percent discount for both the cpu and gpu cluster costs in the form of an integer value (e.g. 30 for 30% discount)                                                                                                                                                                  | N/A                                                                                                                                                  |    N     |
| **verbose**          | True or False to enable verbosity to the wrapper script                                                                                                                                                                                                                               | False if `RAPIDS_USER_TOOLS_LOG_DEBUG` is not set                                                                                                    |    N     |
| **rapids_options**** | A list of valid [Qualification tool options](https://docs.nvidia.com/spark-rapids/user-guide/latest/qualification/jar-usage.html#running-the-qualification-tool-standalone-on-spark-event-logs). Note that (`output-directory`, `platform`) flags are ignored, and that multiple "spark-property" is not supported.                                              | N/A                                                                                                                                                  |    N     |

### Use case scenario to run qualification tool for on-premises Cluster

A typical workflow to successfully run the `qualification` command in local mode is described as follows:

1. Store the Apache Spark event logs in local directory.
2. On a machine with JDK8+ installed:
   1. user installs `spark_rapids_user_tools`
3. User defines the on-premises cluster on which the Spark application were running. Note that the cluster does not have to be
   active.
4. The following script runs qualification tool locally:

   ```
   # define the wrapper cache directory if necessary
   export RAPIDS_USER_TOOLS_CACHE_FOLDER=my_cache_folder
   export EVENTLOGS=local_path_to_eventlogs
   export CLUSTER_PROPERTY_FILE=my-onprem-cpu-cluster_property_file

   spark_rapids_user_tools onprem qualification \
      --eventlogs $EVENTLOGS \
      --cpu_cluster $CLUSTER_PROPERTY_FILE
   ```
   The wrapper generates a unique-Id for each execution in the format of `qual_<YYYYmmddHHmmss>_<0x%08X>`
   The above command will generate a directory containing `qualification_summary.csv` in addition to
   the actual folder of the cuDF plugin Qualification tool.

   ```
    ./qual_<YYYYmmddHHmmss>_<0x%08X>/qualification_summary.csv
    ./qual_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_qualification_output/
   ```

#### Qualification output

For each app, the command output lists the following fields:

- `App ID`: An application is referenced by its application ID, '_app-id_'. When running on YARN,
  each application may have multiple attempts, but there are attempt IDs only for applications
  in cluster mode, not applications in client mode.  Applications in YARN cluster mode can be
  identified by their attempt-id.
- `App Name`: Name of the application
- `Speedup Based Recommendation`: Recommendation based on '_Estimated Speed-up Factor_'. Note that an
  application that has job or stage failures will be labeled '_Not Applicable_'
- `Estimated GPU Speedup`: Speed-up factor estimated for the app. Calculated as the ratio
  between '_App Duration_' and '_Estimated GPU Duration_'.
- `Estimated GPU Duration`: Predicted runtime of the app if it was run on GPU
- `App Duration`: Wall-Clock time measured since the application starts till it is completed.
  If an app is not completed an estimated completion time would be computed.


The command creates a directory with UUID that contains the following:
- Directory generated by the cuDF plugin qualification tool `rapids_4_spark_qualification_output`;
- A CSV file that contains the summary of all the applications along with estimated absolute costs
- Sample directory structure:
    ```
    qual_20230504205509_77Ae613C
    ├── rapids_4_spark_qualification_output
    │   ├── rapids_4_spark_qualification_output.log
    │   ├── rapids_4_spark_qualification_output_stages.csv
    │   ├── ui
    │   │   └── html
    │   │       ├── index.html
    │   │       ├── sql-recommendation.html
    │   │       ├── application.html
    │   │       └── raw.html
    │   ├── rapids_4_spark_qualification_output_execs.csv
    │   └── rapids_4_spark_qualification_output.csv
    └── qualification_summary.csv
    3 directories, 9 files
    ```
### Use case scenario to run qualification tool to get savings on cloud platform based on on-premises cluster config

A typical workflow to successfully run the `qualification` command in local mode to get savings for cloud platform
is described as follows:

1. Store the Apache Spark event logs in local directory.
2. On a machine with JDK8+ installed:
    1. user installs `spark_rapids_user_tools`
3. User defines the cluster configuration of on-premises platform. Template of the required configs is provided below and
   the file should be in yaml format.
4. User specifies the target_platform for which the cost savings and speedup recommendations are required. Currently,
   only `dataproc` platform is supported. We do best match effort based on the number of cores provided in the yaml file
   for on-premises cluster to the GPU supported cluster. Format of the yaml file is as below:
   ```
   config:
    masterConfig:
      numCores: 2
      memory: 7680MiB
    workerConfig:
      numCores: 8
      memory: 7680MiB
      numWorkers: 2
   ```
   The following script runs qualification tool locally:

   ```
   # define the wrapper cache directory if necessary
   export RAPIDS_USER_TOOLS_CACHE_FOLDER=my_cache_folder
   export EVENTLOGS=local_path_to_eventlogs
   export CLUSTER_CONIFG_PROPERTY_FILE=my-cluster-config-file.yaml

   spark_rapids_user_tools onprem qualification --target_platform=dataproc \
      --eventlogs $EVENTLOGS \
      --cpu_cluster $CLUSTER_PROPERTY_FILE
   ```
   The wrapper generates a unique-Id for each execution in the format of `qual_<YYYYmmddHHmmss>_<0x%08X>`
   The above command will generate a directory containing `qualification_summary.csv` in addition to
   the actual folder of the cuDF plugin Qualification tool.

   ```
    ./qual_<YYYYmmddHHmmss>_<0x%08X>/qualification_summary.csv
    ./qual_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_qualification_output/
   ```

#### Qualification output

For each app, the command output lists the following fields:

- `App ID`: An application is referenced by its application ID, '_app-id_'. When running on YARN,
  each application may have multiple attempts, but there are attempt IDs only for applications
  in cluster mode, not applications in client mode.  Applications in YARN cluster mode can be
  identified by their attempt-id.
- `App Name`: Name of the application
- `Speedup Based Recommendation`: Recommendation based on '_Estimated Speed-up Factor_'. Note that an
  application that has job or stage failures will be labeled '_Not Applicable_'
- `Savings Based Recommendation`: Recommendation based on '_Estimated GPU Savings_'.
    - '_Strongly Recommended_': An app with savings GEQ 40%
    - '_Recommended_': An app with savings between (1, 40) %
    - '_Not Recommended_': An app with no savings
    - '_Not Applicable_': An app that has job or stage failures.
- `App Duration(s)`: Wall-Clock time measured since the application starts till it is completed.
  If an app is not completed an estimated completion time would be computed.
- `Estimated GPU Duration(s)`: Predicted runtime of the app if it was run on GPU
- `Estimated GPU Speedup`: Speed-up factor estimated for the app. Calculated as the ratio
  between '_App Duration_' and '_Estimated GPU Duration_'.
- `Estimated GPU Savings(%)`: Percentage of cost savings of the app if it migrates to an
  accelerated cluster of the target_platform. It is calculated as:
  ```
  estimated_saving = 100 - ((100 * gpu_cost) / cpu_cost)

The command creates a directory with UUID that contains the following:
- Directory generated by the cuDF plugin qualification tool `rapids_4_spark_qualification_output`;
- A CSV file that contains the summary of all the applications along with estimated absolute costs
- Sample directory structure:
    ```
    qual_20230601222103_AaA7eb99
    ├── rapids_4_spark_qualification_output
    │   ├── rapids_4_spark_qualification_output.log
    │   ├── rapids_4_spark_qualification_output_stages.csv
    │   ├── ui
    │   │   └── html
    │   │       ├── index.html
    │   │       ├── sql-recommendation.html
    │   │       ├── application.html
    │   │       └── raw.html
    │   ├── rapids_4_spark_qualification_output_execs.csv
    │   └── rapids_4_spark_qualification_output.csv
    └── qualification_summary.csv
    3 directories, 9 files
    ```

## Profiling command

### Local deployment

```
spark_rapids_user_tools onprem profiling [options]
spark_rapids_user_tools onprem profiling -- --help
```

The local deployment runs on the local development machine. It requires:
1. Java 1.8+ development environment.
2. Internet access to download JAR dependencies from mvn: `spark-*.jar`.
3. Dependencies are cached on the local disk to reduce the overhead of the download.

#### Command options

| Option               | Description                                                                                                                                                                                                                                               | Default                                                                                                                                       | Required |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|----------|

| **eventlogs**        | A comma separated list to event logs or directory                                                                                                                                                                                                         | None                                                                                                                                          | N        |
| **local_folder**     | Local work-directory path to store the output and to be used as root directory for temporary folders/files. The final output will go into a subdirectory named `prof-${EXEC_ID}` where `exec_id` is an auto-generated unique identifier of the execution. | If the argument is NONE, the default value is the env variable `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY` if any; or the current working directory. | N        |
| **jvm_heap_size**    | The maximum heap size of the JVM in gigabytes                                                                                                                                                                                                             | 24                                                                                                                                            | N        |
| **tools_jar**        | Path to a bundled jar including cuDF plugin tool. The path is a local filesystem.                                                                                                                                                                              | Downloads the latest `rapids-4-spark-tools_*.jar` from mvn repo                                                                               | N        |
| **verbose**          | True or False to enable verbosity to the wrapper script                                                                                                                                                                                                   | False if `RAPIDS_USER_TOOLS_LOG_DEBUG` is not set                                                                                             | N        |
| **rapids_options**** | A list of valid [Profiling tool options](https://docs.nvidia.com/spark-rapids/user-guide/latest/profiling/jar-usage.html#prof-tool-title-options). Note that (`output-directory`, `auto-tuner`, `combined`) flags are ignored                                                                  | N/A                                                                                                                                           | N        |

The tool will generate recommendations based on the job event log.

#### Use case scenario

A typical workflow to successfully run the `profiling` command in local mode is described as follows:

1. Store the Apache Spark event logs (local folder).
2. On a machine with JDK8+ installed:
    1. user installs `spark_rapids_user_tools`
3. User can optionally define the cluster configuration of on-premises platform if needed for enhanced recommendations.
4. User runs profiling tool CLI command.

For each successful execution, the wrapper generates a new directory in the format of
`prof_<YYYYmmddHHmmss>_<0x%08X>`. The directory contains `tuning_summary.log` in addition to
the actual folder of the cuDF plugin Profiling tool.

   ```
    ./prof_<YYYYmmddHHmmss>_<0x%08X>/tuning_summary.log
    ./prof_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_profile/
   ```

The profiling tool can be run with just the event logs:

    ```
    # Run the profiling cmd
    $> spark_rapids_user_tools onprem profiling \
            --eventlogs $EVENTLOGS
    ```
