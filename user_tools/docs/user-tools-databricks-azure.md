# NVIDIA cuDF plugin for Apache Spark User Tools on Databricks Azure

This is a guide for the tools for the NVIDIA cuDF plugin for Apache Spark on [Databricks Azure](https://www.databricks.com/product/azure). At the end of this guide, the user will be able to run the cuDF plugin tools to analyze the clusters and the applications running on Databricks Azure.

## Assumptions

The tools read Apache Spark event logs from a local path or from ABFS ([Azure Blob File System](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction-abfs-uri)). ABFS paths must use the full form `abfss://<container>@<storage-account>.dfs.core.windows.net/<path>`. The output of the tools is written to a local directory.

## Prerequisites

### 1.Databricks CLI

- Install the Databricks CLI version 0.200+. Follow the instructions on [Install the CLI](https://docs.databricks.com/en/dev-tools/cli/install.html).
- Set the configuration settings and credentials of the Databricks CLI:
  - Set up authentication by following these [instructions](https://docs.databricks.com/en/dev-tools/cli/authentication.html)
  - Verify that the access credentials are stored in the file `~/.databrickscfg` on Unix, Linux, or macOS, or in another file defined by environment variable `DATABRICKS_CONFIG_FILE`.

### 2.Azure CLI

- Install the Azure CLI. Follow the instructions on [How to install the Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
- Set the configuration settings and credentials of the Azure CLI:
  - Set up the authentication by following these [instructions](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli).
  - Configure the Azure CLI by following these [instructions](https://learn.microsoft.com/en-us/cli/azure/azure-cli-configuration).
    - `location` is used for retrieving instance type description (default is `westus`).
    - `output` should use default of `json` in `core` section.
    - Verify that the configurations are stored in the file `$AZURE_CONFIG_DIR/config` where the default value of `AZURE_CONFIG_DIR` is `$HOME/.azure` on Linux or macOS.
- The tools use the Azure CLI to read the storage account key for ABFS event logs. See [Accessing ABFS event logs](#accessing-abfs-event-logs) for the permissions and network access this requires.

### 3.cuDF plugin tools

- Spark event logs:
  - The cuDF plugin tools can process Apache Spark CPU event logs from Spark 2.0 or higher (raw, .lz4, .lzf, .snappy, .zstd).
  - For `qualification` commands, the event logs need to be archived to an accessible local or ABFS folder.

### 4.Install the package

- Install `spark-rapids-user-tools` with python [3.10, 3.12] using:
  - pip:  `pip install spark-rapids-user-tools`
  - wheel-file: `pip install <wheel-file>`
  - from source: `pip install -e .`
- Verify the command is installed correctly by running
  ```bash
    spark_rapids -- --help
  ```

### 5.Environment variables

Before running any command, you can set environment variables to specify configurations.
- User tools variables use the `RAPIDS_USER_TOOLS_*` naming pattern:
  - `RAPIDS_USER_TOOLS_CACHE_FOLDER`: specifies the location of a local directory that the cuDF plugin CLI uses to store and cache the downloaded resources. The default is `/var/tmp/spark_rapids_user_tools_cache`.  Note that caching the resources locally has an impact on the total execution time of the command.
  - `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY`: specifies the location of a local directory that the cuDF plugin CLI uses to generate the output. The CLI argument `--output_folder` overrides that environment variable.
- For Databricks CLI, some environment variables can be set and picked up by the cuDF plugin user tools such as: `DATABRICKS_CONFIG_FILE`, `DATABRICKS_HOST` and `DATABRICKS_TOKEN`. See the description of the variables in [Environment variables](https://docs.databricks.com/en/dev-tools/auth/index.html#environment-variables-and-fields-for-client-unified-authentication).
- For Azure CLI, some environment variables can be set and picked up by the cuDF plugin user tools such as: `AZURE_CONFIG_FILE` and `AZURE_DEFAULTS_LOCATION`.

## Qualification command

### Local deployment

```
spark_rapids qualification --platform databricks-azure [options]
spark_rapids qualification -- --help
```

The local deployment runs on the local development machine. It requires:
1. Installing and configuring the Databricks and Azure CLI
2. Java 1.8+ development environment (Java 23 fails with the Hadoop libraries used by the tools; see [Troubleshooting](#troubleshooting))
3. Internet access to download JAR dependencies from mvn: `spark-*.jar` and `hadoop-azure-*.jar`
4. Dependencies are cached on the local disk to reduce the overhead of the download.

#### Command options

| Option                 | Description                                                                                                                                                                                                                                   | Default                                                                                                                         | Required |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|:--------:|
| **platform** (`-p`)    | The platform on which the applications were executed. Use `databricks-azure`                                                                                                                                                                  | N/A                                                                                                                             |    Y     |
| **eventlogs** (`-e`)   | Event log filenames, ABFS or local directories containing event logs (comma separated), or path to a TXT file containing a list of event log paths. ABFS paths must use the form `abfss://<container>@<storage-account>.dfs.core.windows.net/<path>` | If missing, the `cluster` argument must point to a valid Databricks cluster                                                     |    N     |
| **cluster**            | The Databricks cluster on which the Apache Spark applications were executed. Accepted values are a Databricks cluster id, or a valid path to the cluster properties file (json format) generated by Databricks CLI command `databricks clusters get CLUSTER_ID [flags]` | N/A                                                                                                                             |    N     |
| **output_folder** (`-o`) | Local path to store the output. The final output goes into a subdirectory named `qual_<YYYYmmddHHmmss>_<0x%08X>`                                                                                                                             | The env variable `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY` if any; or the current working directory                                  |    N     |
| **filter_apps** (`-f`) | Filtering criteria of the applications listed in the final STDOUT table: `ALL` or `TOP_CANDIDATES`. `ALL` means no filter applied. `TOP_CANDIDATES` lists the apps whose unsupported-operator stage duration is less than 25% of the app duration and whose estimated speedup is greater than 1.3x. The filter does not affect the CSV report | `TOP_CANDIDATES`                                                                                                                |    N     |
| **custom_model_file**  | An optional local path to a custom XGBoost model file                                                                                                                                                                                         | N/A                                                                                                                             |    N     |
| **tools_jar**          | Path to a bundled jar including cuDF plugin tool. The path is a local filesystem, or remote ABFS url                                                                                                                                          | Downloads the latest `rapids-4-spark-tools_*.jar` from mvn repo                                                                 |    N     |
| **jvm_heap_size**      | The maximum heap size of the JVM in gigabytes                                                                                                                                                                                                 | Calculated from the total memory of the host                                                                                    |    N     |
| **jvm_threads**        | Number of threads to use for parallel processing on the eventlogs batch                                                                                                                                                                       | Calculated from the number of cores and the heap size of the host                                                               |    N     |
| **submission_mode** (`-s`) | Submission mode to run the qualification tool: `local` or `distributed`                                                                                                                                                                   | `local`                                                                                                                         |    N     |
| **tools_config_file**  | Path to a configuration file that contains the tools' options. See the [sample configuration files](https://github.com/NVIDIA/cudf-spark-tools/tree/main/user_tools/tests/spark_rapids_tools_ut/resources/tools_config/valid)                 | N/A                                                                                                                             |    N     |
| **target_cluster_info** | Path to a YAML file that contains the target cluster information. See the [sample files](https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/docs/sampleFiles/targetClusterInfo)                                                      | N/A                                                                                                                             |    N     |
| **tuning_configs**     | Path to a YAML file that contains the tuning configurations. See the [sample file](https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/src/main/resources/bootstrap/tuningConfigs.yaml)                                                  | N/A                                                                                                                             |    N     |
| **qualx_config**       | Path to a `qualx-conf.yaml` file to use for configuration                                                                                                                                                                                     | The [default qualx-conf.yaml](https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-conf.yaml) |    N     |
| **verbose** (`-v`)     | True or False to enable verbosity of the script                                                                                                                                                                                               | False if `RAPIDS_USER_TOOLS_LOG_DEBUG` is not set                                                                               |    N     |
| **rapids_options****   | A list of valid [Qualification tool options](https://docs.nvidia.com/spark-rapids/user-guide/latest/qualification/jar-usage.html#running-the-qualification-tool-standalone-on-spark-event-logs). Note that (`output-directory`, `platform`) flags are ignored, and that multiple "spark-property" is not supported. | N/A                                                                                                                             |    N     |

#### Use case scenario

A typical workflow to successfully run the `qualification` command in local mode is described as follows:

1. Store the Apache Spark event logs in an ABFS folder or a local folder.
2. A user sets up his development machine:
   1. configures Java
   2. installs Databricks CLI and configures the profile and the credentials to make sure the
      access credentials are stored in the file `~/.databrickscfg` on Unix, Linux, or macOS,
      or in another file defined by environment variable `DATABRICKS_CONFIG_FILE`.
   3. installs Azure CLI and configures the credentials so that the tools can read the storage account key
      of the account that stores the event logs (see [Accessing ABFS event logs](#accessing-abfs-event-logs)).
   4. installs `spark-rapids-user-tools`
3. User defines the Databricks cluster on which the Spark applications were running. Note that the cluster does not have to be active; but it has to be visible by the Databricks CLI (i.e., can run `databricks clusters get CLUSTER_ID`).
4. The following script runs qualification and stores the output in a local folder:

   ```
   # define the wrapper cache directory if necessary
   export RAPIDS_USER_TOOLS_CACHE_FOLDER=my_cache_folder
   export EVENTLOGS=abfss://<container>@<storage-account>.dfs.core.windows.net/eventlogs/
   export CLUSTER_ID=my-databricks-cpu-cluster-id

   spark_rapids qualification --platform databricks-azure \
      --eventlogs $EVENTLOGS \
      --cluster $CLUSTER_ID \
      --output_folder ./qual_output
   ```
   The wrapper generates a unique-Id for each execution in the format of `qual_<YYYYmmddHHmmss>_<0x%08X>`.
   The above command generates a directory containing `qualification_summary.csv` in addition to
   the output of the cuDF plugin Qualification tool in `qual_core_output`:

   ```
    ./qual_output/qual_<YYYYmmddHHmmss>_<0x%08X>/qualification_summary.csv
    ./qual_output/qual_<YYYYmmddHHmmss>_<0x%08X>/qual_core_output/
   ```

### Qualification output

For each app, `qualification_summary.csv` includes the following fields:

- `App ID`: An application is referenced by its application ID, '_app-id_'. When running on YARN,
  each application may have multiple attempts, but there are attempt IDs only for applications
  in cluster mode, not applications in client mode.  Applications in YARN cluster mode can be
  identified by their attempt-id.
- `App Name`: Name of the application
- `Estimated GPU Speedup Category`: The category of the estimated speedup of the app.
- `Estimated GPU Speedup`: Speed-up factor estimated for the app. Calculated as the ratio
  between '_App Duration_' and '_Estimated GPU Duration_'.
- `Estimated GPU Duration`: Predicted runtime of the app if it was run on GPU
- `App Duration`: Wall-Clock time measured since the application starts till it is completed.
  If an app is not completed an estimated completion time would be computed.

The command creates a directory with a unique ID that contains the following:
- `qualification_summary.csv`: the summary of all the applications;
- `qual_core_output`: the directory generated by the cuDF plugin qualification tool, including `status.csv`, which reports
  whether each event log was processed successfully;
- `qual_core_stderr.log`: the log of the cuDF plugin qualification tool.
- Sample directory structure:
    ```
    .
    ├── qualification_summary.csv
    ├── qualification_statistics.csv
    ├── qual_core_stderr.log
    └── qual_core_output
        ├── apps_summary.csv
        ├── status.csv
        ├── qual_metrics
        ├── raw_metrics
        └── tuning
    ```

## Profiling command

### Local deployment

```
spark_rapids profiling --platform databricks-azure [options]
spark_rapids profiling -- --help
```

The local deployment runs on the local development machine. It requires:
1. Installing and configuring the Databricks and Azure CLI
2. Java 1.8+ development environment (Java 23 fails with the Hadoop libraries used by the tools; see [Troubleshooting](#troubleshooting))
3. Internet access to download JAR dependencies from mvn: `spark-*.jar`, and `hadoop-azure-*.jar`
4. Dependencies are cached on the local disk to reduce the overhead of the download.

#### Command options

| Option                   | Description                                                                                                                                                                                                                                                            | Default                                                                                                                                       | Required |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
| **platform** (`-p`)      | The platform on which the applications were executed. Use `databricks-azure`                                                                                                                                                                                           | N/A                                                                                                                                           |    Y     |
| **eventlogs** (`-e`)     | Event log filenames, ABFS or local directories containing event logs (comma separated), or path to a TXT file containing a list of event log paths. ABFS paths must use the form `abfss://<container>@<storage-account>.dfs.core.windows.net/<path>`                   | Reads the Spark's property `spark.eventLog.dir` defined in `cluster`                                                                         |    N     |
| **cluster** (`-c`)       | The Databricks cluster on which the Apache Spark applications were executed. Accepted values are a Databricks cluster id, or a valid path to the cluster properties file (json format) generated by Databricks CLI command `databricks clusters get CLUSTER_ID [flags]` | N/A                                                                                                                                           |    N     |
| **driverlog** (`-d`)     | Valid path to the GPU driver log file                                                                                                                                                                                                                                  | N/A                                                                                                                                           |    N     |
| **output_folder** (`-o`) | Local path to store the output. The final output goes into a subdirectory named `prof_<YYYYmmddHHmmss>_<0x%08X>`                                                                                                                                                       | The env variable `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY` if any; or the current working directory                                                |    N     |
| **tools_jar**            | Path to a bundled jar including cuDF plugin tool. The path is a local filesystem, or remote ABFS url                                                                                                                                                                   | Downloads the latest `rapids-4-spark-tools_*.jar` from mvn repo                                                                               |    N     |
| **jvm_heap_size**        | The maximum heap size of the JVM in gigabytes                                                                                                                                                                                                                          | Calculated from the total memory of the host                                                                                                  |    N     |
| **jvm_threads**          | Number of threads to use for parallel processing on the eventlogs batch                                                                                                                                                                                                | Calculated from the number of cores and the heap size of the host                                                                             |    N     |
| **tools_config_file**    | Path to a configuration file that contains the tools' options. See the [sample configuration files](https://github.com/NVIDIA/cudf-spark-tools/tree/main/user_tools/tests/spark_rapids_tools_ut/resources/tools_config/valid)                                          | N/A                                                                                                                                           |    N     |
| **target_cluster_info**  | Path to a YAML file that contains the target cluster information. See the [sample files](https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/docs/sampleFiles/targetClusterInfo)                                                                               | N/A                                                                                                                                           |    N     |
| **tuning_configs**       | Path to a YAML file that contains the tuning configurations. See the [sample file](https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/src/main/resources/bootstrap/tuningConfigs.yaml)                                                                           | N/A                                                                                                                                           |    N     |
| **verbose** (`-v`)       | True or False to enable verbosity of the script                                                                                                                                                                                                                        | False if `RAPIDS_USER_TOOLS_LOG_DEBUG` is not set                                                                                             |    N     |
| **rapids_options****     | A list of valid [Profiling tool options](https://docs.nvidia.com/spark-rapids/user-guide/latest/profiling/jar-usage.html#prof-tool-title-options). Note that (`output-directory`) flags are ignored, and that multiple "spark-property" is not supported               | N/A                                                                                                                                           |    N     |

If the CLI does not provide an argument `cluster`, the autotuner will still recommend tuning settings based on the job event log.

#### Use case scenario

A typical workflow to successfully run the `profiling` command in local mode is described as follows:

1. Store the Apache Spark event logs in an ABFS folder or a local folder.
2. A user sets up his development machine:
   1. configures Java
   2. installs Databricks CLI and configures the profile and the credentials to make sure the
      access credentials are stored in the file `~/.databrickscfg` on Unix, Linux, or macOS,
      or in another file defined by environment variable `DATABRICKS_CONFIG_FILE`.
   3. installs Azure CLI and configures the credentials so that the tools can read the storage account key
      of the account that stores the event logs (see [Accessing ABFS event logs](#accessing-abfs-event-logs)).
   4. installs `spark-rapids-user-tools`
3. Depending on the accessibility of the cluster properties, the user chooses one of the 2 cases below (_"Case-A"_, and _"Case-B"_) to trigger the CLI.

For each successful execution, the wrapper generates a new directory in the format of
`prof_<YYYYmmddHHmmss>_<0x%08X>`. The directory contains `tuning_summary.log` in addition to
the output of the cuDF plugin Profiling tool:

   ```
    ./prof_<YYYYmmddHHmmss>_<0x%08X>/tuning_summary.log
    ./prof_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_profile/
   ```

The `rapids_4_spark_profile` directory includes `profiling_status.csv`, which reports whether each event log
was processed successfully, and `prof_core_stderr.log`, the log of the cuDF plugin profiling tool.

**Case-A: A cluster property file is accessible:**

A cluster property is still accessible if one of the following conditions applies:

1. The cluster is listed by the `databricks clusters get CLUSTER_ID [flags]` cmd. In this case, the CLI will be triggered by providing
   `--cluster $CLUSTER_ID`

   ```
   # run the command using the cluster id
   export RAPIDS_USER_TOOLS_CACHE_FOLDER=my_cache_folder
   export EVENTLOGS=abfss://<container>@<storage-account>.dfs.core.windows.net/eventlogs/
   export CLUSTER_ID=my-databricks-gpu-cluster-id

   spark_rapids profiling --platform databricks-azure \
      --eventlogs $EVENTLOGS \
      --cluster $CLUSTER_ID \
      --output_folder ./prof_output
   ```
2. The cluster properties file is accessible on local disk or a valid ABFS path.

   ```
   $> export CLUSTER_PROPS_FILE=cluster-props.json
   $> databricks clusters get $CLUSTER_ID > $CLUSTER_PROPS_FILE
   ```
   Trigger the CLI by providing the path to the properties file `--cluster $CLUSTER_PROPS_FILE`

   ```
   $> spark_rapids profiling --platform databricks-azure \
        --eventlogs $EVENTLOGS \
        --cluster $CLUSTER_PROPS_FILE \
        --output_folder ./prof_output
   ```

**Case-B: Cluster information is missing:**

In this scenario, if you have no access to the properties of the cluster, the autotuner will still recommend tuning settings based on the job event log:

   ```
   # Run the profiling cmd
   $> spark_rapids profiling --platform databricks-azure \
        --eventlogs $EVENTLOGS \
        --output_folder ./prof_output
   ```

## Accessing ABFS event logs

When an event log path starts with `abfss://`, the tools read the event logs directly from the Azure storage account.

### How the tools authenticate

1. The tools take the storage account name from the first `abfss://` path in `--eventlogs`. The account name is the part
   between `@` and `.dfs.core.windows.net`, so the path must use the full form
   `abfss://<container>@<storage-account>.dfs.core.windows.net/<path>`.
2. The tools run `az storage account show-connection-string --name <storage-account>` and extract the account key.
3. The key is passed to the JVM as the system property
   `rapids.tools.hadoop.fs.azure.account.key.<storage-account>.dfs.core.windows.net`, and the cuDF plugin tool
   copies it into the Hadoop configuration used by the `hadoop-azure` ABFS driver.

This has the following requirements:

- **Permission to list the account keys.** The Azure identity used by the Azure CLI needs the
  `Microsoft.Storage/storageAccounts/listKeys/action` permission on the storage account, for example through the
  `Contributor` or the `Storage Account Key Operator Service Role` role. Data-plane roles such as
  `Storage Blob Data Reader` alone are not enough. See [Manage storage account access keys](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-keys-manage)
  and [Azure built-in roles for Storage](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles/storage).
- **Shared Key access must be allowed** on the storage account. See
  [Prevent Shared Key authorization](https://learn.microsoft.com/en-us/azure/storage/common/shared-key-authorization-prevent).
- **The storage account must be in the active Azure CLI subscription** (`az account show`).
- **All ABFS event logs in a single run must be in the same storage account.** Only the first `abfss://` path is used to
  look up the key.

If the key lookup fails, the tools log the error at `INFO` level and continue without a key. The run then fails to read
the event logs later (see [Troubleshooting](#troubleshooting)).

Note that the key is passed as a JVM `-D` argument, so it is visible in the process list of the host while the tools run.
The tools' log files mask the key.

### Network access

If the storage account firewall is set to deny public network access by default, the host running the tools must be
allowed to reach the account, for example with an IP rule for the host's public IP address, a virtual network rule, or a
private endpoint. Azure RBAC roles do not bypass the firewall. See
[Azure Storage firewall and virtual network rules](https://learn.microsoft.com/en-us/azure/storage/common/storage-network-security).

### Alternatives when key or network access is not possible

- Copy the event logs to a local folder (for example with the Azure CLI or AzCopy) and point
  `--eventlogs` at the local path. No storage account key is needed in this case.
- Run the tools from inside Databricks with the
  [notebook templates](https://github.com/NVIDIA/cudf-spark-examples/tree/main/tools/databricks).

## Troubleshooting

When an event log cannot be read, the command still prints `Processing Completed!`, but no application is processed. Check
the status file and the tool log in the output directory:

- Qualification: `qual_<YYYYmmddHHmmss>_<0x%08X>/qual_core_output/status.csv` and `qual_<YYYYmmddHHmmss>_<0x%08X>/qual_core_stderr.log`
- Profiling: `prof_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_profile/profiling_status.csv` and `prof_<YYYYmmddHHmmss>_<0x%08X>/rapids_4_spark_profile/prof_core_stderr.log`

A `FAILURE` row in the status file includes the reason. Common errors:

| Error                                                                                                                                     | Cause                                                                                                         | Fix                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `IndexError: list index out of range` while submitting the job                                                                            | The ABFS path does not include `@<storage-account>.dfs.core.windows.net`                                       | Use the full form `abfss://<container>@<storage-account>.dfs.core.windows.net/<path>`                                 |
| `KeyProviderException: Failure to initialize configuration`                                                                               | No storage account key was passed to the JVM, because the `az storage account show-connection-string` call failed | Check that the Azure CLI is installed and logged in, that the account is in the active subscription, and that you have the `listKeys` permission. The key lookup error is written to the application log file printed at the end of the run (`~/.spark_rapids_tools/logs/<run id>.log`) |
| `AccessDeniedException: Operation failed: "This request is not authorized to perform this operation.", 403, HEAD, https://<storage-account>.dfs.core.windows.net/...` | The storage account firewall blocks the host running the tools                                                | Allow the host's network access to the storage account (see [Network access](#network-access)), or use a local copy of the event logs |
| `UnsupportedOperationException: getSubject is supported only if a security manager is allowed`                                            | The JVM is too new for the Hadoop libraries used by the tools (seen with Java 23)                              | Use Java 8, 11 or 17                                                                                                  |

## Diagnostic command

```
spark_rapids_user_tools databricks-azure diagnostic [options]
spark_rapids_user_tools databricks-azure diagnostic --help
```

Run diagnostic command to collects information from Databricks cluster, such as OS version, # of worker
nodes, Yarn configuration, Spark version and error logs etc. The cluster has to be running and the
user must have SSH access.

### Diagnostic options

| Option            | Description                                                                                               | Default                                                                                     | Required |
|-------------------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|:--------:|
| **cluster**       | Id of the Databricks cluster running an accelerated computing instance                                    | N/A                                                                                         |    Y     |
| **profile**       | A named Databricks profile that you can specify to get the settings/credentials of the Databricks account | "DEFAULT"                                                                                   |    N     |
| **output_folder** | Path to local directory where the final recommendations is logged                                         | env variable `RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY` if any; or the current working directory. |    N     |
| **port**          | Port number to be used for the ssh connections                                                            | 2200                                                                                        |    N     |
| **key_file**      | Path to the private key file to be used for the ssh connections.                                          | Default ssh key based on the OS                                                             |    N     |
| **thread_num**    | Number of threads to access remote cluster nodes in parallel                                              | 3                                                                                           |    N     |
| **yes**           | auto confirm to interactive question                                                                      | False                                                                                       |    N     |
| **verbose**       | True or False to enable verbosity to the wrapper script                                                   | False if `RAPIDS_USER_TOOLS_LOG_DEBUG` is not set                                                                           |    N     |

### Info collection

The default is to collect info from each cluster node via SSH access and an archive would be created
to output folder at last.

The steps to run the command:

1. The user creates a cluster
2. The user runs the following command:

    ```bash
    spark_rapids_user_tools databricks-azure diagnostic \
      --cluster my-cluster-id
    ```

If the connection to Databricks instances cannot be established through SSH, the command will raise error.
