# Copyright (c) 2023-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI to run tools associated with the NVIDIA cuDF plugin for Apache Spark."""

from typing import Optional

import fire

from spark_rapids_pytools.common.utilities import Utils, ToolLogging
from spark_rapids_pytools.rapids.profiling import ProfilingAsLocal
from spark_rapids_pytools.rapids.qualification import QualificationAsLocal
from spark_rapids_pytools.rapids.qualx.prediction import Prediction
from spark_rapids_pytools.rapids.qualx.train import Train
from spark_rapids_pytools.rapids.qualx.train_and_evaluate import TrainAndEvaluate
from spark_rapids_tools.cmdli.argprocessor import AbsToolUserArgModel
from spark_rapids_tools.enums import CspEnv, QualEstimationModel
from spark_rapids_tools.utils.util import gen_app_banner, init_environment


class ToolsCLI(object):  # pylint: disable=too-few-public-methods
    """CLI that provides a runtime environment that simplifies running performance analysis using
    the cuDF plugin.

    A wrapper script to run cuDF plugin tools locally on the dev machine.
    """

    def qualification(self,
                      eventlogs: str = None,
                      cluster: str = None,
                      platform: str = None,
                      output_folder: str = None,
                      filter_apps: str = None,
                      custom_model_file: str = None,
                      tools_jar: str = None,
                      jvm_heap_size: int = None,
                      jvm_threads: int = None,
                      verbose: bool = None,
                      tools_config_file: str = None,
                      submission_mode: str = None,
                      target_cluster_info: str = None,
                      tuning_configs: str = None,
                      qualx_config: str = None,
                      **rapids_options) -> Optional[str]:
        """The Qualification cmd provides estimated speedups by migrating Apache Spark applications
        to GPU accelerated clusters.

        The Qualification cmd analyzes Spark eventlogs generated from CPU based Spark applications to
        help quantify the expected acceleration of migrating a Spark application or query to GPU.
        The cmd will process each app individually, but will group apps with the same name into the
        same output row after averaging duration metrics accordingly.

        :param eventlogs: Event log filenames, CSP storage directories containing event logs
                (comma separated), or path to a TXT file containing a list of event log paths.

                Skipping this argument requires that the cluster argument points to a valid
                cluster name on the CSP.
        :param cluster: The CPU cluster on which the Spark application(s) were executed.
               Name or ID (for databricks platforms) of cluster or path to cluster-properties.
        :param platform: Defines one of the following: "onprem", "emr", "dataproc", "dataproc-gke",
               "databricks-aws", and "databricks-azure".
        :param output_folder: Local path to store the output.
        :param tools_jar: Path to a bundled jar including Rapids tool. The path is a local filesystem,
                or remote cloud storage url. If missing, the wrapper downloads the latest rapids-4-spark-tools_*.jar
                from maven repository.
        :param filter_apps: Filtering criteria of the applications listed in the final STDOUT table,
                is one of the following ("ALL", "TOP_CANDIDATES"). Default is "TOP_CANDIDATES".

                Note that this filter does not affect the CSV report.
                "ALL" means no filter applied. "TOP_CANDIDATES" lists all apps that have unsupported operators
                stage duration less than 25% of app duration and speedups greater than 1.3x.
        :param custom_model_file: An optional local path to a custom XGBoost model file.
        :param jvm_heap_size: The maximum heap size of the JVM in gigabytes.
                Default is calculated based on a function of the total memory of the host.
        :param jvm_threads: Number of threads to use for parallel processing on the eventlogs batch.
                Default is calculated as a function of the total number of cores and the heap size on the host.
        :param verbose: True or False to enable verbosity of the script.
        :param tools_config_file: Path to a configuration file that contains the tools' options.
                For sample configuration files, please visit
                https://github.com/NVIDIA/cudf-spark-tools/tree/main/user_tools/tests/spark_rapids_tools_ut/resources/tools_config/valid
        :param submission_mode: Submission mode to run the qualification tool.
                Supported modes are "local" and "distributed".
        :param target_cluster_info: Path to a YAML file that contains the target cluster information.
                For sample target cluster info files, please visit
                https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/src/main/resources/targetClusterInfo
        :param rapids_options: A list of valid Qualification tool options.
                Note that the wrapper ignores ["output-directory", "platform"] flags, and it does not support
                multiple "spark-property" arguments.
                For more details on Qualification tool options, please visit
                https://docs.nvidia.com/spark-rapids/user-guide/latest/qualification/jar-usage.html#running-the-qualification-tool-standalone-on-spark-event-logs
        :param tuning_configs: Path to a YAML file that contains the tuning configurations.
                For sample tuning configs files, please visit
                https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/src/main/resources/bootstrap/tuningConfigs.yaml
        :param qualx_config: Path to a qualx-conf.yaml file to use for configuration.
               If not provided, the wrapper will use the default:
               https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-conf.yaml.
        :return: The output folder where the qualification tool output is stored.
        :rtype: Optional[str]
        """
        eventlogs = Utils.get_value_or_pop(eventlogs, rapids_options, 'e')
        platform = Utils.get_value_or_pop(platform, rapids_options, 'p')
        output_folder = Utils.get_value_or_pop(output_folder, rapids_options, 'o')
        filter_apps = Utils.get_value_or_pop(filter_apps, rapids_options, 'f')
        verbose = Utils.get_value_or_pop(verbose, rapids_options, 'v', False)
        submission_mode = Utils.get_value_or_pop(submission_mode, rapids_options, 's')
        if verbose:
            ToolLogging.enable_debug_mode()
        init_environment('qual')
        estimation_arg_valid = {
            'toolName': 'qualification',
            'validatorName': 'estimation_model_args'
        }
        estimation_model_args = AbsToolUserArgModel.create_tool_args(estimation_arg_valid,
                                                                     estimation_model=None,
                                                                     custom_model_file=custom_model_file,
                                                                     qualx_config=qualx_config)
        if estimation_model_args is None:
            return None
        qual_args = AbsToolUserArgModel.create_tool_args('qualification',
                                                         eventlogs=eventlogs,
                                                         cluster=cluster,
                                                         platform=platform,
                                                         output_folder=output_folder,
                                                         tools_jar=tools_jar,
                                                         jvm_heap_size=jvm_heap_size,
                                                         jvm_threads=jvm_threads,
                                                         filter_apps=filter_apps,
                                                         estimation_model_args=estimation_model_args,
                                                         tools_config_path=tools_config_file,
                                                         submission_mode=submission_mode,
                                                         target_cluster_info=target_cluster_info,
                                                         tuning_configs=tuning_configs)
        if qual_args:
            rapids_options.update(qual_args['rapidOptions'])
            tool_obj = QualificationAsLocal(platform_type=qual_args['runtimePlatform'],
                                            output_folder=qual_args['outputFolder'],
                                            wrapper_options=qual_args,
                                            rapids_options=rapids_options)
            tool_obj.launch()
            return tool_obj.csp_output_path
        return None

    def profiling(self,
                  eventlogs: str = None,
                  cluster: str = None,
                  platform: str = None,
                  driverlog: str = None,
                  output_folder: str = None,
                  tools_jar: str = None,
                  jvm_heap_size: int = None,
                  jvm_threads: int = None,
                  verbose: bool = None,
                  tools_config_file: str = None,
                  target_cluster_info: str = None,
                  tuning_configs: str = None,
                  **rapids_options) -> Optional[str]:
        """The Profiling cmd provides information which can be used for debugging and profiling
        Apache Spark applications running on GPU accelerated clusters.

        The Profiling tool analyzes both CPU or GPU generated eventlogs and generates information
        including the Spark version, executor details, properties, etc.
        The tool also will recommend setting for the application assuming that the job will be able
        to use all the cluster resources (CPU and GPU) when it is running.

        :param eventlogs: Event log filenames, cloud storage directories
                containing event logs (comma separated), or path to a TXT file containing a list of event log paths.
                If missing, the wrapper reads the Spark's
                property `spark.eventLog.dir` defined in the `cluster`.
        :param cluster: The cluster on which the Spark applications were executed. The argument
                can be a cluster name or ID (for databricks platforms) or a valid path to the cluster's
                properties file (json format) generated by the CSP SDK.
        :param platform: defines one of the following "onprem", "emr", "dataproc", "databricks-aws",
                and "databricks-azure".
        :param driverlog: Valid path to the GPU driver log file.
        :param output_folder: path to store the output.
        :param tools_jar: Path to a bundled jar including Rapids tool. The path is a local filesystem,
                or remote cloud storage url. If missing, the wrapper downloads the latest rapids-4-spark-tools_*.jar
                from maven repository.
        :param jvm_heap_size: The maximum heap size of the JVM in gigabytes.
                Default is calculated based on a function of the total memory of the host.
        :param jvm_threads: Number of thread to use for parallel processing on the eventlogs batch.
                Default is calculated as a function of the total number of cores and the heap size on the host.
        :param verbose: True or False to enable verbosity of the script.
        :param tools_config_file: Path to a configuration file that contains the tools' options.
                For sample configuration files, please visit
                https://github.com/NVIDIA/cudf-spark-tools/tree/main/user_tools/tests/spark_rapids_tools_ut/resources/tools_config/valid
        :param target_cluster_info: Path to a YAML file that contains the target cluster information.
                For sample target cluster info files, please visit
                https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/src/main/resources/targetClusterInfo
        :param rapids_options: A list of valid Profiling tool options.
                Note that the wrapper ignores ["output-directory"] flags, and it does not support
                multiple "spark-property" arguments.
                For more details on Profiling tool options, please visit
                https://docs.nvidia.com/spark-rapids/user-guide/latest/profiling/jar-usage.html#prof-tool-title-options
        :param tuning_configs: Path to a YAML file that contains the tuning configurations.
                For sample tuning configs files, please visit
                https://github.com/NVIDIA/cudf-spark-tools/tree/main/core/src/main/resources/bootstrap/tuningConfigs.yaml
        :return: The output folder where the profiling tool output is stored.
        :rtype: Optional[str]
        """
        eventlogs = Utils.get_value_or_pop(eventlogs, rapids_options, 'e')
        cluster = Utils.get_value_or_pop(cluster, rapids_options, 'c')
        platform = Utils.get_value_or_pop(platform, rapids_options, 'p')
        driverlog = Utils.get_value_or_pop(driverlog, rapids_options, 'd')
        output_folder = Utils.get_value_or_pop(output_folder, rapids_options, 'o')
        verbose = Utils.get_value_or_pop(verbose, rapids_options, 'v', False)
        if verbose:
            ToolLogging.enable_debug_mode()
        init_environment('prof')
        prof_args = AbsToolUserArgModel.create_tool_args('profiling',
                                                         eventlogs=eventlogs,
                                                         cluster=cluster,
                                                         platform=platform,
                                                         driverlog=driverlog,
                                                         jvm_heap_size=jvm_heap_size,
                                                         jvm_threads=jvm_threads,
                                                         output_folder=output_folder,
                                                         tools_jar=tools_jar,
                                                         tools_config_path=tools_config_file,
                                                         target_cluster_info=target_cluster_info,
                                                         tuning_configs=tuning_configs)
        if prof_args:
            rapids_options.update(prof_args['rapidOptions'])
            tool_obj = ProfilingAsLocal(platform_type=prof_args['runtimePlatform'],
                                        output_folder=prof_args['outputFolder'],
                                        wrapper_options=prof_args,
                                        rapids_options=rapids_options)
            tool_obj.launch()
            return tool_obj.csp_output_path
        return None

    def prediction(self,
                   qual_output: str = None,
                   output_folder: str = None,
                   custom_model_file: str = None,
                   platform: str = 'onprem',
                   qualx_config: str = None) -> None:
        """The Prediction cmd takes existing qualification tool output and runs the
        estimation model in the qualification tools for GPU speedups.

        :param qual_output: path to the directory, which contains the qualification tool output. E.g. user should
                            specify the parent directory $WORK_DIR where $WORK_DIR/rapids_4_spark_qualification_output
                            exists.
        :param output_folder: path to store the output.
        :param custom_model_file: An optional Path to a custom XGBoost model file. The path is a local filesystem,
                or remote cloud storage url.
        :param platform: defines one of the following "onprem", "dataproc", "databricks-aws",
                         and "databricks-azure", "emr", default to "onprem".
        :param qualx_config: Path to a qualx-conf.yaml file to use for configuration.
               If not provided, the wrapper will use the default:
               https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-conf.yaml.
        """
        # Since prediction is an internal tool with frequent output, we enable debug mode by default
        ToolLogging.enable_debug_mode()

        init_environment('pred')
        estimation_arg_valid = {
            'toolName': 'prediction',
            'validatorName': 'estimation_model_args'
        }
        estimation_model_args = AbsToolUserArgModel.create_tool_args(estimation_arg_valid,
                                                                     estimation_model=QualEstimationModel.XGBOOST,
                                                                     custom_model_file=custom_model_file)
        if estimation_model_args is None:
            return None
        predict_args = AbsToolUserArgModel.create_tool_args('prediction',
                                                            platform=platform,
                                                            qual_output=qual_output,
                                                            output_folder=output_folder,
                                                            estimation_model_args=estimation_model_args,
                                                            qualx_config=qualx_config)

        if predict_args:
            tool_obj = Prediction(platform_type=predict_args['runtimePlatform'],
                                  qual_output=predict_args['qual_output'],
                                  output_folder=predict_args['output_folder'],
                                  wrapper_options=predict_args)
            tool_obj.launch()
        return None

    def train(self,
              dataset: str = None,
              model: str = None,
              output_folder: str = None,
              n_trials: int = None,
              base_model: str = None,
              features_csv_dir: str = None,
              qualx_config: str = None):
        """The Train cmd trains an XGBoost model on the input data to estimate the speedup of a
         Spark CPU application.

        :param dataset: Path to a folder containing one or more dataset JSON files.
        :param model: Path to save the trained XGBoost model.
        :param output_folder: Path to store the output.
        :param n_trials: Number of trials for hyperparameter search.
        :param base_model: Path to pre-trained XGBoost model to continue training from.
        :param features_csv_dir: Path to a folder containing one or more features.csv files.  These files are
                                 produced during prediction, and must be manually edited to provide a label column
                                 (Duration_speedup) and value.
        :param qualx_config: Path to a qualx-conf.yaml file to use for configuration.
               If not provided, the wrapper will use the default:
               https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-conf.yaml.
        """
        # Since train is an internal tool with frequent output, we enable debug mode by default
        ToolLogging.enable_debug_mode()
        init_environment('train')

        train_args = AbsToolUserArgModel.create_tool_args('train',
                                                          platform=CspEnv.get_default(),
                                                          dataset=dataset,
                                                          model=model,
                                                          output_folder=output_folder,
                                                          n_trials=n_trials,
                                                          base_model=base_model,
                                                          features_csv_dir=features_csv_dir,
                                                          qualx_config=qualx_config)

        tool_obj = Train(platform_type=train_args['runtimePlatform'],
                         output_folder=output_folder,
                         wrapper_options=train_args)
        tool_obj.launch()

    def train_and_evaluate(self, qualx_pipeline_config: str = None):
        """The Train and Evaluate cmd trains an XGBoost model and evaluates it using matched CPU and GPU eventlogs.

        This API supports online training by allowing multiple invocations with new data.
        Each invocation will create a new dataset JSON file with an incrementing number.

        :param qualx_pipeline_config: Path to YAML config file containing the required training parameters.
                                      For sample qualx-pipeline-conf-example.yaml files, please visit
                                      https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-pipeline-conf-example.yaml.
        """
        # Since pipeline is an internal tool with frequent output, we enable debug mode by default
        ToolLogging.enable_debug_mode()
        init_environment('train_and_evaluate')

        pipeline_args = AbsToolUserArgModel.create_tool_args('train_and_evaluate',
                                                             platform=CspEnv.get_default(),
                                                             qualx_pipeline_config=qualx_pipeline_config)

        tool_obj = TrainAndEvaluate(platform_type=pipeline_args['runtimePlatform'],
                                    qualx_pipeline_config=qualx_pipeline_config,
                                    wrapper_options=pipeline_args)
        tool_obj.launch()


def main():
    # Make Python Fire not use a pager when it prints a help text
    fire.core.Display = lambda lines, out: out.write('\n'.join(lines) + '\n')
    print(gen_app_banner())
    fire.Fire(ToolsCLI())


if __name__ == '__main__':
    main()
