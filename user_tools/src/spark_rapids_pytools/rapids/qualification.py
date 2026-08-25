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

"""Implementation class representing wrapper around the cuDF plugin Qualification tool."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Callable, Optional, Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

from spark_rapids_pytools.cloud_api.sp_types import ClusterBase
from spark_rapids_pytools.common.cluster_inference import ClusterInference, ClusterType
from spark_rapids_pytools.common.prop_manager import JSONPropertiesContainer, convert_dict_to_camel_case
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import Utils, TemplateGenerator, ToolLogging
from spark_rapids_pytools.rapids.qualification_core import QualificationCore
from spark_rapids_tools.enums import QualFilterApp, QualEstimationModel, SubmissionMode
from spark_rapids_tools.tools.additional_heuristics import AdditionalHeuristics
from spark_rapids_tools.tools.cluster_config_recommender import ClusterConfigRecommender
from spark_rapids_tools.tools.qualification_stats_report import SparkQualificationStats
from spark_rapids_tools.tools.qualx.qualx_main import predict
from spark_rapids_tools.tools.speedup_category import SpeedupCategory
from spark_rapids_tools.tools.top_candidates import TopCandidates
from spark_rapids_tools.tools.unsupported_ops_stage_duration import UnsupportedOpsStageDuration
from spark_rapids_tools.utils.util import Utilities


@dataclass
class QualificationSummary:
    """
    Encapsulates the logic to organize Qualification report.
    """
    total_apps: pd.DataFrame = field(init=True)  # Total apps, including failed or skipped
    tools_processed_apps: pd.DataFrame = field(init=True)  # Apps after tools processing and heuristic filtering
    filter_apps_count: int = field(default=0, init=False)  # Count after applying console filters (top candidates)
    comments: Any = None
    sections_generators: List[Callable] = field(default_factory=lambda: [])

    def _has_apps(self) -> bool:
        return self.total_apps is not None and not self.total_apps.empty

    def _has_gpu_recommendation(self) -> bool:
        return self.filter_apps_count > 0

    def generate_report(self,
                        app_name: str,
                        wrapper_output_files_info: dict,
                        csp_report_provider: Callable[[], List[str]] = lambda: [],
                        stdout_table_pprinter: Callable[[pd.DataFrame, pd.DataFrame], TopCandidates] = lambda: None,
                        output_pprinter: Any = None) -> list:
        report_content = []
        if not self._has_apps():
            # Qualification tool has no output
            report_content.append(f'\n{app_name} tool did not generate any valid rows')
            if self.comments:
                report_content.append(Utils.gen_multiline_str(self.comments))
            return report_content

        if output_pprinter is not None:
            report_content.append(output_pprinter())

        # Output files comments should be generated even if there are no apps to show
        self._generate_output_files_comments(wrapper_output_files_info, report_content)
        table_pprinter_obj = stdout_table_pprinter(self.total_apps, self.tools_processed_apps)
        report_content.extend(table_pprinter_obj.generate_summary(app_name))
        self.filter_apps_count = table_pprinter_obj.get_filtered_apps_count()
        if self.comments:
            report_content.append(Utils.gen_report_sec_header('Notes'))
            report_content.extend(f' - {line}' for line in self.comments)
        if self._has_gpu_recommendation():
            csp_report = csp_report_provider()
            if csp_report:
                report_content.extend(csp_report)
        # append an empty line at the end of the report
        report_content.append('')
        return report_content

    @classmethod
    def _generate_output_files_comments(cls, output_files_info: dict, report_content: list) -> None:
        """
        Generate comments for the output files to be displayed in the console report.
        :param output_files_info: Dictionary containing the output files information.
        :param report_content: List to which the output files comments will be appended.
        """
        for entry in output_files_info.values():
            path = entry.get('path', None)
            output_comment = entry.get('outputComment', None)
            if path is not None and output_comment is not None:
                abs_path = FSUtil.get_abs_path(path)
                if FSUtil.resource_exists(abs_path):  # check if the file exists
                    report_content.append(f'    - {output_comment}: {abs_path}')


@dataclass
class Qualification(QualificationCore):
    """
    Wrapper layer around Qualification Tool.
    """
    name = 'qualification'

    def _process_rapids_args(self):
        """
        Qualification tool processes extra arguments:
        1. filter out applications.
        """
        self.logger.info('Qualification tool processing the arguments')
        super()._process_rapids_args()

    def _process_cpu_cluster_args(self, offline_cluster_opts: dict = None):
        # get the name of the cpu_cluster
        cpu_cluster_arg = offline_cluster_opts.get('cpuCluster')
        if cpu_cluster_arg is not None:
            cpu_cluster_obj = self._create_migration_cluster('CPU', cpu_cluster_arg)
            self.ctxt.set_ctxt('cpuClusterProxy', cpu_cluster_obj)

    # process a single cluster specified by the user
    def _process_offline_cluster_args(self) -> None:
        # read the wrapper option defined by the spark_rapids cmd if any.
        offline_cluster_opts = self.wrapper_options.get('migrationClustersProps', {})
        self._process_cpu_cluster_args(offline_cluster_opts)

    def __process_filter_args(self, arg_val: str) -> None:
        selected_filter = QualFilterApp.fromstring(arg_val)
        self.ctxt.set_ctxt('filterApps', selected_filter)

    def _process_estimation_model_args(self) -> None:
        # set the estimation model
        estimation_model_args = self.wrapper_options.get('estimationModelArgs')
        if estimation_model_args is None or not estimation_model_args:
            selected_model = QualEstimationModel.get_default()
            estimation_model_args = QualEstimationModel.create_default_model_args(selected_model)
        self.ctxt.set_ctxt('estimationModelArgs', estimation_model_args)

    def _process_submission_mode_arg(self) -> None:
        """
        Process the value provided by `--submission_mode` argument.
        """
        submission_mode_arg = self.wrapper_options.get('submissionMode')
        if submission_mode_arg is None or not submission_mode_arg:
            submission_mode = SubmissionMode.get_default()
        else:
            submission_mode = SubmissionMode.fromstring(submission_mode_arg)
        self.ctxt.set_ctxt('submissionMode', submission_mode)

    def _process_custom_args(self) -> None:
        """
        Qualification tool processes extra arguments:
        1. filter out applications.
        2. gpu-device type to be used for the cost estimation.
        3. gpu_per_machine: number of gpu installed on a worker node.
        4. cuda version
        """
        gpu_device = self.ctxt.get_value('sparkRapids', 'gpu', 'device')
        gpu_device_arg = self.wrapper_options.get('gpuDevice')
        if gpu_device_arg is not None:
            gpu_device = gpu_device_arg
        gpu_per_machine = int(self.ctxt.get_value('sparkRapids', 'gpu', 'workersPerNode'))
        gpu_per_machine_arg = self.wrapper_options.get('gpuPerMachine')
        if gpu_per_machine_arg is not None:
            gpu_per_machine = gpu_per_machine_arg
        cuda = self.ctxt.get_value('sparkRapids', 'gpu', 'cudaVersion')
        cuda_arg = self.wrapper_options.get('cuda')
        if cuda_arg is not None:
            cuda = cuda_arg
        self.ctxt.set_ctxt('gpuPerMachine', gpu_per_machine)
        self.ctxt.set_ctxt('gpuDevice', gpu_device)
        self.ctxt.set_ctxt('cuda', cuda)
        # we need to process each argument to verify it is valid. otherwise, we may crash late
        self.__process_filter_args(self.wrapper_options.get('filterApps'))
        self._process_estimation_model_args()
        self._process_offline_cluster_args()
        self._process_submission_mode_arg()
        # This is noise to dump everything
        # self.logger.debug('%s custom arguments = %s', self.pretty_name(), self.ctxt.props['wrapperCtx'])
        super()._process_custom_args()

    def __remap_columns_and_prune(self, all_rows) -> pd.DataFrame:
        cols_subset = self.ctxt.get_value('toolOutput', 'csv', 'summaryReport', 'columns')
        # for backward compatibility, filter out non-existing columns
        existing_cols_subset = Utilities.get_valid_df_columns(cols_subset, all_rows)
        cols_map = self.ctxt.get_value('toolOutput', 'csv', 'summaryReport', 'mapColumns')
        subset_data = all_rows.loc[:, existing_cols_subset]
        if cols_map:
            for col_rename in cols_map:
                subset_data.columns = subset_data.columns.str.replace(col_rename,
                                                                      cols_map.get(col_rename),
                                                                      regex=False)
        # Drop columns with only NA values for a cleaner final output.
        return subset_data.dropna(axis=1, how='all')

    def __group_apps_by_name(self, all_apps) -> (pd.DataFrame, str):
        """
        For TCO, group apps by name, cluster id, cluster name and recalculate metrics
        """
        all_apps_count = len(all_apps)
        notes = []
        group_info = self.ctxt.get_value('toolOutput', 'csv', 'summaryReport', 'groupColumns')
        if group_info['enabled'] is False:
            return all_apps, notes
        valid_group_cols = Utilities.get_valid_df_columns(group_info['keys'], all_apps)
        for agg_info in group_info['aggregate']:
            agg_col = agg_info['column']
            if agg_col in all_apps.columns:
                # Group by columns can contain NaN values, so we need to include them in the grouping
                all_apps[agg_col] = all_apps.groupby(valid_group_cols, dropna=False)[agg_col].transform(
                    agg_info['function'])

        drop_arr = self.ctxt.get_value('toolOutput', 'csv', 'summaryReport', 'dropDuplicates')
        valid_drop_cols = Utilities.get_valid_df_columns(drop_arr, all_apps)
        subset_data = all_apps.drop_duplicates(subset=valid_drop_cols)

        if len(subset_data) != all_apps_count:
            notes = 'Apps with the same name are grouped together and their metrics are averaged'

        # recalculate estimated GPU speedup. If no GPU-speedup; then set GPU speedup to 1.0
        result_df = subset_data.copy()
        result_df.loc[:, 'Estimated GPU Speedup'] = np.where(
            result_df['Estimated GPU Duration'] != 0,
            result_df['App Duration'].div(result_df['Estimated GPU Duration'], axis=0),
            1.0)
        # fetch the column names required to recalculate the unsupported operators stage duration percent
        unsupported_ops_col_name = self.ctxt.get_value('local', 'output', 'unsupportedOperators',
                                                       'resultColumnName')
        unsupported_ops_perc_col_name = self.ctxt.get_value('local', 'output', 'unsupportedOperators',
                                                            'percentResultColumnName')
        # recalculate unsupported operators stage duration percent.
        # The equation takes into consideration division by zero.
        result_df[unsupported_ops_perc_col_name] = np.where(
            result_df['SQL Stage Durations Sum'] != 0,
            result_df[unsupported_ops_col_name] * 100.0 / result_df['SQL Stage Durations Sum'],
            100.0
        )
        return result_df, notes

    def __generate_recommended_configs_report(self) -> list:
        # This method will generate the report for the recommended configurations.
        # The configurations disable that section by default.
        report_content = []
        if self.ctxt.get_ctxt('recommendedConfigs'):
            conversion_items = []
            recommended_configs = self.ctxt.get_ctxt('recommendedConfigs')
            for config in recommended_configs:
                conversion_items.append([config, recommended_configs[config]])
            report_content.append(tabulate(conversion_items))
        # the report should be appended to the log_summary file
        rapids_output_dir = self.ctxt.get_rapids_output_folder()
        rapids_log_file = FSUtil.build_path(rapids_output_dir,
                                            self.ctxt.get_value('toolOutput', 'textFormat', 'summaryLog',
                                                                'fileName'))
        with open(rapids_log_file, 'a', encoding='UTF-8') as summary_log_file:
            log_report = [Utils.gen_report_sec_header('Recommended Spark configurations for running on GPUs',
                                                      hrule=False)]
            log_report.extend(report_content)
            summary_log_file.write(Utils.gen_multiline_str(log_report))
        return report_content

    def __generate_cluster_shape_report(self) -> Optional[str]:
        if bool(self.ctxt.platform.ctxt['notes']):
            return Utils.gen_multiline_str(self.ctxt.platform.ctxt['notes'].get('clusterShape'))
        return None

    def __build_global_report_summary(self,
                                      processed_apps: pd.DataFrame,
                                      total_apps: pd.DataFrame,
                                      unsupported_ops_df: pd.DataFrame,
                                      output_files_info: JSONPropertiesContainer) -> QualificationSummary:
        """
        Build global report summary using QualCoreHandler implementation.
        Uses qual_handler to read data instead of direct file access.
        """
        if processed_apps.empty:
            # No need to run saving estimator or process the data frames.
            return QualificationSummary(total_apps=total_apps, tools_processed_apps=processed_apps)

        # Generate the statistics report
        try:
            stats_report = SparkQualificationStats(ctxt=self.ctxt)
            stats_report.report_qualification_stats()
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('Failed to generate the statistics report: %s', e)

        # Calculate unsupported operators stage duration
        # This returns a DF with the following additional columns:
        # 1. Unsupported Operators Stage Duration: Total duration of stages with unsupported operators
        # 2. Unsupported Operators Stage Duration Percent: Percentage of unsupported operators stage duration
        unsupported_ops_obj = UnsupportedOpsStageDuration(
            self.ctxt.get_value('local', 'output', 'unsupportedOperators'))
        all_apps = unsupported_ops_obj.prepare_apps_with_unsupported_stages(processed_apps, unsupported_ops_df)
        apps_pruned_df = self.__remap_columns_and_prune(all_apps)

        # Apply additional heuristics to skip apps not suitable for GPU acceleration
        heuristics_ob = AdditionalHeuristics(
            props=self.ctxt.get_value('local', 'output', 'additionalHeuristics'),
            tools_output_dir=self.ctxt.get_rapids_output_folder(),
            output_file=output_files_info.get_value('intermediateOutput', 'files', 'heuristics', 'path'))
        # The apply heuristics will return DF with two additional columns:
        # 1. Skip by heuristics: True/False
        # 2. Reason: the reason why the app was skipped
        apps_pruned_df = heuristics_ob.apply_heuristics(apps_pruned_df)

        # Group the applications and recalculate metrics
        apps_grouped_df, group_notes = self.__group_apps_by_name(apps_pruned_df)

        apps_with_runtime_df = self.__assign_spark_runtime_to_apps(apps_grouped_df)

        speedup_category_confs = self.ctxt.get_value('local', 'output', 'speedupCategories')
        speedup_category_ob = SpeedupCategory(speedup_category_confs)
        # Adds the Speedup Category Column to the DataFrame.
        # Also processes the SpeedUp assigned using some eligibility conditions present in the qualification-conf.yaml
        # in case that needs to be updated
        df_final_result = speedup_category_ob.build_category_column(apps_with_runtime_df)
        # Fill the missing values in 'Not Recommended Reason' with N/A
        df_final_result['Not Recommended Reason'] = df_final_result['Not Recommended Reason'].fillna('N/A')

        reshaped_notes = self.__generate_cluster_shape_report()
        report_comments = [group_notes] if group_notes else []
        if reshaped_notes:
            report_comments.append(reshaped_notes)

        # Write the final result to the output file
        csv_out = output_files_info.get_value('summary', 'path')
        if not df_final_result.empty:
            self.logger.info('Generating GPU Estimated Speedup: as %s', csv_out)
            df_final_result.to_csv(csv_out, float_format='%.2f', sep=',')

        # Add columns for cluster configuration recommendations and tuning configurations to the processed_apps.
        recommender = ClusterConfigRecommender(self.ctxt)
        df_final_result = recommender.add_cluster_and_tuning_recommendations(df_final_result)
        df_final_result = pd.merge(df_final_result, total_apps[['Event Log', 'App ID']],
                                   left_on='App ID', right_on='App ID')

        # Write the app metadata
        app_metadata_info = output_files_info.get_value('appMetadata')
        config_recommendations_info = output_files_info.get_value('configRecommendations')
        self._write_app_metadata(df_final_result, app_metadata_info, config_recommendations_info)

        # Return the summary
        return QualificationSummary(total_apps=total_apps,
                                    tools_processed_apps=df_final_result,
                                    comments=report_comments)

    @staticmethod
    def __map_cluster_info_table() -> dict:
        """
        Get the mapping from JSON column names (after json_normalize) to CSV column names
        for cluster information files.
        This is a temporary mapping that will be removed once the underlying cluster inference
        code is updated to use the newer JSON column names. All expected column names
        have been added to the mapping.
        """
        return {
            'appId': 'App ID',
            'appName': 'App Name',
            'eventLogPath': 'Event Log',
            'sourceClusterInfo.vendor': 'Vendor',
            'sourceClusterInfo.driverHost': 'Driver Host',
            'sourceClusterInfo.clusterId': 'Cluster Id',
            'sourceClusterInfo.clusterName': 'Cluster Name',
            'sourceClusterInfo.workerNodeType': 'Worker Node Type',
            'sourceClusterInfo.driverNodeType': 'Driver Node Type',
            'sourceClusterInfo.numWorkerNodes': 'Num Worker Nodes',
            'sourceClusterInfo.numExecsPerNode': 'Num Executors Per Node',
            'sourceClusterInfo.numExecutors': 'Num Executors',
            'sourceClusterInfo.executorHeapMemory': 'Executor Heap Memory',
            'sourceClusterInfo.dynamicAllocationEnabled': 'Dynamic Allocation Enabled',
            'sourceClusterInfo.dynamicAllocationMaxExecutors': 'Dynamic Allocation Max Executors',
            'sourceClusterInfo.dynamicAllocationMinExecutors': 'Dynamic Allocation Min Executors',
            'sourceClusterInfo.dynamicAllocationInitialExecutors': 'Dynamic Allocation Initial Executors',
            'sourceClusterInfo.coresPerExecutor': 'Cores Per Executor',
            'recommendedClusterInfo.driverNodeType': 'Recommended Driver Node Type',
            'recommendedClusterInfo.workerNodeType': 'Recommended Worker Node Type',
            'recommendedClusterInfo.numExecutors': 'Recommended Num Executors',
            'recommendedClusterInfo.numWorkerNodes': 'Recommended Num Worker Nodes',
            'recommendedClusterInfo.coresPerExecutor': 'Recommended Cores Per Executor',
            'recommendedClusterInfo.gpuDevice': 'Recommended GPU Device',
            'recommendedClusterInfo.numGpusPerNode': 'Recommended Num GPUs Per Node',
            'recommendedClusterInfo.vendor': 'Recommended Vendor',
            'recommendedClusterInfo.dynamicAllocationEnabled': 'Recommended Dynamic Allocation Enabled',
            'recommendedClusterInfo.dynamicAllocationMaxExecutors': 'Recommended Dynamic Allocation Max Executors',
            'recommendedClusterInfo.dynamicAllocationMinExecutors': 'Recommended Dynamic Allocation Min Executors',
            'recommendedClusterInfo.dynamicAllocationInitialExecutors':
                'Recommended Dynamic Allocation Initial Executors'
        }

    def _process_output(self) -> None:
        if not self._evaluate_rapids_jar_tool_output_exist():
            return

        output_files_info = self.__build_output_files_info()

        def create_stdout_table_pprinter(total_apps: pd.DataFrame,
                                         tools_processed_apps: pd.DataFrame) -> TopCandidates:
            """
            Creates a `TopCandidates` object. This will be used for generating a table report with the given DataFrames.
            This method can be extended to return a different object for different stdout table formats.
            """
            view_dic = self.ctxt.get_value('local', 'output', 'topCandidates')
            view_dic.update({
                'filterEnabled': self.ctxt.get_ctxt('filterApps') == QualFilterApp.TOP_CANDIDATES,
                'configRecommendationsPath': output_files_info.get_value('configRecommendations', 'path')
            })
            return TopCandidates(props=view_dic, total_apps=total_apps, tools_processed_apps=tools_processed_apps)

        # 1. Read summary report using QualCoreHandler
        with self.core_handler.csv('qualCoreCSVSummary') as q_sum_res:
            df = q_sum_res.data
        # 1. Operations related to XGboost modelling
        if not df.empty and self.ctxt.get_ctxt('estimationModelArgs')['xgboostEnabled']:
            try:
                df = self.__update_apps_with_prediction_info(df,
                                                             self.ctxt.get_ctxt('estimationModelArgs'))
            except Exception as e:  # pylint: disable=broad-except
                # If an error occurs while updating the apps with prediction info (speedups and durations),
                # raise an error and stop the execution as the tool cannot continue without this information.
                raise RuntimeError(
                    'Failed to use XGBoost estimation model for speedups. Qualification tool cannot continue. '
                    f'Reason - {type(e).__name__}: {e}'
                ) from e
        # 2. Operations related to cluster information
        try:
            with self.core_handler.csv_combiner(
                    'clusterInfoJSONReport'
            ).suppress_failure() as c_builder:
                # convert the json columns to csv columns
                c_builder.apply_on_report(lambda x: x.map_cols(Qualification.__map_cluster_info_table()))
                # use "App ID" included in the json report
                c_builder.combiner.disable_apps_injection()
                cluster_info_df = c_builder.build()
            # Merge using a left join on 'App Name' and 'App ID'. This ensures `df` includes all cluster
            # info columns, even if `cluster_info_df` is empty.
            df = pd.merge(df, cluster_info_df, on=['App Name', 'App ID'], how='left')
            if len(cluster_info_df) > 0:
                self._infer_clusters_for_apps(cluster_info_df)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('Unable to process cluster information. Cost savings will be disabled. '
                              'Reason - %s:%s', type(e).__name__, e)

        # 3. Operations related to reading qualification output (unsupported operators and apps status)
        with self.core_handler.csv_combiner(
                'unsupportedOpsCSVReport'
        ).suppress_failure() as c_builder:
            # use "App ID" column name on the injected apps
            c_builder.combiner.on_app_fields({'app_id': 'App ID'})
            unsupported_ops_df = c_builder.build()

        with self.core_handler.csv('coreCSVStatus') as status_res:
            apps_status_df = status_res.data

        # 4. Operations related to output
        report_gen = self.__build_global_report_summary(df,
                                                        apps_status_df,
                                                        unsupported_ops_df,
                                                        output_files_info)
        summary_report = report_gen.generate_report(app_name=self.pretty_name(),
                                                    wrapper_output_files_info=output_files_info.props,
                                                    csp_report_provider=self._generate_platform_report_sections,
                                                    stdout_table_pprinter=create_stdout_table_pprinter,
                                                    output_pprinter=self._report_tool_full_location)
        self.ctxt.set_ctxt('wrapperOutputContent', summary_report)

    def _write_summary(self) -> None:
        wrapper_out_content = self.ctxt.get_ctxt('wrapperOutputContent')
        if wrapper_out_content is not None:
            if not ToolLogging.should_suppress_report_summary():
                print(Utils.gen_multiline_str(wrapper_out_content))

    def _generate_section_lines(self, sec_conf: dict) -> List[str]:
        if sec_conf.get('sectionID') == 'gpuClusterCreationScript':
            gpu_cluster = self.ctxt.get_ctxt('gpuClusterProxy')
            script_content = gpu_cluster.generate_create_script()
            highlighted_code = TemplateGenerator.highlight_bash_code(script_content)
            return ['```bash', highlighted_code, '```']
        if sec_conf.get('sectionID') == 'runUserToolsBootstrap':
            gpu_cluster = self.ctxt.get_ctxt('gpuClusterProxy')
            override_args = {'CLUSTER_NAME': '$CLUSTER_NAME'}
            script_content = gpu_cluster.generate_bootstrap_script(overridden_args=override_args)
            highlighted_code = TemplateGenerator.highlight_bash_code(script_content)
            return ['```bash', highlighted_code, '```', '']
        if sec_conf.get('sectionID') == 'gpuBootstrapRecommendedConfigs':
            # This is disabled by default in the config files
            return self.__generate_recommended_configs_report()
        return super()._generate_section_content(sec_conf)

    def _init_rapids_arg_list(self) -> List[str]:
        return super()._init_rapids_arg_list() + self._init_rapids_arg_list_for_qual()

    def _init_rapids_arg_list_for_qual(self) -> List[str]:
        return self._create_autotuner_rapids_args()

    def _infer_cluster_per_app(self, cluster_info_df: pd.DataFrame,
                               cluster_type: ClusterType) -> Dict[str, Optional[ClusterBase]]:
        """
        Infers clusters for each app in the DataFrame and returns a dictionary of Cluster objects.

        :param cluster_info_df: DataFrame containing cluster information for each app.
        :param cluster_type: The type of cluster to infer.
        :return: A dictionary where the key is the app ID and the value is the inferred Cluster object.
        """
        cluster_inference_obj = ClusterInference(platform=self.ctxt.platform, cluster_type=cluster_type)
        return {
            row['App ID']: cluster_inference_obj.infer_cluster(cluster_info_df.iloc[[index]])
            for index, row in cluster_info_df.iterrows()
        }

    def _infer_clusters_for_apps(self, cluster_info_df: pd.DataFrame) -> None:
        """
        Infer CPU and GPU clusters for each app in the DataFrame and set the inferred clusters in the context.
        """
        # if cluster inference is not supported, skip the inference
        if not self.ctxt.platform.cluster_inference_supported:
            self.logger.info('Cluster inference is not supported for platform: %s',
                             self.ctxt.platform.get_platform_name())
            return
        cpu_cluster_cols = self.ctxt.get_value('local', 'output', 'clusterInference', 'cpuClusterColumns')
        gpu_cluster_cols = self.ctxt.get_value('local', 'output', 'clusterInference', 'gpuClusterColumns')
        # ==  Infer CPU clusters per app ==
        # Drop GPU/Recommended columns to infer the CPU cluster information
        cpu_cluster_df = cluster_info_df.drop(columns=gpu_cluster_cols, errors='ignore')
        cpu_clusters_per_app = self._infer_cluster_per_app(cpu_cluster_df, ClusterType.CPU)
        self.ctxt.set_ctxt('cpuClusterInfoPerApp', cpu_clusters_per_app)
        # ==  Infer GPU clusters per app ==
        # Drop CPU columns to infer the GPU cluster information
        gpu_cluster_df = cluster_info_df.drop(columns=cpu_cluster_cols, errors='ignore')
        # Rename GPU columns to drop the 'Recommended' prefix
        gpu_cluster_df.rename(columns=dict(zip(gpu_cluster_cols, cpu_cluster_cols)), inplace=True)
        # Assumption: num executors per node will be same as num gpus per node
        gpu_cluster_df['Num Executors Per Node'] = cluster_info_df['Recommended Num GPUs Per Node']
        gpu_clusters_per_app = self._infer_cluster_per_app(gpu_cluster_df, ClusterType.GPU)
        self.ctxt.set_ctxt('gpuClusterInfoPerApp', gpu_clusters_per_app)

    def __build_output_files_info(self) -> JSONPropertiesContainer:
        """
        Build the full output path for the output files.
        """
        files_info = self.ctxt.get_value('local', 'output', 'files')
        output_folder = self.ctxt.get_csp_output_path()
        output_files_raw = self.__update_files_info_with_paths(files_info, output_folder)
        return JSONPropertiesContainer(output_files_raw, file_load=False)

    def __build_prediction_output_files_info(self) -> dict:
        """
        Build the full output path for the predictions output files
        """
        predictions_info = self.ctxt.get_value('local', 'output', 'predictionModel')
        output_dir = FSUtil.build_path(self.ctxt.get_csp_output_path(), predictions_info['outputDirectory'])
        FSUtil.make_dirs(output_dir)
        return self.__update_files_info_with_paths(predictions_info['files'], output_dir)

    @classmethod
    def __update_files_info_with_paths(cls, files_info: dict, output_dir: str) -> dict:
        """
        Update the given files_info dictionary with full file paths.
        """
        for _, entry in files_info.items():
            file_name = entry['name']
            path = FSUtil.build_path(output_dir, file_name)
            # if entry is a directory, create the directory and update the files info recursively
            if entry.get('isDirectory'):
                FSUtil.make_dirs(path)
                entry['files'] = cls.__update_files_info_with_paths(entry['files'], path)
            entry['path'] = path
        return files_info

    def __update_apps_with_prediction_info(self,
                                           all_apps: pd.DataFrame,
                                           estimation_model_args: dict) -> pd.DataFrame:
        """
        Executes the prediction model, merges prediction data into the apps df, and applies transformations
        based on the prediction model's output and specified mappings.
        """
        # Execute the prediction model
        model_name = self.ctxt.platform.get_prediction_model_name()
        output_info = self.__build_prediction_output_files_info()
        try:
            # use the qual_core_handler to read qual-core reports and raise exception if the result is empty
            if self.core_handler.is_empty():
                raise RuntimeError('QualCoreHandler has no data to process. '
                                   'Please ensure the qualification tool has run successfully.')
            predictions_df = predict(
                platform=model_name,
                qual=self.csp_output_path,
                output_info=output_info,
                model=estimation_model_args['customModelFile'],
                config=estimation_model_args['qualxConfig'],
                qual_handlers=[
                    self.core_handler
                ]
            )
        except Exception as e:  # pylint: disable=broad-except
            predictions_df = pd.DataFrame()
            self.logger.error(
                'Failed to execute the prediction model. Using default speed up of 1.0 for all apps. '
                'Reason - %s:%s', type(e).__name__, e)

        if predictions_df.empty:
            result_df = all_apps.copy()
            # If the prediction model fails, set the estimated GPU speedup to 1.0 and the estimated GPU duration to
            # the app duration.
            result_df['Estimated GPU Speedup'] = 1.0
            result_df['Estimated GPU Duration'] = result_df['App Duration']
        else:
            result_info = self.ctxt.get_value('local', 'output', 'predictionModel', 'updateResult')
            # Merge with a left join to include all rows from all apps and relevant rows from model predictions
            result_df = pd.merge(all_apps, predictions_df[result_info['subsetColumns']],
                                 how='left', left_on='App ID', right_on='appId')
            # Replace columns in all apps with values from corresponding XGBoost columns.
            for remap_column in result_info['remapColumns']:
                src_col, dst_col = remap_column['srcCol'], remap_column['dstCol']
                # Drop the dest column if it exists
                result_df.drop(columns=dst_col, errors='ignore', inplace=True)
                # Rename the source column to the destination column
                result_df.rename(columns={src_col: dst_col}, errors='ignore', inplace=True)
            # if the qualx does not have a speedup value, default to 1.0
            result_df.fillna({'Estimated GPU Speedup': 1.0}, inplace=True)
            # if the qualx does not have a duration value, default to App Duration
            result_df.fillna({'Estimated GPU Duration': result_df['App Duration']}, inplace=True)
        # We need to be careful about other columns that depend on remapped columns
        result_df['Estimated GPU Time Saved'] = result_df['App Duration'] - result_df['Estimated GPU Duration']
        return result_df

    def _write_app_metadata(self, tools_processed_apps: pd.DataFrame,
                            metadata_file_info: dict, config_recommendations_dir_info: dict) -> None:
        """
        Write the metadata for apps to a JSON file.
        :param tools_processed_apps: Processed applications from tools
        :param metadata_file_info: Metadata file information
        :param config_recommendations_dir_info: Configuration recommendations directory information
        """
        if not tools_processed_apps.empty:
            try:
                valid_cols = Utilities.get_valid_df_columns(metadata_file_info.get('columns'), tools_processed_apps)
                app_metadata_df = tools_processed_apps[valid_cols].copy()
                # 1. Prepend parent dir to the config recommendations columns (only for the JSON file, not stdout)
                parent_dir = config_recommendations_dir_info.get('path')

                # Helper function to prepend the parent directory to the config file
                def _prepend_parent_dir(conf_file: str) -> str:
                    conf_file_full = FSUtil.build_path(parent_dir, conf_file)
                    return conf_file_full if FSUtil.resource_exists(conf_file_full) else ''

                for col in config_recommendations_dir_info.get('columns'):
                    if col in app_metadata_df.columns:
                        app_metadata_df[col] = app_metadata_df[col].apply(_prepend_parent_dir)

                # 2. Convert column names to camel case for JSON file writing
                # First, remove any non-alphanumeric characters from column names and convert to lowercase
                app_metadata_df.rename(columns=lambda x: re.sub(r'[^a-z\s]', '', x.lower()), inplace=True)
                # Then, convert df to dict with camel case keys
                app_metadata_dict = convert_dict_to_camel_case(app_metadata_df.to_dict(orient='records'),
                                                               delim=' ')
                with open(metadata_file_info.get('path'), 'w', encoding='UTF-8') as f:
                    json.dump(app_metadata_dict, f, indent=2)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.error('Error writing the app metadata report. Reason - %s:%s',
                                  type(e).__name__, e)
        else:
            self.logger.warning('No applications to write to the metadata report.')

    def __assign_spark_runtime_to_apps(self,
                                       tools_processed_apps: pd.DataFrame) -> pd.DataFrame:
        """
        Uses QualCoreHandler to read application information files.
        Assigns the Spark Runtime (Spark/Photon) to each application. This will be used to categorize
        applications into speedup categories (Small/Medium/Large).
        """
        with self.core_handler.csv_combiner(
                'coreRawApplicationInformationCSV'
        ).suppress_failure() as c_builder:
            # customize the report loading to only select required columns and rename them
            c_builder.apply_on_report(
                lambda r: r.pd_args(
                    {'usecols': ['appId', 'sparkRuntime']}
                ).map_cols({'appId': 'App ID', 'sparkRuntime': 'Spark Runtime'})
            )
            # customize the combiner to not inject app column as we already have it (appId)
            c_builder.combiner.disable_apps_injection()
            # return a single combined dataframe
            spark_runtime_df = c_builder.build()
        return tools_processed_apps.merge(spark_runtime_df, on='App ID', how='left')


@dataclass
class QualificationAsLocal(Qualification):
    """
    Qualification tool running on local development.
    """
    description: str = 'This is the localQualification'

    def _copy_dependencies_to_remote(self):
        self.logger.info('Skipping preparing remote dependency folder')

    def _process_job_submission_args(self):
        self._process_local_job_submission_args()

    def _prepare_job_arguments(self):
        super()._prepare_local_job_arguments()

    def _delete_remote_dep_folder(self):
        self.logger.debug('Local mode skipping deleting the remote workdir')

    def _download_remote_output_folder(self):
        self.logger.debug('Local mode skipping downloading the remote output workdir')

    def _archive_results(self):
        self._archive_local_results()
