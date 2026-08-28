# Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

"""Implementation of the Qualification Stats Report."""


from dataclasses import dataclass, field
from logging import Logger

import pandas as pd

from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import ToolLogging
from spark_rapids_pytools.rapids.tool_ctxt import ToolContext


@dataclass
class SparkQualificationStats:
    """
    Encapsulates the logic to generate the Qualification Stats Report.

    This class processes qualification tool output csv files to produce a report that includes
    critical metrics such as the duration and count of each operator, and whether the operator
    is supported or unsupported on the cuDF plugin. Statistics are aggregated at the
    SQL ID level, and it is provided for the operators which has stage ID mapping.

    Qualification Stats Report include:
    - Operator-Level Statistics: Detailed stats for each operator used in the Spark application,
      the number of occurrences, the stage task duration of which the operator is part of,
      total SQL task duration of which the operator is part of and whether the operator
      is supported to run on the cuDF plugin.
    - Supported vs. Unsupported Operators: The report clearly differentiates between operators
      that can benefit from GPU acceleration and those that cannot, providing insight into
      potential performance bottlenecks.

App ID  SQL ID   Operator  Count StageTaskDuration TotalSQLTaskDuration  % of TotalSQL. Supported
 1       1        Filter      2       70                230                30.43         True
 1       1       Project      3       110               230                47.82         False
 1       1       Project      1       50                230                21.73         True
 1       1         Sort       3       170               230                73.91         True
 1       3    HashAggregate   1       100               100                100.00        False
    """
    logger: Logger = field(default=None, init=False)
    unsupported_operators_df: pd.DataFrame = field(default=None, init=False)
    stages_df: pd.DataFrame = field(default=None, init=False)
    result_df: pd.DataFrame = field(default=None, init=False)
    execs_df: pd.DataFrame = field(default=None, init=False)
    output_columns: dict = field(default=None, init=False)
    ctxt: ToolContext = field(default=None, init=True)

    def __post_init__(self) -> None:
        self.logger = ToolLogging.get_and_setup_logger('rapids.tools.qualification.stats')
        self.output_columns = self.ctxt.get_value('local', 'output', 'files', 'statistics')

    def _read_csv_files(self) -> None:
        self.logger.info('Reading CSV files...')

        core_handler = self.ctxt.get_ctxt('coreHandler')
        if core_handler is None:
            raise ValueError('QualCoreHandler not found in context')

        self.logger.info('Using QualCoreHandler to read data...')
        with core_handler.csv_combiner('unsupportedOpsCSVReport').suppress_failure() as c_builder:
            # 1- use "App ID" column name on the injected apps
            # 2- process successful entries by dropping na rows
            c_builder.combiner.on_app_fields(
                {'app_id': 'App ID'}
            ).entry_success_cb(lambda x, y, z: z.dropna(subset=['Unsupported Operator']))
            self.unsupported_operators_df = c_builder.build()
        with core_handler.csv_combiner('stagesCSVReport').suppress_failure() as c_builder:
            # use "App ID" column name on the injected apps
            c_builder.combiner.on_app_fields({'app_id': 'App ID'})
            self.stages_df = c_builder.build()
        with core_handler.csv_combiner('execCSVReport').suppress_failure() as c_builder:
            # 1- use "App ID" column name on the injected apps
            # 2- process successful entries by dropping na rows
            c_builder.combiner.on_app_fields(
                {'app_id': 'App ID'}
            ).entry_success_cb(lambda x, y, z: z.dropna(subset=['Exec Stages', 'Exec Name']))
            self.execs_df = c_builder.build()

        self.logger.info('Reading data using QualCoreHandler completed.')

    def _convert_durations(self) -> None:
        # Convert durations from milliseconds to seconds
        self.stages_df[['Stage Task Duration', 'Unsupported Task Duration']] /= 1000

    def _preprocess_dataframes(self) -> None:
        self.logger.info('Preprocessing dataframes...')

        # Filter out 'WholeStageCodegen' operators as the child operators are already included
        # in the other rows
        self.execs_df = self.execs_df[
            ~self.execs_df['Exec Name'].str.startswith('WholeStageCodegen')]

        # Split 'Exec Stages' and explode the list into separate rows so that the stageID
        # from this dataframe can be matched with the stageID of stages dataframe
        self.execs_df['Exec Stages'] = self.execs_df['Exec Stages'].str.split(':')
        self.execs_df = (self.execs_df.explode('Exec Stages').
                         rename(columns={'Exec Stages': 'Stage ID'}))
        self.execs_df['Stage ID'] = self.execs_df['Stage ID'].astype(int)

        # Remove duplicate 'Stage ID' rows and rename some columns so that join on dataframes
        # can be done easily
        self.stages_df = self.stages_df.drop_duplicates(subset=['App ID', 'Stage ID'])
        self.stages_df.rename(columns={'Stage Task Duration': 'StageTaskDuration'}, inplace=True)
        self.execs_df.rename(columns={'Exec Name': 'Operator'}, inplace=True)
        self.unsupported_operators_df.rename(columns={'Unsupported Operator': 'Operator'},
                                             inplace=True)
        self.logger.info('Preprocessing dataframes completed.')

    def _merge_dataframes(self) -> None:
        self.logger.info('Merging dataframes to get stats...')
        self._preprocess_dataframes()

        # Merge execs_df with stages_df
        merged_df = self.execs_df.merge(self.stages_df, on=['App ID', 'Stage ID'], how='inner')

        # Count occurrences in unsupported_df and exec_df
        unsupported_count = (self.unsupported_operators_df.
                             groupby(['App ID', 'SQL ID', 'Stage ID', 'Operator']).
                             size().reset_index(name='Unsupported Count'))
        exec_count = (merged_df.
                      groupby(['App ID', 'SQL ID', 'Stage ID', 'Operator']).
                      size().reset_index(name='Exec Count'))

        # Get the number of unsupported count per operator, per stage so that we can mark
        # the operator as supported or unsupported later
        merged_df = merged_df.merge(
            unsupported_count,
            on=['App ID', 'SQL ID', 'Stage ID', 'Operator'], how='left')
        merged_df = merged_df.merge(exec_count,
                                    on=['App ID', 'SQL ID', 'Stage ID', 'Operator'], how='left')
        merged_df['Unsupported Count'] = merged_df['Unsupported Count'].fillna(0).astype(int)
        merged_df['cumulcount'] = (merged_df.
                                   groupby(['App ID', 'SQL ID', 'Stage ID', 'Operator']).cumcount())

        # Set Supported to False where cumulcount(cumulative count) is less than Unsupported Count
        merged_df['Supported'] = merged_df['cumulcount'] >= merged_df['Unsupported Count']
        merged_df.drop(columns=['Unsupported Count', 'Exec Count', 'cumulcount'], inplace=True)

        # Calculate total duration by summing unique stages per SQLID
        total_duration_df = merged_df.drop_duplicates(subset=['App ID', 'SQL ID', 'Stage ID']) \
            .groupby(['App ID', 'SQL ID'])['StageTaskDuration'] \
            .sum().reset_index().rename(columns={'StageTaskDuration': 'TotalSQLTaskDuration'})
        merged_df = merged_df.merge(total_duration_df, on=['App ID', 'SQL ID'], how='left')

        # Mark unique stage task durations
        merged_df['Unique StageTaskDuration'] = ~merged_df.duplicated(
            ['App ID', 'SQL ID', 'Operator', 'Stage ID', 'Supported'])
        merged_df['Adjusted StageTaskDuration'] = (merged_df['StageTaskDuration'] *
                                                   merged_df['Unique StageTaskDuration'])

        final_df = merged_df.groupby(['App ID', 'SQL ID', 'Operator', 'Supported']).agg({
            'Adjusted StageTaskDuration': 'sum',
            'Stage ID': 'count'
        }).reset_index().rename(columns={'Stage ID': 'Count',
                                         'Adjusted StageTaskDuration': 'StageTaskDuration'})

        # Merge total duration and calculate percentage
        final_df = final_df.merge(total_duration_df, on=['App ID', 'SQL ID'], how='left')
        final_df['% of Total SQL Task Duration'] = (
                final_df['StageTaskDuration'] / final_df['TotalSQLTaskDuration'] * 100)

        # Rename columns
        final_df.rename(columns={
            'StageTaskDuration': 'Stage Task Exec Duration(s)',
            'TotalSQLTaskDuration': 'Total SQL Task Duration(s)'
        }, inplace=True)
        self.result_df = final_df[self.output_columns.get('columns')].copy()
        self.logger.info('Merging stats dataframes completed.')

    def _write_results(self) -> None:
        self.logger.info('Writing stats results...')
        result_output_dir = self.ctxt.get_output_folder()
        outputfile_path = self.ctxt.get_value('local', 'output', 'files', 'statistics', 'name')
        output_file = FSUtil.build_path(result_output_dir, outputfile_path)
        self.result_df.to_csv(output_file, float_format='%.2f', index=False)
        self.logger.info('Results have been saved to %s', output_file)

    def report_qualification_stats(self) -> None:
        """
        Reports qualification stats by reading qual tool output CSV files

        If an error occurs, the caller should handle the exception.
        """
        self._read_csv_files()
        self._convert_durations()
        self._merge_dataframes()
        self._write_results()
