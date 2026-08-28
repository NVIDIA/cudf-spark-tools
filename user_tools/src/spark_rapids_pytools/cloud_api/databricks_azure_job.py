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

"""Implementation of Job submissions on Databricks Azure"""

from dataclasses import dataclass

from spark_rapids_pytools.common.prop_manager import JSONPropertiesContainer
from spark_rapids_pytools.rapids.rapids_job import RapidsLocalJob
from spark_rapids_tools.storagelib.adls.adlspath import AdlsPath


@dataclass
class DBAzureLocalRapidsJob(RapidsLocalJob):
    """
    Implementation of a cuDF plugin job that runs local on a local machine.
    """
    job_label = 'DBAzureLocal'

    @classmethod
    def get_account_name(cls, eventlogs: list):
        if not eventlogs:
            return ''
        for path in eventlogs:
            if path.startswith('abfss://'):
                # assume all eventlogs are under the same storage account
                return AdlsPath.get_abfs_account_name(path)
        return ''

    def _build_jvm_args(self):
        vm_args = super()._build_jvm_args()

        eventlogs = self.exec_ctxt.get_value('wrapperCtx', 'eventLogs')
        if not eventlogs:
            self.logger.info('The list of Apache Spark event logs is empty.')

        key = ''
        account_name = self.get_account_name(eventlogs)

        if account_name:
            try:
                cmd_args = ['az storage account show-connection-string', '--name', account_name]
                std_out = self.exec_ctxt.platform.cli.run_sys_cmd(cmd_args)
                conn_str = JSONPropertiesContainer(prop_arg=std_out, file_load=False).get_value('connectionString')
                key = conn_str.split('AccountKey=')[1].split(';')[0]
            except Exception as ex:  # pylint: disable=broad-except
                self.logger.info('Error retrieving access key for storage account %s: %s', account_name, ex)
                key = ''

        if key:
            vm_args.append(f'-Drapids.tools.hadoop.fs.azure.account.key.{account_name}.dfs.core.windows.net={key}')

        return vm_args
