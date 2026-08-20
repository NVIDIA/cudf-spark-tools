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


"""Wrapper class to run tools associated with the NVIDIA cuDF plugin for Apache Spark on DATABRICKS_AZURE."""

from spark_rapids_pytools.common.utilities import ToolLogging
from spark_rapids_pytools.rapids.diagnostic import Diagnostic
from spark_rapids_tools import CspEnv


class CliDBAzureLocalMode:  # pylint: disable=too-few-public-methods
    """
    A wrapper that runs the cuDF plugin tools locally on the dev machine for DATABRICKS_AZURE.
    """

    @staticmethod
    def diagnostic(cluster: str,
                   profile: str = None,
                   output_folder: str = None,
                   credentials_file: str = None,
                   port: int = 2200,
                   key_file: str = None,
                   thread_num: int = 3,
                   yes: bool = False,
                   verbose: bool = False) -> None:
        """
        Diagnostic tool to collect information from Databricks cluster, such as OS version, # of worker nodes,
        Yarn configuration, Spark version and error logs etc. Please note, some sensitive information might
        be collected by this tool, e.g. access secret configured in configuration files or dumped to log files.
        :param cluster: ID of the Databricks cluster running an accelerated computing instance.
        :param profile: A named Databricks profile to get the settings/credentials of the Databricks CLI.
        :param output_folder: Local path where the archived result will be saved.
               Note that this argument only accepts local filesystem. If the argument is NONE,
               the default value is the env variable "RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY" if any;
               or the current working directory.
        :param credentials_file: The local path of JSON file that contains the application credentials.
               If missing, the wrapper looks for "DATABRICKS_CONFIG_FILE" environment variable
               to provide the location of a credential file. The default credentials file exists as
               "~/.databrickscfg" on Unix, Linux, or macOS.
        :param port: Port number to be used for the ssh connections.
        :param key_file: Path to the private key file to be used for the ssh connections.
        :param thread_num: Number of threads to access remote cluster nodes in parallel. The valid value
               is 1~10. The default value is 3.
        :param yes: auto confirm to interactive question.
        :param verbose: True or False to enable verbosity to the wrapper script.
        """
        if verbose:
            # when debug is set to true set it in the environment.
            ToolLogging.enable_debug_mode()
        wrapper_diag_options = {
            'platformOpts': {
                'profile': profile,
                'credentialFile': credentials_file,
                'sshPort': port,
                'sshKeyFile': key_file,
            },
            'threadNum': thread_num,
            'yes': yes,
        }
        diag_tool = Diagnostic(platform_type=CspEnv.DATABRICKS_AZURE,
                               cluster=cluster,
                               output_folder=output_folder,
                               wrapper_options=wrapper_diag_options)
        diag_tool.launch()


class DBAzureWrapper:  # pylint: disable=too-few-public-methods
    """
    A wrapper script to run cuDF plugin tools Diagnostic on Databricks_Azure.
    """

    def __init__(self):
        self.diagnostic = CliDBAzureLocalMode.diagnostic
