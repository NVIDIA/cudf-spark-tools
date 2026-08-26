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


"""Wrapper class to run tools associated with the NVIDIA cuDF plugin for Apache Spark on AWS-EMR."""

from spark_rapids_pytools.common.utilities import ToolLogging
from spark_rapids_pytools.rapids.bootstrap import Bootstrap
from spark_rapids_pytools.rapids.diagnostic import Diagnostic
from spark_rapids_tools import CspEnv


class CliEmrLocalMode:  # pylint: disable=too-few-public-methods
    """
    A wrapper that runs the cuDF plugin tools locally on the dev machine.
    """

    @staticmethod
    def bootstrap(cluster: str,
                  profile: str = None,
                  output_folder: str = None,
                  dry_run: bool = True,
                  key_pair_path: str = None,
                  verbose: bool = False) -> None:
        """
        Bootstrap tool analyzes the CPU and GPU configuration of the EMR cluster
        and updates the Spark default configuration on the cluster's master nodes.

        :param cluster: Name of the EMR cluster running an accelerated computing instance class g4dn.*
        :param profile: A named AWS profile to get the settings/credentials of the AWS account.
        :param output_folder: Local path where the final recommendations will be saved.
               Note that this argument only accepts local filesystem. If the argument is NONE,
               the default value is the env variable "RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY" if any;
               or the current working directory.
        :param dry_run: True or False to update the Spark config settings on EMR master node.
        :param key_pair_path: A '.pem' file path that enables to connect to EC2 instances using SSH.
               If missing, the wrapper reads the env variable 'RAPIDS_USER_TOOLS_KEY_PAIR_PATH' if any.
               For more details on creating key pairs,
               visit https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html.
        :param verbose: True or False to enable verbosity to the wrapper script.
        """
        if verbose:
            # when debug is set to true set it in the environment.
            ToolLogging.enable_debug_mode()
        wrapper_boot_options = {
            'platformOpts': {
                'profile': profile,
                'keyPairPath': key_pair_path
            },
            'dryRun': dry_run
        }
        bootstrap_tool = Bootstrap(platform_type=CspEnv.EMR,
                                   cluster=cluster,
                                   output_folder=output_folder,
                                   wrapper_options=wrapper_boot_options)
        bootstrap_tool.launch()

    @staticmethod
    def diagnostic(cluster: str,
                   profile: str = None,
                   output_folder: str = None,
                   key_pair_path: str = None,
                   thread_num: int = 3,
                   yes: bool = False,
                   verbose: bool = False) -> None:
        """
        Diagnostic tool to collect information from EMR cluster, such as OS version, # of worker nodes,
        Yarn configuration, Spark version and error logs etc. Please note, some sensitive information might
        be collected by this tool, e.g. access secret configured in configuration files or dumped to log files.
        :param cluster: Name of the EMR cluster running an accelerated computing instance class g4dn.*
        :param profile: A named AWS profile to get the settings/credentials of the AWS account.
        :param output_folder: Local path where the archived result will be saved.
               Note that this argument only accepts local filesystem. If the argument is NONE,
               the default value is the env variable "RAPIDS_USER_TOOLS_OUTPUT_DIRECTORY" if any;
               or the current working directory.
        :param key_pair_path: A '.pem' file path that enables to connect to EC2 instances using SSH.
               If missing, the wrapper reads the env variable 'RAPIDS_USER_TOOLS_KEY_PAIR_PATH' if any.
               For more details on creating key pairs,
               visit https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html.
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
                'keyPairPath': key_pair_path,
            },
            'threadNum': thread_num,
            'yes': yes,
        }
        diag_tool = Diagnostic(platform_type=CspEnv.EMR,
                               cluster=cluster,
                               output_folder=output_folder,
                               wrapper_options=wrapper_diag_options)
        diag_tool.launch()


class EMRWrapper:  # pylint: disable=too-few-public-methods
    """
    A wrapper script to run cuDF plugin tools (Bootstrap, and Diagnostic) on Amazon EMR.
    """

    def __init__(self):
        self.bootstrap = CliEmrLocalMode.bootstrap
        self.diagnostic = CliEmrLocalMode.diagnostic
