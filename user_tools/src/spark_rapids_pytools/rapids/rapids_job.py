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

"""Abstract representation of a wrapper Job"""

import os
from dataclasses import dataclass, field
from logging import Logger
from typing import List, Optional, Union

from spark_rapids_pytools.common.prop_manager import JSONPropertiesContainer
from spark_rapids_pytools.common.utilities import ToolLogging, Utils
from spark_rapids_pytools.rapids.tool_ctxt import ToolContext
from spark_rapids_tools import CspPath
from spark_rapids_tools.storagelib import LocalPath
from spark_rapids_tools_distributed.distributed_main import DistributedToolsExecutor
from spark_rapids_tools_distributed.jar_cmd_args import JarCmdArgs


@dataclass
class RapidsJobPropContainer(JSONPropertiesContainer):
    """
    Manages properties and arguments needed to running cuDF plugin tools.
    """

    def _init_fields(self):
        if self.get_value_silent('rapidsArgs') is None:
            self.props['rapidsArgs'] = {}
        if self.get_value_silent('sparkConfArgs') is None:
            self.props['sparkConfArgs'] = {}
        if self.get_value_silent('platformArgs') is None:
            self.props['platformArgs'] = {}
        if self.get_value_silent('distributedToolsConfigs') is None:
            self.props['distributedToolsConfigs'] = {}

    def get_jar_file(self):
        return self.get_value('rapidsArgs', 'jarFile')

    def get_jar_main_class(self):
        return self.get_value('rapidsArgs', 'className')

    def get_rapids_args(self):
        return self.get_value('rapidsArgs', 'jarArgs')

    def get_distribution_tools_configs(self):
        return self.get_value('distributedToolsConfigs')


@dataclass
class RapidsJob:
    """
    Represents an actual execution of a cuDF plugin tools job on the cloud platform.
    """
    prop_container: RapidsJobPropContainer
    exec_ctxt: ToolContext
    output_path: str = field(default=None, init=False)
    job_label: str = field(default=None, init=False)
    logger: Logger = field(default=None, init=False)

    def get_platform_name(self):
        return self.exec_ctxt.get_platform_name()

    def _init_fields(self):
        self.logger = ToolLogging.get_and_setup_logger(f'rapids.tools.submit.{self.job_label}')
        output_directory = self.prop_container.get_value_silent('outputDirectory')
        if output_directory is not None:
            # use LocalPath to add the 'file://' prefix to the path
            self.output_path = str(LocalPath(output_directory))

    def __post_init__(self):
        self._init_fields()

    def _get_rapids_args_per_platform(self) -> List[str]:
        """Left as placeholder for future use"""
        return []

    def _get_persistent_rapids_args(self):
        rapids_args = self._get_rapids_args_per_platform()[:]
        rapids_args.extend(['--output-directory', self.output_path])
        return rapids_args

    def _build_rapids_args(self):
        rapids_arguments = self._get_persistent_rapids_args()
        extra_rapids_args = self.prop_container.get_rapids_args()
        if extra_rapids_args is None:
            return rapids_arguments
        rapids_arguments.extend(extra_rapids_args)
        return rapids_arguments

    def _build_submission_cmd(self) -> Union[list, JarCmdArgs]:
        raise NotImplementedError

    def _submit_job(self, cmd_args: Union[list, JarCmdArgs]) -> str:
        raise NotImplementedError

    def _print_job_output(self, job_output: str):
        stdout_splits = job_output.splitlines()
        if len(stdout_splits) > 0:
            std_out_lines = Utils.gen_multiline_str([f'\t| {line}' for line in stdout_splits])
            stdout_str = f'\n\t<STDOUT>\n{std_out_lines}'
            self.logger.info('%s job output:%s', self.get_platform_name(), stdout_str)

    def _cleanup_temp_log4j_files(self) -> None:
        """Cleanup temporary log4j file created during the job execution"""
        tmp_file = self.exec_ctxt.get_local('tmp_log4j')
        try:
            os.remove(tmp_file)
            self.logger.info('Temporary log4j properties file removed: %s', tmp_file)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('Error removing temporary log4j properties file: %s', e)

    def run_job(self):
        self.logger.info('Prepare job submission command')
        cmd_args = self._build_submission_cmd()
        self.logger.info('Running the Rapids Job...')
        try:
            job_output = self._submit_job(cmd_args)
            if not ToolLogging.is_debug_mode_enabled():
                # we check the debug level because we do not want the output displayed twice.
                self._print_job_output(job_output)
        finally:
            self._cleanup_temp_log4j_files()
        return job_output

    def _get_hadoop_classpath(self) -> Optional[str]:
        """
        Gets the Hadoop's configuration directory from the environment variables.
        The first valid directory found is returned in the following order:
        1. HADOOP_CONF_DIR
        2. HADOOP_HOME/conf
        3. HADOOP_HOME/etc/hadoop
        Otherwise, returns None.

        """
        hadoop_dir_lookups = {
            'hadoopConfDir': {
                'envVar': 'HADOOP_CONF_DIR',
                'postfix': ''
            },
            'hadoopHomeV1': {
                'envVar': 'HADOOP_HOME',
                'postfix': '/conf'
            },
            'hadoopHomeV2': {
                'envVar': 'HADOOP_HOME',
                'postfix': '/etc/hadoop'
            }
        }
        # Iterate on the hadoop_dir_lookups to return the first valid directory found.
        for dir_key, dir_value in hadoop_dir_lookups.items():
            env_var_value = Utils.get_sys_env_var(dir_value['envVar'])
            if env_var_value is not None:
                postfix = dir_value['postfix']
                conf_dir = f'{env_var_value}{postfix}'
                try:
                    conf_dir_path = LocalPath(conf_dir)
                    if conf_dir_path.is_dir() and conf_dir_path.exists():
                        # return the first valid directory found without the URI prefix
                        return conf_dir_path.no_scheme
                except Exception as e:  # pylint: disable=broad-except
                    self.logger.debug(
                        'Could not build hadoop classpath from %s. Reason: %s', dir_key, e)
        return None

    def _build_classpath(self) -> List[str]:
        deps_arr = [self.prop_container.get_jar_file()]
        hadoop_cp = self._get_hadoop_classpath()
        # append hadoop conf dir if any
        if hadoop_cp is not None:
            deps_arr.append(hadoop_cp)
        dependencies = self.prop_container.get_value_silent('platformArgs', 'dependencies')
        if dependencies is not None:
            deps_arr.extend(dependencies)
        dps_str = Utils.gen_joined_str(':', deps_arr)
        return ['-cp', dps_str]

    def _build_jvm_args(self):
        jvm_args = self.prop_container.get_value_silent('platformArgs', 'jvmArgs')
        vm_args = []
        if jvm_args is not None:
            for jvm_k, jvm_arg in jvm_args.items():
                if jvm_k.startswith('D'):
                    if jvm_k == 'Dlog4j.configuration':
                        rapids_output_folder = self.exec_ctxt.get_rapids_output_folder()
                        log4j_file_name = self.exec_ctxt.get_log4j_properties_file()
                        jvm_arg = ToolLogging.modify_log4j_properties(
                            jvm_arg,
                            f'{rapids_output_folder}/{log4j_file_name}'
                        )
                        self.exec_ctxt.set_local('tmp_log4j', jvm_arg)
                    val = f'-{jvm_k}={jvm_arg}'
                else:
                    val = f'-{jvm_k}'
                vm_args.append(val)
        return vm_args


@dataclass
class RapidsLocalJob(RapidsJob):
    """
    Implementation of a cuDF plugin job that runs local on a machine.
    """

    def _build_submission_cmd(self) -> list:
        # env vars are added later as a separate dictionary
        classpath_arr = self._build_classpath()
        jvm_args_arr = self._build_jvm_args()
        cmd_arg = ['java']
        cmd_arg.extend(jvm_args_arr)
        cmd_arg.extend(classpath_arr)
        cmd_arg.append(self.prop_container.get_jar_main_class())
        cmd_arg.extend(self._build_rapids_args())
        return cmd_arg

    def _submit_job(self, cmd_args: list) -> str:
        env_args = self.prop_container.get_value_silent('platformArgs', 'envArgs')
        out_std = self.exec_ctxt.platform.cli.run_sys_cmd(cmd=cmd_args,
                                                          env_vars=env_args)
        return out_std


@dataclass
class RapidsDistributedJob(RapidsJob):
    """
    Implementation of a cuDF plugin job that runs distributed on a cluster.
    """

    def _build_submission_cmd(self) -> JarCmdArgs:
        classpath_arr = self._build_classpath()
        hadoop_cp = self._get_hadoop_classpath()
        jvm_args_arr = self._build_jvm_args()
        jar_main_class = self.prop_container.get_jar_main_class()
        jar_output_dir_args = self._get_persistent_rapids_args()
        extra_rapids_args = self.prop_container.get_rapids_args()
        return JarCmdArgs(jvm_args_arr, classpath_arr, hadoop_cp, jar_main_class,
                          jar_output_dir_args, extra_rapids_args)

    def _build_classpath(self) -> List[str]:
        """
        Only the cuDF plugin tools JAR file is needed for the classpath.
        Assumption: Each worker node should have the Spark Jars pre-installed.
        TODO: Ship the Spark JARs to the cluster to avoid version mismatch issues.
        """
        return ['-cp', self.prop_container.get_jar_file()]

    def _submit_job(self, cmd_args: JarCmdArgs) -> None:
        """
        Submit the Tools JAR cmd to the Spark cluster.
        """
        user_configs = self.prop_container.get_distribution_tools_configs()
        executor = DistributedToolsExecutor(user_submission_configs=user_configs.submission,
                                            cli_output_path=CspPath(self.exec_ctxt.get_output_folder()),
                                            jar_cmd_args=cmd_args)
        executor.run_as_spark_app()
