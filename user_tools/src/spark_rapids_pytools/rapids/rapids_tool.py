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

"""Abstract class representing wrapper around the RAPIDS acceleration tools."""

import concurrent
import copy
import logging
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Generic

import spark_rapids_pytools
from spark_rapids_pytools import get_spark_dep_version
from spark_rapids_pytools.cloud_api.sp_types import get_platform, \
    ClusterBase, DeployMode, NodeHWInfo
from spark_rapids_pytools.common.prop_manager import YAMLPropertiesContainer, AbstractPropertiesContainer
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import ToolLogging, Utils, ToolsSpinner
from spark_rapids_pytools.rapids.rapids_job import RapidsJobPropContainer
from spark_rapids_pytools.rapids.tool_ctxt import ToolContext
from spark_rapids_tools import CspEnv
from spark_rapids_tools.api_v1 import ToolResultHandlerT
from spark_rapids_tools.api_v1 import APIResHandler, QualCore, ProfCore
from spark_rapids_tools.configuration.common import RuntimeDependency
from spark_rapids_tools.configuration.submission.distributed_config import DistributedToolsConfig
from spark_rapids_tools.configuration.tools_config import ToolsConfig
from spark_rapids_tools.enums import DependencyType
from spark_rapids_tools.storagelib import LocalPath, CspFs
from spark_rapids_tools.storagelib.tools.fs_utils import untar_file
from spark_rapids_tools.utils import Utilities, AbstractPropContainer
from spark_rapids_tools.utils.util import get_path_as_uri_for_hadoop
from spark_rapids_tools.utils.net_utils import DownloadTask


@dataclass
class RapidsTool(object):
    """
    A generic class that represents a RAPIDS plugin tool.
    :param platform_type: the type of platform associated with the current execution.
    :param cluster: name of the cluster on which the application will be running
    :param output_folder: location to store the output of the execution
    :param config_path: location of the configuration file of the current tool
    :param wrapper_options: dictionary containing options specific to the wrapper tool execution.
    :param rapids_options: dictionary containing the options to be passed as CLI arguments to the RAPIDS Accelerator.
    :param name: the name of the tool
    :param ctxt: context manager for the current tool execution.
    :param logger: the logger instant associated to the current tool.
    """
    platform_type: CspEnv
    cluster: str = None
    output_folder: str = None
    config_path: str = None
    wrapper_options: dict = field(default_factory=dict)
    rapids_options: dict = field(default_factory=dict)
    name: str = field(default=None, init=False)
    ctxt: ToolContext = field(default=None, init=False)
    logger: Logger = field(default=None, init=False)
    spinner: ToolsSpinner = field(default=None, init=False)

    @property
    def csp_output_path(self) -> str:
        """
        Get the output path for the current tool execution. This is different than self.output_folder
        because it includes the folder created by the tool context
        :return: The output path as a string.
        """
        return self.ctxt.get_csp_output_path()

    def get_tools_config_obj(self) -> Optional['ToolsConfig']:
        """
        Get the tools configuration object if provided in the CLI arguments.
        :return: An object containing all the tools configuration or None if not provided.
        """
        return self.wrapper_options.get('toolsConfig')

    def pretty_name(self):
        return self.name.capitalize()

    def get_exec_cluster(self) -> ClusterBase:
        return self.ctxt.get_ctxt('execCluster')

    def is_remote_cluster_execution(self) -> bool:
        """
        used to verify whether a rapids tool runs on a remote cluster submission.
        This does not include the serverlessMode
        :return: True when the tool needs to have a remote cluster established
        """
        return self.ctxt.get_deploy_mode() == DeployMode.REMOTE_CLUSTER

    def requires_remote_folder(self) -> bool:
        """
        used to verify whether a rapids tool running remotely has a defined remote path to generate the
        rapids tool output.
        :return: True when the tool needs to have a remote cluster folder
        """
        deply_mode: DeployMode = self.ctxt.get_deploy_mode()
        return deply_mode.requires_remote_storage()

    def requires_cluster_connection(self) -> bool:
        return self.is_remote_cluster_execution()

    def timeit(timed_item: str):  # pylint: disable=no-self-argument
        def decorator(func_cb: Callable):
            def wrapper(self, *args, **kwargs):
                start_time = time.monotonic()
                func_cb(self, *args, **kwargs)  # pylint: disable=not-callable
                end_time = time.monotonic()
                self.logger.info('Total Execution Time: %s => %s seconds', timed_item,
                                 f'{(end_time-start_time):,.3f}')
            return wrapper
        return decorator

    def phase_banner(phase_name: str,  # pylint: disable=no-self-argument
                     enable_prologue: bool = True,
                     enable_epilogue: bool = True):
        def decorator(func_cb: Callable):
            def wrapper(self, *args, **kwargs):
                try:
                    if enable_prologue:
                        self.logger.info('******* [%s]: Starting *******', phase_name)
                    func_cb(self, *args, **kwargs)  # pylint: disable=not-callable
                    if enable_epilogue:
                        self.logger.info('======= [%s]: Finished =======', phase_name)
                except Exception:    # pylint: disable=broad-except
                    logging.exception('%s. Raised an error in phase [%s]\n',
                                      self.pretty_name(),
                                      phase_name)
                    # clean up the run through the cleanup method
                    self.cleanup_run()
                    sys.exit(1)
            return wrapper
        return decorator

    def __post_init__(self):
        # when debug is set to true set it in the environment.
        self.logger = ToolLogging.get_and_setup_logger(f'rapids.tools.{self.name}')
        self.logger.info('Using Spark RAPIDS user tools version %s', spark_rapids_pytools.__version__)

    def _check_environment(self) -> None:
        self.ctxt.platform.setup_and_validate_env()

    def _process_output_args(self):
        self.logger.debug('Processing Output Arguments')
        # make sure output_folder is absolute
        if self.output_folder is None:
            self.output_folder = Utils.get_or_set_rapids_tools_env('OUTPUT_DIRECTORY', os.getcwd())
        try:
            output_folder_path = LocalPath(self.output_folder)
            self.output_folder = output_folder_path.no_scheme
            self.ctxt.set_local_directories(self.output_folder)
        except Exception as ex:  # pylint: disable=broad-except
            self.logger.error('Failed in processing output arguments. Output_folder must be a local directory')
            raise ex
        self.logger.debug('Root directory of local storage is set as: %s', self.output_folder)
        self.ctxt.load_prepackaged_resources()

    def _process_rapids_args(self):
        pass

    def _process_custom_args(self):
        pass

    def _process_job_submission_args(self):
        pass

    @phase_banner('Process-Arguments')
    def _process_arguments(self):
        try:
            # 0- process the output location
            self._process_output_args()
            # 1- process any arguments to be passed to the RAPIDS tool
            self._process_rapids_args()
            # 2- we need to process the arguments of the CLI
            self._process_custom_args()
            # 3- process submission arguments
            self._process_job_submission_args()
        except Exception as ex:  # pylint: disable=broad-except
            self.logger.error('Failed in processing arguments')
            raise ex

    @phase_banner('Initialization')
    def _init_tool(self):
        self._init_ctxt()
        self._check_environment()

    def _init_ctxt(self):
        if self.config_path is None:
            self.config_path = Utils.resource_path(f'{self.name}-conf.yaml')
        self.ctxt = ToolContext(platform_cls=get_platform(self.platform_type),
                                platform_opts=self.wrapper_options.get('platformOpts'),
                                prop_arg=self.config_path,
                                name=self.name)

    def _run_rapids_tool(self):
        # 1- copy dependencies to remote server
        # 2- prepare the arguments
        # 3- create a submission job
        # 4- execute
        pass

    @phase_banner('Execution')
    def _execute(self):
        """
        Phase representing actual execution of the wrapper command.
        """
        self._run_rapids_tool()

    def _process_output(self):
        pass

    def cleanup_run(self) -> None:
        try:
            if self.ctxt and self.ctxt.do_cleanup_tmp_directory():
                # delete the local tmp directory
                local_tmp_dir = self.ctxt.get_local_tmp_folder()
                FSUtil.remove_path(local_tmp_dir, fail_ok=True)
        except Exception:  # pylint: disable=broad-except
            # Ignore the exception here because this might be called toward the end/failure
            # and we do want to avoid nested exceptions.
            self.logger.debug('Failed to cleanup run')
        finally:
            # Clear RUN_ID to avoid leaking across runs in same process (single-run assumption)
            env_key = Utils.find_full_rapids_tools_env_key('RUN_ID')
            os.environ.pop(env_key, None)

    def _delete_local_dep_folder(self):
        # clean_up the local dependency folder
        local_dep_folder = self.ctxt.get_local_work_dir()
        if self.ctxt.platform.storage.resource_exists(local_dep_folder):
            self.ctxt.platform.storage.remove_resource(local_dep_folder)

    def _delete_remote_dep_folder(self):
        # clean up the remote dep folder first
        remote_dep_folder = self.ctxt.get_remote('depFolder')
        if self.ctxt.platform.storage.resource_exists(remote_dep_folder):
            # delete the folder. Note that for dataproc this will also delete the parent directory when it is empty
            self.ctxt.platform.storage.remove_resource(remote_dep_folder)

    def _download_remote_output_folder(self):
        # download the output folder in to the local one with overriding
        remote_output_folder = self.ctxt.get_remote('workDir')
        # for dataproc it is possible that the entire directory has been deleted when it is empty
        if self.ctxt.platform.storage.resource_exists(remote_output_folder):
            local_folder = self.ctxt.get_csp_output_path()
            self.ctxt.platform.storage.download_resource(remote_output_folder, local_folder)

    def _download_output(self):
        # clean up the remote dep folder first
        self._delete_remote_dep_folder()
        # download the output folder in to the local one with overriding
        self._download_remote_output_folder()

    @phase_banner('Generating Report Summary',
                  enable_epilogue=False)
    def _finalize(self):
        if not ToolLogging.should_suppress_report_summary():
            print(Utils.gen_str_header(f'{self.pretty_name().upper()} Report',
                                       ruler='_',
                                       line_width=100))
        self._write_summary()
        self.cleanup_run()

    def _write_summary(self):
        pass

    @phase_banner('Archiving Tool Output')
    def _archive_phase(self):
        self._archive_results()

    def _archive_results(self):
        pass

    @phase_banner('Collecting-Results')
    def _collect_result(self):
        """
        Following a successful run, collect and process data as needed
        :return:
        """
        self._download_output()
        self._process_output()

    @phase_banner('Connecting to Execution Cluster')
    def _connect_to_execution_cluster(self):
        """
        Connecting to execution cluster
        :return:
        """
        if self.requires_cluster_connection():
            self.logger.info('%s requires the execution cluster %s to be running. '
                             'Establishing connection to cluster',
                             self.pretty_name(),
                             self.cluster)
            exec_cluster = self.ctxt.platform.connect_cluster_by_name(self.cluster)
            self.ctxt.set_ctxt('execCluster', exec_cluster)
            self._verify_exec_cluster()
        else:
            self.logger.info('%s requires no execution cluster. Skipping phase', self.pretty_name())

    def _handle_non_running_exec_cluster(self, err_msg: str) -> None:
        self.logger.warning(err_msg)

    def _verify_exec_cluster(self):
        # For remote job we should fail once we find that the cluster is not actually running
        exec_cluster = self.get_exec_cluster()
        if exec_cluster and exec_cluster.is_cluster_running():
            return
        # For remote cluster mode, the execution cluster must be running
        if not exec_cluster:
            msg = 'An error initializing the execution cluster'
        else:
            msg = f'Remote execution Cluster [{exec_cluster.get_name()}] is not active. ' \
                  f'The execution cluster should be in RUNNING state'
        self._handle_non_running_exec_cluster(msg)

    def launch(self):
        # Spinner should not be enabled in debug mode
        enable_spinner = not ToolLogging.is_debug_mode_enabled()
        with ToolsSpinner(enabled=enable_spinner) as self.spinner:
            self._init_tool()
            self._connect_to_execution_cluster()
            self._process_arguments()
            self._execute()
            self._collect_result()
            self._archive_phase()
            self._finalize()

    def _report_tool_full_location(self) -> str:
        pass

    def _report_results_are_empty(self):
        return [f'The {self.pretty_name()} tool did not generate any output. Nothing to display.']

    def _generate_section_lines(self, sec_conf: dict) -> List[str]:
        all_lines = sec_conf['content'].get('lines')
        if all_lines:
            return all_lines
        return None

    def _generate_section_content(self, sec_conf: dict) -> List[str]:
        sec_title = sec_conf.get('sectionName')
        rep_lines = []
        if sec_title:
            rep_lines.append(Utils.gen_report_sec_header(sec_title, title_width=20))
        if sec_conf.get('content'):
            headers = sec_conf['content'].get('header')
            if headers:
                rep_lines.extend(headers)
            all_lines = self._generate_section_lines(sec_conf)
            if all_lines:
                rep_lines.extend(all_lines)
        return rep_lines

    def _generate_platform_report_sections(self) -> List[str]:
        section_arr = self.ctxt.platform.configs.get_value_silent('wrapperReporting',
                                                                  self.name,
                                                                  'sections')
        if section_arr:
            rep_lines = []
            for curr_sec in section_arr:
                required_flag = curr_sec.get('requiresBoolFlag')
                # if section requires a condition that was not enabled the section is skipped
                if not required_flag or self.ctxt.get_ctxt(required_flag):
                    rep_lines.extend(self._generate_section_content(curr_sec))
            return rep_lines
        return None

    # TODO: This code does not match the AutoTuner.scala output anymore.
    #       We need to consider removing this method or updating it to match the new output.
    def _calculate_spark_settings(self, worker_info: NodeHWInfo) -> dict:
        """
        Calculate the cluster properties that we need to append to the /etc/defaults of the spark
        if necessary.
        :param worker_info: the hardware info as extracted from the worker. Note that we assume
                            that all the workers have the same configurations.
        :return: dictionary containing 7 spark properties to be set by default on the cluster.
        """
        num_gpus = worker_info.gpu_info.num_gpus
        gpu_mem = worker_info.gpu_info.gpu_mem
        num_cpus = worker_info.sys_info.num_cpus
        cpu_mem = worker_info.sys_info.cpu_mem

        config_path = Utils.resource_path('cluster-configs.yaml')
        constants = YAMLPropertiesContainer(prop_arg=config_path).get_value('clusterConfigs', 'constants')
        executors_per_node = num_gpus
        num_executor_cores = max(1, num_cpus // executors_per_node)
        gpu_concurrent_tasks = min(constants.get('maxGpuConcurrent'), gpu_mem // constants.get('gpuMemPerTaskMB'))
        # account for system overhead
        usable_worker_mem = max(0, cpu_mem - constants.get('systemReserveMB'))
        executor_container_mem = usable_worker_mem // executors_per_node
        # reserve 10% of heap as memory overhead
        max_executor_heap = max(0, int(executor_container_mem * (1 - constants.get('heapOverheadFraction'))))
        # give up to 2GB of heap to each executor core
        executor_heap = min(max_executor_heap, constants.get('heapPerCoreMB') * num_executor_cores)
        executor_mem_overhead = int(executor_heap * constants.get('heapOverheadFraction'))
        # use default for pageable_pool to add to memory overhead
        pageable_pool = constants.get('defaultPageablePoolMB')
        # pinned memory uses any unused space up to 4GB
        pinned_mem = min(constants.get('maxPinnedMemoryMB'),
                         executor_container_mem - executor_heap - executor_mem_overhead - pageable_pool)
        executor_mem_overhead += pinned_mem + pageable_pool
        max_sql_files_partitions = constants.get('maxSqlFilesPartitionsMB')
        res = {
            'spark.executor.cores': num_executor_cores,
            'spark.executor.memory': f'{executor_heap}m',
            'spark.executor.memoryOverhead': f'{executor_mem_overhead}m',
            'spark.rapids.sql.concurrentGpuTasks': gpu_concurrent_tasks,
            'spark.rapids.memory.pinnedPool.size': f'{pinned_mem}m',
            'spark.sql.files.maxPartitionBytes': f'{max_sql_files_partitions}m',
            'spark.task.resource.gpu.amount': 1 / num_executor_cores,
            'spark.rapids.shuffle.multiThreaded.reader.threads': num_executor_cores,
            'spark.rapids.shuffle.multiThreaded.writer.threads': num_executor_cores,
            'spark.rapids.sql.multiThreadedRead.numThreads': max(20, num_executor_cores)
        }
        return res

    @classmethod
    def get_rapids_tools_dependencies(cls, deploy_mode: str,
                                      json_props: AbstractPropertiesContainer) -> Optional[list]:
        """
        Get the tools dependencies from the platform configuration.
        """
        # allow defining default buildver per platform
        buildver_from_conf = json_props.get_value_silent('dependencies', 'deployMode', deploy_mode, 'activeBuildVer')
        active_buildver = get_spark_dep_version(buildver_from_conf)
        depend_arr = json_props.get_value_silent('dependencies', 'deployMode', deploy_mode, active_buildver)
        if depend_arr is None:
            raise ValueError(f'Invalid SPARK dependency version [{active_buildver}]')
        # convert the json array to a list of RuntimeDependency objects
        runtime_dep_arr = [RuntimeDependency(**dep) for dep in depend_arr]
        return runtime_dep_arr


@dataclass
class RapidsJarTool(RapidsTool, Generic[ToolResultHandlerT]):
    """
    A wrapper class to represent wrapper commands that require RAPIDS jar file.
    """
    _HADOOP_PATH_ARG_KEYS = frozenset({
        'driverlog',
        'target_cluster_info',
        'tuning_configs',
    })
    # Keep this aligned with EventLogPathProcessor.expandTextFile in Scala. The rewritten
    # local .txt file is passed back into the JAR, so both sides must agree on how entries
    # inside an eventlog list file are tokenized.
    _EVENTLOG_LIST_TOKEN_SPLIT_PATTERN = re.compile(r'[\s,;]+')

    @staticmethod
    def _prepare_hadoop_arg_path(path_value: str) -> str:
        """Normalize a single Hadoop/JVM-bound path string."""
        return get_path_as_uri_for_hadoop(path_value)

    @classmethod
    def _prepare_hadoop_arg_value(cls, option_key: str, option_value: Any) -> Any:
        """
        Normalize supported JAR-bound wrapper options before they are serialized as CLI args.

        `driverlog`, `target_cluster_info`, and `tuning_configs` are the `rapidOptions`
        entries that cross the Hadoop/JVM boundary, and all three are scalar string paths
        produced by the argprocessor.
        """
        if option_key not in cls._HADOOP_PATH_ARG_KEYS:
            return option_value
        return cls._prepare_hadoop_arg_path(option_value)

    @classmethod
    def _rewrite_local_eventlog_list_file(cls, list_file: str, work_dir: str) -> str:
        """
        Rewrite a local eventlog list file so every contained path is explicit for Hadoop.

        Eventlog list files are treated as token lists rather than free-form text. The
        current contract accepts paths separated by newlines, whitespace, commas, or
        semicolons, then rewrites the file as one normalized path per line.
        The rewritten file is created under the wrapper work directory so it remains available
        for the JAR invocation.
        """
        list_path = LocalPath(list_file)
        if not list_path.is_file():
            return list_file

        with open(list_path.no_scheme, mode='r', encoding='utf-8') as src_file:
            list_contents = src_file.read()

        prepared_tokens = [
            cls._prepare_hadoop_arg_path(token)
            for token in cls._EVENTLOG_LIST_TOKEN_SPLIT_PATTERN.split(list_contents)
            if token
        ]
        if not prepared_tokens:
            return list_file
        with tempfile.NamedTemporaryFile(mode='w',
                                         encoding='utf-8',
                                         delete=False,
                                         dir=work_dir,
                                         prefix='eventlogs-',
                                         suffix='.txt') as tmp_file:
            tmp_file.write('\n'.join(prepared_tokens))
            tmp_file.write('\n')
            return f'file://{tmp_file.name}'

    @classmethod
    def _prepare_eventlog_arg_for_hadoop(cls, eventlog: str, work_dir: str) -> str:
        """
        Normalize one eventlog entry before it is handed to the JAR.

        An eventlog entry may itself be a local `.txt` file that expands to multiple eventlog
        paths. Those files require a second pass over their contents before the JAR consumes them.
        """
        prepared_eventlog = cls._prepare_hadoop_arg_path(eventlog)
        if not prepared_eventlog.lower().endswith('.txt'):
            return prepared_eventlog
        if not prepared_eventlog.lower().startswith('file:'):
            return prepared_eventlog
        return cls._rewrite_local_eventlog_list_file(prepared_eventlog, work_dir)

    @classmethod
    def _prepare_eventlog_args_for_hadoop(cls, eventlogs: List[str], work_dir: str) -> List[str]:
        """
        Normalize the final eventlog list stored in wrapper context before JAR handoff.

        `_process_eventlogs_args` resolves the top-level wrapper input into `List[str]`; this
        method applies per-entry normalization and local `.txt` list-file rewriting.
        """
        return [cls._prepare_eventlog_arg_for_hadoop(entry, work_dir) for entry in eventlogs]

    @cached_property
    def core_handler(self) -> APIResHandler[ToolResultHandlerT]:
        """
        Create and return a coreHandler instance for reading core reports.
        This property should always be called after the scala code has executed.
        Otherwise, the property has to be refreshed
        :return: An instance of ToolResultHandlerT which could be QualCoreResultHandler
                 or ProfCoreResultHandler.
        :raises ValueError: If the tool name does not match any known core handler.
        """
        normalized_tool_name = self.name.lower()
        if 'qualification' in normalized_tool_name:
            return QualCore(self.csp_output_path)
        if 'profiling' in normalized_tool_name:
            return ProfCore(self.csp_output_path)
        raise ValueError(f'Tool name [{normalized_tool_name}] has no CoreHandler associated with it.')

    def _process_jar_arg(self):
        # TODO: use the StorageLib to download the jar file
        tools_jar_url = self.wrapper_options.get('toolsJar')
        try:
            if tools_jar_url is None:
                tools_jar_url = self.ctxt.get_rapids_jar_url()

            # Check if this is a local JAR from resources (to avoid race condition)
            if self.ctxt.use_local_tools_jar():
                # Use JAR directly from resources - no need to copy to work directory
                jar_path = tools_jar_url
                self.logger.info('Using tools jar directly from resources %s', jar_path)
            else:
                # download the jar for remote/external jars
                self.logger.info('Downloading the tools jars %s', tools_jar_url)
                jar_path = self.ctxt.platform.storage.download_resource(tools_jar_url,
                                                                        self.ctxt.get_local_work_dir(),
                                                                        fail_ok=False,
                                                                        create_dir=True)
                self.logger.info('RAPIDS accelerator tools jar is downloaded to work_dir %s', jar_path)
        except Exception as e:    # pylint: disable=broad-except
            self.logger.exception('Exception occurred processing jar %s', tools_jar_url)
            raise e

        # get the jar file name
        jar_file_name = FSUtil.get_resource_name(jar_path)
        version_match = re.search(r'\d{2}\.\d{2}\.\d+', jar_file_name)
        jar_version = version_match.group() if version_match else 'Unknown'
        self.logger.info('Using Spark RAPIDS Accelerator Tools jar version %s', jar_version)
        #  add jar file name to the tool args
        self.ctxt.add_rapids_args('jarFileName', jar_file_name)
        self.ctxt.add_rapids_args('jarFilePath', jar_path)

    def __accept_tool_option(self, option_key: str) -> bool:
        defined_tool_options = self.ctxt.get_value_silent('sparkRapids', 'cli', 'toolOptions')
        if defined_tool_options is not None:
            if option_key not in defined_tool_options:
                self.logger.warning('Ignoring tool option [%s]. Invalid option.', option_key)
                return False
        return True

    def _process_tool_args_from_input(self) -> list:
        """
        Process the arguments passed from the CLI if any and return a list of strings representing
        the arguments to be passed to the final command running the job. This needs processing
        because we need to verify the arguments and handle hyphens
        :return: list of the rapids arguments added by the user
        """
        arguments_list = []
        self.logger.debug('Processing Rapids plugin Arguments %s', self.rapids_options)
        raw_tool_opts: Dict[str, Any] = {}
        for key, value in self.rapids_options.items():
            value = self._prepare_hadoop_arg_value(key, value)
            if not isinstance(value, bool):
                # a boolean flag, does not need to have its value added to the list
                if isinstance(value, str):
                    # if the argument is multiple word, then protect it with single quotes.
                    if re.search(r'\s|\(|\)|,', value):
                        value = f"'{value}'"
                raw_tool_opts.setdefault(key, []).append(value)
            else:
                if value:
                    raw_tool_opts.setdefault(key, [])
                else:
                    # argument parser removes the "no-" prefix and set the value to false.
                    # we need to restore the original key
                    raw_tool_opts.setdefault(f'no{key}', [])
        for key, value in raw_tool_opts.items():
            self.logger.debug('Processing tool CLI argument.. %s:%s', key, value)
            if len(key) > 1:
                # python forces "_" to "-". we need to reverse that back.
                fixed_key = key.replace('_', '-')
                prefix = '--'
            else:
                # shortcut argument
                fixed_key = key
                prefix = '-'
            if self.__accept_tool_option(fixed_key):
                k_arg = f'{prefix}{fixed_key}'
                if len(value) >= 1:
                    # handle list options
                    for value_entry in value[0:]:
                        arguments_list.append(f'{k_arg}')
                        arguments_list.append(f'{value_entry}')
                else:
                    # this could be a boolean type flag that has no arguments
                    arguments_list.append(f'{k_arg}')

        # If the target cluster info is provided and contains worker info,
        # set the context to indicate that worker info is provided.
        # This is used later to determine if the Speed up calculation should be skipped if
        # running the Qualification Tool.
        target_cluster_info_file = self.rapids_options.get('target_cluster_info')
        target_cluster_info = (
            AbstractPropContainer.load_from_file(target_cluster_info_file)
            if target_cluster_info_file else None
        )
        # workerInfo may or may not be present in the target cluster info.
        worker_info = target_cluster_info.get_value_silent('workerInfo') if target_cluster_info else None
        if worker_info:
            self.ctxt.set_ctxt('targetWorkerInfoProvided', True)
        return arguments_list

    def _process_tool_args(self):
        """
        Process the arguments passed from the CLI if any and return a string representing the
        arguments to be passed to the final command running the job.
        :return:
        """
        self.ctxt.add_rapids_args('rapidsOpts', self._process_tool_args_from_input())

    def _process_dependencies(self):
        """
        For local deployment mode, we need to process the extra dependencies specific to the platform
        :return:
        """
        if 'deployMode' in self.ctxt.platform_opts:
            # process the deployment
            # we need to download the dependencies locally if necessary
            self._download_dependencies()

    def timeit(timed_item: str):  # pylint: disable=no-self-argument
        def decorator(func_cb: Callable):
            def wrapper(self, *args, **kwargs):
                start_time = time.monotonic()
                func_cb(self, *args, **kwargs)  # pylint: disable=not-callable
                end_time = time.monotonic()
                self.logger.info('Total Execution Time: %s => %s seconds', timed_item,
                                 f'{(end_time-start_time):,.3f}')
            return wrapper
        return decorator

    @timeit('Downloading dependencies for local Mode')  # pylint: disable=too-many-function-args
    def _download_dependencies(self):
        # Default timeout in seconds (120 minutes)
        default_download_timeout = 7200

        def exception_handler(future):
            # Handle any exceptions raised by the task
            exception = future.exception()
            if exception:
                self.logger.error('Error while downloading dependency: %s', exception)

        def cache_single_dependency(dep: RuntimeDependency) -> str:
            """
            Downloads the specified URL and saves it to disk
            """
            self.logger.info('Checking dependency %s', dep.name)
            dest_folder = self.ctxt.get_cache_folder()
            verify_opts = {}
            download_configs = {}
            download_configs['timeOut'] = default_download_timeout
            if dep.verification is not None:
                verify_opts = dict(dep.verification)
            src_url = Utilities.resolve_maven_url(dep.uri)
            download_task = DownloadTask(src_url=src_url,     # pylint: disable=no-value-for-parameter)
                                         dest_folder=dest_folder,
                                         verification=verify_opts,
                                         configs=download_configs)
            download_result = download_task.run_task()
            self.logger.info('Completed downloading of dependency [%s] => %s',
                             dep.name,
                             f'{download_result.pretty_print()}')
            if not download_result.success:
                msg = f'Failed to download dependency {dep.name}, reason: {download_result.download_error}'
                raise RuntimeError(f'Could not download all dependencies. Aborting Executions.\n\t{msg}')
            destination_path = self.ctxt.get_local_work_dir()
            destination_cspath = LocalPath(destination_path)
            # set the default dependency type to jar
            defined_dep_type = DependencyType.get_default()
            if dep.dependency_type:
                defined_dep_type = dep.dependency_type.dep_type
            if defined_dep_type == DependencyType.ARCHIVE:
                uncompressed_cspath = untar_file(download_result.resource, destination_cspath)
                dep_item = uncompressed_cspath.no_scheme
                if dep.dependency_type.relative_path is not None:
                    dep_item = f'{dep_item}/{dep.dependency_type.relative_path}'
            elif defined_dep_type == DependencyType.JAR:
                # copy the jar into dependency folder
                CspFs.copy_resources(download_result.resource, destination_cspath)
                final_dep_csp = destination_cspath.create_sub_path(download_result.resource.base_name())
                dep_item = final_dep_csp.no_scheme
            else:
                raise ValueError(f'Invalid dependency type [{defined_dep_type}]')
            return dep_item

        def cache_all_dependencies(dep_arr: List[RuntimeDependency]) -> List[str]:
            """
            Create a thread pool and download specified urls
            """
            futures_list = []
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for dep in dep_arr:
                    futures = executor.submit(cache_single_dependency, dep)
                    futures.add_done_callback(exception_handler)
                    futures_list.append(futures)

                try:
                    for future in concurrent.futures.as_completed(futures_list, timeout=default_download_timeout):
                        result = future.result()
                        results.append(result)
                except Exception as ex:    # pylint: disable=broad-except
                    raise ex
            return results

        def populate_dependency_list() -> List[RuntimeDependency]:
            # check if the dependencies is defined in a config file
            config_obj = self.get_tools_config_obj()
            if config_obj is not None:
                if config_obj.runtime and config_obj.runtime.dependencies:
                    return config_obj.runtime.dependencies
                self.logger.info('The ToolsConfig did not specify the dependencies. '
                                 'Falling back to the default dependencies.')
            # load dependency list from the platform configuration
            deploy_mode = DeployMode.tostring(self.ctxt.get_deploy_mode())
            return self.get_rapids_tools_dependencies(deploy_mode, self.ctxt.platform.configs)

        depend_arr = populate_dependency_list()
        if depend_arr:
            classpath_deps = [dep for dep in depend_arr if dep.dependency_type and
                              dep.dependency_type.dep_type == DependencyType.CLASSPATH]
            downloadable_deps = [dep for dep in depend_arr if not dep.dependency_type or
                                 dep.dependency_type.dep_type != DependencyType.CLASSPATH]
            dep_list = []
            if downloadable_deps:
                # download the dependencies
                self.logger.info('Downloading dependencies %s', downloadable_deps)
                dep_list = cache_all_dependencies(downloadable_deps)
                if any(dep_item is None for dep_item in dep_list):
                    raise RuntimeError('Could not download all dependencies. Aborting Executions.')
                self.logger.info('Downloadable dependencies are processed as: %s',
                                 Utils.gen_joined_str(join_elem='; ', items=dep_list))
            if classpath_deps:
                for dep_item in classpath_deps:
                    dep_list.append(dep_item.uri)
            self.logger.info('Dependencies are processed as: %s',
                             Utils.gen_joined_str(join_elem='; ',
                                                  items=dep_list))
            self.ctxt.add_rapids_args('javaDependencies', dep_list)

    def _process_rapids_args(self):
        # add a dictionary to hold the rapids arguments
        self._process_jar_arg()
        self._process_dependencies()
        self._process_tool_args()

    def _process_offline_cluster_args(self):
        pass

    def _process_gpu_cluster_args(self, offline_cluster_opts: dict = None):
        pass

    def _copy_dependencies_to_remote(self):
        self.logger.info('Skipping preparing remote dependency folder')

    def _prepare_job_arguments(self):
        self._prepare_local_job_arguments()

    def _run_rapids_tool(self):
        # 1- copy dependencies to remote server
        self._copy_dependencies_to_remote()
        # 2- prepare the arguments
        #  2.a -check if the app_id is not none
        self._prepare_job_arguments()
        # 3- create a submit jobs
        self._submit_jobs()

    def _get_main_cluster_obj(self):
        return self.ctxt.get_ctxt('cpuClusterProxy')

    def _process_eventlogs_args(self):
        def eventlogs_arg_is_requires():
            if self.wrapper_options.get('requiresEventlogs') is not None:
                return self.wrapper_options.get('requiresEventlogs')
            return self.ctxt.requires_eventlogs()

        eventlog_arg = self.wrapper_options.get('eventlogs')
        if eventlog_arg is None:
            # get the eventlogs from spark properties
            cpu_cluster_obj = self._get_main_cluster_obj()
            if cpu_cluster_obj:
                spark_event_logs = cpu_cluster_obj.get_eventlogs_from_config()
            else:
                self.logger.warning('Eventlogs is not set properly. The property cannot be pulled '
                                    'from cluster because it is not defined')
                spark_event_logs = []
        else:
            if isinstance(eventlog_arg, tuple):
                spark_event_logs = List[eventlog_arg]
            elif isinstance(eventlog_arg, str):
                spark_event_logs = eventlog_arg.split(',')
            else:
                spark_event_logs = eventlog_arg
        if eventlogs_arg_is_requires() and len(spark_event_logs) < 1:
            self.logger.error('Eventlogs list is empty. '
                              'The cluster Spark properties may be missing "spark.eventLog.dir". '
                              'Re-run the command passing "--eventlogs" flag to the wrapper.')
            raise RuntimeError('Invalid arguments. The list of Apache Spark event logs is empty.')
        self.ctxt.set_ctxt(
            'eventLogs',
            self._prepare_eventlog_args_for_hadoop(spark_event_logs, self.ctxt.get_local_work_dir())
        )

    def _create_migration_cluster(self, cluster_type: str, cluster_arg: str) -> ClusterBase:
        if cluster_arg is None:
            raise RuntimeError(f'The {cluster_type} cluster argument is not set.')
        arg_is_file = self.ctxt.platform.storage.is_file_path(cluster_arg)
        if not arg_is_file:
            self.logger.info('Loading %s cluster properties by name %s. Note that this will fail '
                             'if the cluster was permanently deleted.',
                             cluster_type,
                             cluster_arg)
            # create a cluster by name
            cluster_obj = self.ctxt.platform.connect_cluster_by_name(cluster_arg)
        else:
            self.logger.info('Loading %s cluster properties from file %s',
                             cluster_type,
                             cluster_arg)
            # create cluster by loading properties files
            # download the file to the working directory
            cluster_conf_path = self.ctxt.platform.storage.download_resource(cluster_arg,
                                                                             self.ctxt.get_local_work_dir())
            cluster_obj = self.ctxt.platform.load_cluster_by_prop_file(cluster_conf_path)
        return cluster_obj

    def _gen_output_tree(self) -> List[str]:
        tree_conf = self.ctxt.get_value('local', 'output', 'treeDirectory')
        if tree_conf and tree_conf.get('enabled'):
            level = tree_conf.get('depthLevel')
            indentation = tree_conf.get('indentation', '\t')
            ex_patterns = tree_conf.get('excludedPatterns', {})
            exc_dirs = ex_patterns.get('directories')
            exc_files = ex_patterns.get('files')
            out_folder_path = self.ctxt.get_csp_output_path()
            out_tree_list = FSUtil.gen_dir_tree(out_folder_path,
                                                depth_limit=level,
                                                indent=indentation,
                                                exec_dirs=exc_dirs,
                                                exec_files=exc_files)
            doc_url = self.ctxt.get_value('sparkRapids', 'outputDocURL')
            out_tree_list.append(f'{indentation}- To learn more about the output details, visit {doc_url}')
            # If running qualification Tool and the target worker info is provided,
            # then add a comment about speed up estimation being inaccurate.
            target_worker_info_provided = self.ctxt.get_ctxt('targetWorkerInfoProvided')
            # Lazy import Qualification to avoid circular import issues
            from spark_rapids_pytools.rapids.qualification import Qualification  # pylint: disable=import-outside-toplevel
            if isinstance(self, Qualification) and target_worker_info_provided:
                inaccurate_speedup_comment =\
                    self.ctxt.get_value('local', 'output', 'stdout', 'inaccurateSpeedupComment')
                out_tree_list.append(f'{indentation}- {inaccurate_speedup_comment}')
            return out_tree_list
        return []

    def _report_tool_full_location(self) -> str:
        if not self._rapids_jar_tool_has_output():
            return None
        out_folder_path = self.ctxt.get_rapids_output_folder()
        res_arr = [Utils.gen_report_sec_header('Output'),
                   f'{self.pretty_name()} tool output: {out_folder_path}']
        out_tree_list = self._gen_output_tree()
        return Utils.gen_multiline_str(res_arr, out_tree_list)

    def _init_core_handler(self) -> None:
        """
        Initializes the core_handler object and store it into the context.
        This method is used to force the refresh of the core_handler property just in case
        the property was called before the tool execution.
        """
        # force the refresh of the core handler property
        if 'core_handler' in self.__dict__:
            del self.__dict__['core_handler']
        self.ctxt.set_ctxt('coreHandler', self.core_handler)

    def _evaluate_rapids_jar_tool_output_exist(self) -> bool:
        """
        Used as a subtask of self._process_output(). this method has the responsibility of
        checking if the tools produced no output and take the necessary action
        :return: True if the tool has generated an output
        """
        self._init_core_handler()
        res = True
        if self.core_handler.is_empty():
            if not self.core_handler.out_path.exists():
                # There is no output_folder at all
                res = False
                self.ctxt.set_ctxt('wrapperOutputContent', self._report_results_are_empty())
                self.logger.info('The Rapids jar tool did not generate an output directory')
        self.ctxt.set_ctxt('rapidsOutputIsGenerated', res)
        return res

    def _rapids_jar_tool_has_output(self) -> bool:
        return self.ctxt.get_ctxt('rapidsOutputIsGenerated')

    @timeit('Processing job submission arguments')  # pylint: disable=too-many-function-args
    def _process_job_submission_args(self):
        self._process_local_job_submission_args()

    def _set_remote_folder_for_submission(self, requires_remote_storage: bool) -> dict:
        res = {}
        submission_args = self.wrapper_options.get('jobSubmissionProps')
        # get the root remote folder and make sure it exists
        remote_folder = submission_args.get('remoteFolder')
        # If remote_folder is not specified, then ignore it
        if requires_remote_storage:
            # if the remote storage required and no remote folder specified. then try to assign the
            # tmp storage of the exec_cluster to be used for storage
            archive_enabled = True
            if not remote_folder:
                # get the execCluster
                exec_cluster = self.get_exec_cluster()
                if exec_cluster:
                    remote_folder = exec_cluster.get_tmp_storage()
                    if remote_folder:
                        archive_enabled = False
            self.ctxt.set_ctxt('archiveToRemote', archive_enabled)
        if remote_folder is None:
            # the output is only for local machine
            self.logger.info('No remote output folder specified.')
            if requires_remote_storage:
                raise RuntimeError(f'Remote folder [{remote_folder}] is invalid.')
        else:
            if not self.ctxt.platform.storage.resource_exists(remote_folder):
                raise RuntimeError(f'Remote folder invalid path. [{remote_folder}] does not exist.')
            # now we should make the subdirectory to indicate the output folder,
            # by appending the name of the execution folder
            exec_full_name = self.ctxt.get_ctxt('execFullName')
            remote_workdir = FSUtil.build_url_from_parts(remote_folder, exec_full_name)
            self.ctxt.set_remote('rootFolder', remote_folder)
            self.ctxt.set_remote('workDir', remote_workdir)
            self.logger.info('Remote workdir is set as %s', remote_workdir)
            remote_dep_folder = FSUtil.build_url_from_parts(remote_workdir,
                                                            self.ctxt.get_ctxt('depFolderName'))
            self.ctxt.set_remote('depFolder', remote_dep_folder)
            self.logger.info('Remote dependency folder is set as %s', remote_dep_folder)
        if requires_remote_storage:
            res.update({'outputDirectory': self.ctxt.get_remote('workDir')})
        else:
            # the output folder has to be set any way
            res.update({'outputDirectory': self.ctxt.get_csp_output_path()})
        return res

    def _process_local_job_submission_args(self):
        job_args = {}
        submission_args = self.wrapper_options.get('jobSubmissionProps')
        job_args.update(self._set_remote_folder_for_submission(self.requires_remote_folder()))
        platform_args = submission_args.get('platformArgs')
        if platform_args is not None:
            processed_platform_args = self.ctxt.platform.cli.build_local_job_arguments(platform_args)
            ctxt_rapids_args = self.ctxt.get_ctxt('rapidsArgs')
            dependencies = ctxt_rapids_args.get('javaDependencies')
            processed_platform_args.update({'dependencies': dependencies})
            job_args['platformArgs'] = processed_platform_args
        self.ctxt.update_job_args(job_args)

    def _init_rapids_arg_list(self) -> List[str]:
        # TODO: Make sure we add this argument only for jar versions 23.02+
        platform_args = ['--platform', self.ctxt.platform.get_platform_name().replace('_', '-')]
        return platform_args

    def _create_autotuner_rapids_args(self) -> list:
        # Add the autotuner argument
        if self.ctxt.get_rapids_auto_tuner_enabled():
            return ['--auto-tuner']
        return []

    def _get_job_submission_resources(self, tool_name: str) -> dict:
        # Used for backward compatibility with legacy CLI that does not calculate job resources
        submission_args = self.wrapper_options.get('jobSubmissionProps')
        job_resources = submission_args.get('jobResources')
        if job_resources is None:
            # default values
            platform_args = submission_args.get('platformArgs')
            job_resources = Utilities.adjust_tools_resources(platform_args.get('jvmMaxHeapSize'), jvm_processes=1)
        return job_resources.get(tool_name)

    def _get_rapids_threads_count(self, tool_name) -> List[str]:
        """
        Get the number of threads to be used by the Rapids tool
        :return: number of threads
        """
        job_resources = self._get_job_submission_resources(tool_name)
        threads_count = job_resources['rapidsThreads']
        return ['--num-threads', f'{threads_count}']

    def _re_evaluate_platform_args(self, tool_name: str) -> dict:
        # re-evaluate the platform arguments
        # get the job arguments
        job_args = self.ctxt.get_ctxt('jobArgs')
        result = copy.deepcopy(job_args)
        job_resources = self._get_job_submission_resources(tool_name)
        jvm_min_heap = job_resources['jvmMinHeapSize']
        jvm_max_heap = job_resources['jvmMaxHeapSize']
        jvm_max_heap_key = f'Xmx{jvm_max_heap}g'
        jvm_min_heap_key = f'Xms{jvm_min_heap}g'
        # At this point, we need to set the heap argument for the JVM. Otherwise, the process uses
        # its default values.
        result['platformArgs']['jvmArgs'].update({
            jvm_min_heap_key: '',
            jvm_max_heap_key: ''
        })
        return result

    @timeit('Building Job Arguments and Executing Job CMD')  # pylint: disable=too-many-function-args
    def _prepare_local_job_arguments(self):
        job_args = self._re_evaluate_platform_args(self.name)
        # now we can create the job object
        # Todo: For dataproc, this can be autogenerated from cluster name
        rapids_arg_list = self._init_rapids_arg_list()
        ctxt_rapids_args = self.ctxt.get_ctxt('rapidsArgs')
        jar_file_path = ctxt_rapids_args.get('jarFilePath')
        rapids_opts = ctxt_rapids_args.get('rapidsOpts')
        if rapids_opts:
            rapids_arg_list.extend(rapids_opts)
        # add the eventlogs at the end of all the tool options
        rapids_arg_list.extend(self.ctxt.get_ctxt('eventLogs'))
        class_name = self.ctxt.get_value('sparkRapids', 'mainClass')
        rapids_arg_obj = {
            'jarFile': jar_file_path,
            'jarArgs': rapids_arg_list,
            'className': class_name
        }
        platform_args = job_args.get('platformArgs')
        spark_conf_args = {}
        job_properties_json = {
            'outputDirectory': job_args.get('outputDirectory'),
            'rapidsArgs': rapids_arg_obj,
            'sparkConfArgs': spark_conf_args,
            'platformArgs': platform_args
        }
        # Set the configuration for the distributed tools
        distributed_tools_configs = self._get_distributed_tools_configs()
        if distributed_tools_configs:
            job_properties_json['distributedToolsConfigs'] = distributed_tools_configs
        rapids_job_container = RapidsJobPropContainer(prop_arg=job_properties_json,
                                                      file_load=False)
        self.ctxt.set_ctxt('rapidsJobContainers', [rapids_job_container])

    def _get_distributed_tools_configs(self) -> Optional[DistributedToolsConfig]:
        """
        Parse the tools configuration and return as distributed tools configuration object
        """
        config_obj = self.get_tools_config_obj()
        if config_obj and config_obj.submission:
            if self.ctxt.is_distributed_mode():
                return config_obj
            self.logger.warning(
                'Distributed tool configurations detected, but distributed mode is not enabled.'
                'Use \'--submission_mode distributed\' flag to enable distributed mode. Switching to local mode.'
            )
        elif self.ctxt.is_distributed_mode():
            self.logger.warning(
                'Distributed mode is enabled, but no distributed tool configurations were provided. '
                'Using default settings.'
            )
        return None

    def _archive_results(self):
        self._archive_local_results()

    def _archive_local_results(self):
        remote_work_dir = self.ctxt.get_remote('workDir')
        if remote_work_dir and self._rapids_jar_tool_has_output():
            local_folder = self.ctxt.get_csp_output_path()
            # TODO make sure it worth issuing the command
            self.ctxt.platform.storage.upload_resource(local_folder, remote_work_dir)

    def _submit_jobs(self):
        # create submission jobs
        rapids_job_containers = self.ctxt.get_ctxt('rapidsJobContainers')
        futures_list = []
        results = []
        executors_cnt = len(rapids_job_containers) if Utilities.conc_mode_enabled else 1
        with ThreadPoolExecutor(max_workers=executors_cnt) as executor:
            for rapids_job in rapids_job_containers:
                if self.ctxt.is_distributed_mode():
                    job_obj = self.ctxt.platform.create_distributed_submission_job(job_prop=rapids_job,
                                                                                   ctxt=self.ctxt)
                else:
                    job_obj = self.ctxt.platform.create_local_submission_job(job_prop=rapids_job,
                                                                             ctxt=self.ctxt)
                futures = executor.submit(job_obj.run_job)
                futures_list.append(futures)
            try:
                for future in concurrent.futures.as_completed(futures_list):
                    result = future.result()
                    results.append(result)
            except Exception as ex:    # pylint: disable=broad-except
                self.logger.error('Failed to submit jobs %s', ex)
                raise ex
