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

"""Service providers defined types"""

import configparser
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from logging import Logger
from typing import Type, Any, List, Callable, Union, Optional

from spark_rapids_pytools.common.prop_manager import AbstractPropertiesContainer, JSONPropertiesContainer, \
    get_elem_non_safe
from spark_rapids_pytools.common.sys_storage import StorageDriver, FSUtil
from spark_rapids_pytools.common.utilities import ToolLogging, SysCmd, Utils, TemplateGenerator
from spark_rapids_tools import EnumeratedType, CspEnv
from spark_rapids_tools.storagelib.s3.aws_config import resolve_s3_endpoint_override


class DeployMode(EnumeratedType):
    """List of tools deployment methods"""
    # The rapids job is running on local node
    LOCAL = 'local'
    # The rapids job is submitted on a remote cluster
    REMOTE_CLUSTER = 'remote'

    def requires_remote_storage(self) -> bool:
        return self.value in [self.REMOTE_CLUSTER]


class GpuDevice(EnumeratedType):
    """List of supported GPU devices"""
    T4 = 't4'
    V100 = 'v100'
    K80 = 'k80'
    A100 = 'a100'
    P100 = 'P100'
    P4 = 'P4'
    L4 = 'l4'
    A10 = 'a10'
    A10G = 'a10g'

    @classmethod
    def get_default(cls):
        return cls.T4

    def get_gpu_mem(self) -> list:
        memory_hash = {
            self.T4: [16384],
            self.L4: [24576],
            self.A100: [40960, 81920],
            self.P4: [8192],
            self.K80: [12288],
            self.V100: [16384],
            self.P100: [16384],
            self.A10: [24576],
            self.A10G: [24576]
        }
        return memory_hash.get(self)


class ClusterState(EnumeratedType):
    """
    Standard states for a cluster.
    """
    STARTING = 'starting'
    BOOTSTRAPPING = 'bootstrapping'
    WAITING = 'waiting'
    RUNNING = 'running'
    TERMINATING = 'terminating'
    TERMINATED = 'terminated'
    TERMINATED_WITH_ERRORS = 'terminated_with_errors'
    STOPPED = 'stopped'
    OFFLINE = 'offline'
    UNKNOWN = 'unknown'

    @classmethod
    def get_default(cls):
        return cls.UNKNOWN


class SparkNodeType(EnumeratedType):
    """
    Node type from Spark perspective. We either have a master node or a worker node.
    Note that the provider could have different grouping.
    For example EMR has: master, task, and core.
    Another categorization: onDemand..etc.
    """
    MASTER = 'master'
    WORKER = 'worker'


@dataclass
class SysInfo:
    num_cpus: int = None
    cpu_mem: int = None


@dataclass
class GpuHWInfo:
    num_gpus: int = None
    gpu_mem: int = None
    gpu_device: GpuDevice = GpuDevice.get_default()

    def get_gpu_device_name(self) -> str:
        return GpuDevice.tostring(self.gpu_device)


@dataclass
class NodeHWInfo:
    sys_info: SysInfo = None
    gpu_info: GpuHWInfo = None

    def is_gpu_node(self) -> bool:
        return self.gpu_info is not None


@dataclass
class ClusterNode:
    """
    Represents a single cluster node
    :param node_type: type from Spark perspective (Worker vs Master)
    :param name: name of the node used to remote access in SSH
    :param instance_type: the instance type running on the node
    :param mc_props: holds the properties of the instance type running on the node.
                    This is used for further processing
    :param hw_info: contains hardware settings of the node: System and GPU.
    """
    node_type: SparkNodeType
    name: str = field(default=None, init=False)
    instance_type: str = field(default=None, init=False)
    platform_name: str = field(default=None, init=False)
    props: AbstractPropertiesContainer = field(default=None, init=False)
    mc_props: AbstractPropertiesContainer = field(default=None, init=False)
    hw_info: NodeHWInfo = field(default=None, init=False)

    def set_fields_from_dict(self, field_values: dict = None):
        if field_values is not None:
            for field_name in field_values:
                setattr(self, field_name, field_values.get(field_name))
        self._set_fields_from_props()
        return self

    def _set_fields_from_props(self):
        pass

    def _pull_and_set_mc_props(self, cli=None) -> None:
        instances_description = cli.describe_node_instance(self.instance_type) if cli else None
        self.mc_props = JSONPropertiesContainer(prop_arg=instances_description, file_load=False)

    def _pull_gpu_hw_info(self, cli=None) -> Optional[GpuHWInfo]:
        raise NotImplementedError

    def _pull_sys_info(self) -> SysInfo:
        cpu_mem = self.mc_props.get_value('MemoryInMB')
        num_cpus = self.mc_props.get_value('VCpuCount')
        return SysInfo(num_cpus=num_cpus, cpu_mem=cpu_mem)

    def get_name(self) -> str:
        return self.name

    def construct_hw_info(self,
                          cli=None,
                          gpu_info: GpuHWInfo = None,
                          sys_info: SysInfo = None):
        del cli  # Unused cli, defined for future use.
        self.hw_info = NodeHWInfo(sys_info=sys_info,
                                  gpu_info=gpu_info)

    def fetch_and_set_hw_info(self, cli=None):
        self._pull_and_set_mc_props(cli)
        sys_info = self._pull_sys_info()
        try:
            # if a node has no gpu, then it is expected that setting the gpu info fails
            gpu_info = self._pull_gpu_hw_info(cli)
        except Exception as ex:  # pylint: disable=broad-except
            cli.logger.info(f'Could not pull GPU info for '
                            f'{SparkNodeType.tostring(self.node_type)} node {self.name}: {ex}')
            gpu_info = None
        self.construct_hw_info(cli=cli, gpu_info=gpu_info, sys_info=sys_info)

    def find_best_cpu_conversion(self, target_list: dict):
        target_cpus = self.hw_info.sys_info.num_cpus
        best_match = None
        last_record = None
        for prof_id, hw_info in target_list.items():
            # assume that the list is sorted by number of CPUs
            last_record = prof_id
            if hw_info.sys_info.num_cpus <= target_cpus:
                best_match = prof_id
        if best_match is None:
            best_match = last_record
        return best_match

    @classmethod
    def create_worker_node(cls) -> Any:
        return cls(SparkNodeType.WORKER)

    @classmethod
    def create_master_node(cls) -> Any:
        return cls(SparkNodeType.MASTER)

    @classmethod
    def create_node(cls, value):
        if isinstance(value, SparkNodeType):
            if value == SparkNodeType.MASTER:
                return cls.create_master_node()
            if value == SparkNodeType.WORKER:
                return cls.create_worker_node()
        raise RuntimeError(f'Invalid node type while creating cluster node {value}')


@dataclass
class ClusterGetAccessor:
    """
    Represents the interface used to access the cluster information that is
    used by other entities such as the SavingEstimator
    """
    def get_node(self, node_type: SparkNodeType) -> ClusterNode:
        raise NotImplementedError

    def get_all_nodes(self) -> list:
        raise NotImplementedError

    def get_nodes_cnt(self, node_type: SparkNodeType) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def get_node_core_count(self, node_type: SparkNodeType) -> int:
        node = self.get_node(node_type)
        return node.hw_info.sys_info.num_cpus

    def get_node_mem_mb(self, node_type: SparkNodeType) -> int:
        node = self.get_node(node_type)
        return node.hw_info.sys_info.cpu_mem

    def get_gpu_per_node(self, node_type: SparkNodeType) -> (int, str):
        node = self.get_node(node_type)
        gpu_info = node.hw_info.gpu_info
        if gpu_info:
            num_gpus, gpu_device = gpu_info.num_gpus, gpu_info.gpu_device
        else:
            num_gpus, gpu_device = 0, GpuDevice.get_default()
        return num_gpus, GpuDevice.tostring(gpu_device)

    def get_node_instance_type(self, node_type: SparkNodeType) -> str:
        node = self.get_node(node_type)
        return node.instance_type

    def get_workers_instant_types(self) -> str:
        return self.get_node_instance_type(SparkNodeType.WORKER)

    def get_workers_count(self) -> int:
        return self.get_nodes_cnt(SparkNodeType.WORKER)

    def get_workers_cores_count(self) -> int:
        return self.get_node_core_count(SparkNodeType.WORKER)

    def get_workers_mem_mb(self) -> int:
        return self.get_node_mem_mb(SparkNodeType.WORKER)

    def get_gpu_per_worker(self) -> (int, str):
        return self.get_gpu_per_node(SparkNodeType.WORKER)


@dataclass
class CMDDriverBase:
    """
    Represents the command interface that will be used by the platform
    It has:
    1- the command used by the platform. for example gcloud, gsutil, or AWS
    2- the ssh command to nodes (it could include authentications)
    3- normal commands
    :param cloud_ctxt: dictionary containing all the necessary configurations related to the CSP
    :param timeout: How long to wait (in seconds) for the command to finish (optional).
    :param: env_vars: dictionary containing all the variables required by the driver.
    """
    cloud_ctxt: dict
    timeout: int = 0
    env_vars: dict = field(default_factory=dict, init=False)
    logger: Logger = None
    instance_descriptions: JSONPropertiesContainer = field(default=None, init=False)

    def get_env_var(self, key: str):
        return self.env_vars.get(key)

    def get_region(self) -> str:
        return self.env_vars.get('region')

    def get_zone(self) -> str:
        return self.env_vars.get('zone')

    def get_cmd_run_configs(self) -> dict:
        return self.env_vars.get('cmdRunnerProperties')

    def get_required_props(self) -> list:
        cmd_runner_props = self.get_cmd_run_configs()
        if cmd_runner_props:
            return cmd_runner_props.get('inheritedProps')
        return None

    def get_system_prerequisites(self) -> list:
        res = []
        cmd_runner_props = self.get_cmd_run_configs()
        if cmd_runner_props:
            res.extend(cmd_runner_props.get('systemPrerequisites'))
        return res

    def get_piggyback_props(self) -> list:
        res = []
        cmd_runner_props = self.get_cmd_run_configs()
        if cmd_runner_props:
            res = cmd_runner_props.get('cliPiggyBackEnvVars')['definedVars']
        return res

    def get_piggyback_arguments(self) -> list:
        res = []
        cmd_runner_props = self.get_cmd_run_configs()
        if cmd_runner_props:
            res = cmd_runner_props.get('cliPiggyBackArgs')['definedArgs']
        return res

    def get_rapids_job_configs(self, deploy_mode: DeployMode) -> dict:
        cmd_runner_props = self.get_cmd_run_configs()
        if cmd_runner_props and deploy_mode is not None:
            deploy_mode_configs = get_elem_non_safe(cmd_runner_props,
                                                    ['rapidsJobs', DeployMode.tostring(deploy_mode)])
            return deploy_mode_configs
        return None

    def get_and_set_env_vars(self):
        """For that driver, try to get all the available system environment for the system."""
        for item_key in self.cloud_ctxt:
            # save all not-None entries to the env_vars
            item_value = self.cloud_ctxt.get(item_key)
            if item_value is not None:
                self.env_vars[item_key] = item_value

    def _list_inconsistent_configurations(self) -> list:
        """
        List all the inconsistent configuration in the platform
        :return: a list of inconsistencies
        """
        incorrect_envs = []
        for sys_tool in self.get_system_prerequisites():
            if not Utils.is_system_tool(sys_tool):
                incorrect_envs.append(f'Tool {sys_tool} is not in installed or not in the PATH environment')
        if self.get_region() is None:
            incorrect_envs.append('Platform region is not set.')
        return incorrect_envs

    def _handle_inconsistent_configurations(self, incorrect_envs: list) -> None:
        if len(incorrect_envs) > 0:
            # we do not want to raise a runtime error because some of the flags are not required by
            # all the tools.
            # TODO: improve this by checking the requirements for each tool.
            exc_msg = Utils.gen_joined_str('; ', incorrect_envs)
            self.logger.warning('Environment report: %s', exc_msg)

    def validate_env(self):
        incorrect_envs = self._list_inconsistent_configurations()
        self._handle_inconsistent_configurations(incorrect_envs)

    def run_sys_cmd(self,
                    cmd: Union[str, list],
                    cmd_input: str = None,
                    fail_ok: bool = False,
                    env_vars: dict = None) -> str:

        def process_credentials_option(cmd: list):
            res = []
            for i, arg in enumerate(cmd):
                if 'account-key' in cmd[i - 1]:
                    arg = 'MY_ACCESS_KEY'
                elif 'fs.azure.account.key' in arg:
                    arg = arg.split('=')[0] + '=MY_ACCESS_KEY'
                res.append(arg)
            return res

        def process_streams(std_out, std_err):
            if ToolLogging.is_debug_mode_enabled():
                # reformat lines to make the log more readable
                stdout_splits = std_out.splitlines()
                stderr_splits = std_err.splitlines()
                stdout_str = ''
                if len(stdout_splits) > 0:
                    std_out_lines = Utils.gen_multiline_str([f'\t| {line}' for line in stdout_splits])
                    stdout_str = f'\n\t<STDOUT>\n{std_out_lines}'
                # if the command is already a list, use it as-is. Otherwise, split the string into a list.
                cmd_list = cmd if isinstance(cmd, list) else cmd.split(' ')
                cmd_log_str = Utils.gen_joined_str(' ', process_credentials_option(cmd_list))
                if len(stderr_splits) > 0:
                    std_err_lines = Utils.gen_multiline_str([f'\t| {line}' for line in stderr_splits])
                    stderr_str = f'\n\t<STDERR>\n{std_err_lines}'
                    self.logger.debug('executing CMD:\n\t<CMD: %s>[%s]; [%s]',
                                      cmd_log_str,
                                      stdout_str,
                                      stderr_str)

        def is_sdk_cmd(original_cmd, cmd_prefix: str) -> bool:
            if isinstance(original_cmd, list):
                # the command is an array, then we should only pick the first index
                cmd_token = original_cmd[0]
            else:
                cmd_token = original_cmd
            return cmd_token.startswith(cmd_prefix)

        def append_to_cmd(original_cmd, extra_args: list) -> Any:
            if isinstance(original_cmd, list):
                # We do not append at the end of the cmd because this can break some commands like
                # spark-submit
                res = []
                ind = 0
                # loop until we find the first argument (starts with --))
                while ind < len(original_cmd) and not original_cmd[ind].startswith('--'):
                    res.append(original_cmd[ind])
                    ind += 1
                res.extend(extra_args)
                if ind < len(original_cmd):
                    res.extend(original_cmd[ind:])
                return res
            extra_args_flatten = Utils.gen_joined_str(' ', extra_args)
            return f'{original_cmd} {extra_args_flatten}'

        # process the env_variables of the command
        piggyback_vars = self.get_piggyback_props()
        for var_entry in piggyback_vars:
            prop_label = var_entry['confProperty']
            var_value = self.get_env_var(prop_label)
            if var_value:
                if not env_vars:
                    env_vars = {var_entry['varKey']: var_value}
                else:
                    # use setdefault in case the env_car was already defined
                    env_vars.setdefault(var_entry['varKey'], var_value)
        # process the pigyybacked sdk arguments
        piggyback_args = []
        piggyback_args_raw = self.get_piggyback_arguments()
        for arg_entry in piggyback_args_raw:
            if is_sdk_cmd(cmd, arg_entry['sdkCommand']):
                # we should apply the
                arg_k = arg_entry['argKey']
                piggyback_args.append(f'--{arg_k}')
                if 'argValue' in arg_entry:
                    arg_v = arg_entry['argValue']
                    piggyback_args.append(f'{arg_v}')
                else:
                    arg_value = self.get_env_var(arg_entry['confProperty'])
                    piggyback_args.append(arg_value)
        if piggyback_args:
            cmd = append_to_cmd(cmd, piggyback_args)

        cmd_args = {
            'cmd': cmd,
            'fail_ok': fail_ok,
            'cmd_input': cmd_input,
            'env_vars': env_vars,
            'process_streams_cb': process_streams
        }
        sys_cmd = SysCmd().build(cmd_args)
        return sys_cmd.exec()

    def _build_cmd_ssh_prefix_for_node(self, node: ClusterNode) -> str:
        del node  # Unused by super method.
        return ''

    def _build_cmd_scp_to_node(self, node: ClusterNode, src: str, dest: str) -> str:  # pylint: disable=unused-argument
        del node  # Unused by super method.
        return ''

    def _build_cmd_scp_from_node(self, node: ClusterNode, src: str, dest: str) -> str:  # pylint: disable=unused-argument
        del node  # Unused by super method.
        return ''

    def _construct_ssh_cmd_with_prefix(self, prefix: str, remote_cmd: str) -> str:
        return f'{prefix} {remote_cmd}'

    def ssh_cmd_node(self, node: ClusterNode, ssh_cmd: str, cmd_input: str = None) -> str:
        prefix_cmd = self._build_cmd_ssh_prefix_for_node(node=node)
        full_ssh_cmd = self._construct_ssh_cmd_with_prefix(prefix=prefix_cmd, remote_cmd=ssh_cmd)
        return self.run_sys_cmd(full_ssh_cmd, cmd_input=cmd_input)

    def scp_to_node(self, node: ClusterNode, src: str, dest: str) -> str:
        cmd = self._build_cmd_scp_to_node(node=node, src=src, dest=dest)
        return self.run_sys_cmd(cmd)

    def scp_from_node(self, node: ClusterNode, src: str, dest: str) -> str:
        cmd = self._build_cmd_scp_from_node(node=node, src=src, dest=dest)
        return self.run_sys_cmd(cmd)

    def pull_cluster_props_by_args(self, args: dict) -> str or None:
        del args  # Unused by super method.
        return ''

    def init_instance_descriptions(self) -> None:
        """
        Load instance description file from resources based on platform type.
        """
        platform = CspEnv.pretty_print(self.cloud_ctxt['platformType'])
        if platform != CspEnv.ONPREM:
            # we do not have instance descriptions for on-prem
            instance_description_file_path = Utils.resource_path(f'{platform}-instance-catalog.json')
            self.logger.info('Loading instance descriptions from file: %s', instance_description_file_path)
            self.instance_descriptions = JSONPropertiesContainer(instance_description_file_path)

    def describe_node_instance(self, instance_type: str) -> str:
        instance_info = self.instance_descriptions.get_value(instance_type)
        if instance_info is None:
            raise RuntimeError(f'Instance type {instance_type} is not found in catalog.')
        return instance_info

    def _build_platform_list_cluster(self,
                                     cluster,
                                     query_args: dict = None) -> list:
        raise NotImplementedError

    def pull_node_pool_props_by_args(self, args: dict) -> str:
        raise NotImplementedError

    def exec_platform_list_cluster_instances(self,
                                             cluster,
                                             query_args: dict = None) -> str:
        cmd_args = self._build_platform_list_cluster(cluster=cluster, query_args=query_args)
        return self.run_sys_cmd(cmd_args)

    def exec_platform_describe_accelerator(self,
                                           accelerator_type: str,
                                           **cmd_args) -> str:
        """
        Some platforms like Dataproc represent GPUs as accelerators.
        To get the information of each accelerator, we need to run describe cmd
        :param accelerator_type: the name of the GPU accelerator which can be platform specific
        :param cmd_args: the arguments to be sent to the sdk
        :return: a string in json format representing the information about the accelerator
        """
        del accelerator_type  # Unused accelerator_type
        del cmd_args  # Unused cmd_args
        return ''

    def build_local_job_arguments(self, submit_args: dict) -> dict:
        """
        an implementation specific to the platform that build a dictionary to store argument and
        sys env-vars needed for the submission of a local mode on that platform
        :param submit_args: the arguments specified by the user that reflects on the platform.
        :return: a dictionary in the format of {"jvmArgs": {}, "envArgs": {}}
        """
        res = {
            'jvmArgs': {
                # Use the full AWS SDK default credential chain for S3A, which covers:
                # env vars, ~/.aws/credentials (with AWS_PROFILE), EC2 instance profiles,
                # and ECS container credentials — matching AWS CLI / boto3 behavior.
                'Drapids.tools.hadoop.fs.s3a.aws.credentials.provider':
                    'com.amazonaws.auth.DefaultAWSCredentialsProviderChain'
            },
            'envArgs': {}
        }
        # Pass custom S3 endpoint to hadoop-aws if set, matching the Python-side
        # S3Fs handler logic.
        s3_endpoint = resolve_s3_endpoint_override()
        if s3_endpoint:
            res['jvmArgs']['Drapids.tools.hadoop.fs.s3a.endpoint'] = s3_endpoint
            res['jvmArgs']['Drapids.tools.hadoop.fs.s3a.path.style.access'] = 'true'
        jvm_gc_type = submit_args.get('jvmGC')
        if jvm_gc_type is not None:
            xgc_key = f'XX:{jvm_gc_type}'
            res['jvmArgs'].update({xgc_key: ''})
        log4j_config_path = submit_args.get('Dlog4j.configuration')
        if log4j_config_path is not None:
            res['jvmArgs'].update({'Dlog4j.configuration': log4j_config_path})

        rapids_configs = self.get_rapids_job_configs(self.cloud_ctxt.get('deployMode'))
        if not rapids_configs:
            return res
        global_sys_vars = rapids_configs.get('definedVars')
        if not global_sys_vars:
            return res
        env_args_table = {}
        for sys_var in global_sys_vars:
            var_key = sys_var['varKey']
            # OS env vars take precedence over credentials file (matches AWS SDK behavior)
            os_value = os.environ.get(var_key)
            if os_value:
                env_args_table.setdefault(var_key, os_value)
            else:
                prop_value = self.get_env_var(sys_var['confProperty'])
                if prop_value:
                    env_args_table.setdefault(var_key, prop_value)
        res.update({'envArgs': env_args_table})
        return res

    def get_submit_spark_job_cmd_for_cluster(self,
                                             cluster_name: str,
                                             submit_args: dict) -> List[str]:
        raise NotImplementedError

    def _process_instance_description(self, instance_descriptions: str) -> dict:
        raise NotImplementedError

    def get_instance_description_cli_params(self) -> List[str]:
        raise NotImplementedError

    def generate_instance_description(self, fpath: str) -> None:
        """
        Generates CSP instance type descriptions and store them in a json file.
        Json file entry example ('GpuInfo' is optional):
          {
            "instance_name": {
              "VCpuCount": 000,
              "MemoryInMB": 000,
              "GpuInfo": [
                {
                  "Name": gpu_name,
                  "Count": [
                    000
                  ]
                }
              ]
            }
          }
        :param fpath: the output json file path.
        :return:
        """
        cmd_params = self.get_instance_description_cli_params()
        raw_instance_descriptions = self.run_sys_cmd(cmd_params)
        json_instance_descriptions = self._process_instance_description(raw_instance_descriptions)
        with open(fpath, 'w', encoding='UTF-8') as output_file:
            json.dump(json_instance_descriptions, output_file, indent=2)

    def __post_init__(self):
        self.logger = ToolLogging.get_and_setup_logger('rapids.tools.cmd_driver')
        self.init_instance_descriptions()


@dataclass
class PlatformBase:
    """
    Represents the common methods used by all other platforms.
    We need to load constants about platform:
    1- supported machine types
    2- supported Images or non-supported images
    3- GPU specs
    4- pricing catalog
    """
    ctxt_args: dict
    type_id: CspEnv = field(default_factory=lambda: CspEnv.NONE, init=False)
    platform: str = field(default=None, init=False)
    cli: CMDDriverBase = field(default=None, init=False)
    storage: StorageDriver = field(default=None, init=False)
    ctxt: dict = field(default_factory=dict, init=False)
    configs: JSONPropertiesContainer = field(default=None, init=False)
    logger: Logger = field(default=ToolLogging.get_and_setup_logger('rapids.tools.csp'), init=False)
    cluster_inference_supported: bool = field(default=False, init=False)

    @classmethod
    def list_supported_gpus(cls):
        return [GpuDevice.T4, GpuDevice.A100, GpuDevice.L4, GpuDevice.A10]

    def load_from_config_parser(self, conf_file, **prop_args) -> dict:
        res = None
        try:
            parser_obj = configparser.ConfigParser()
            parser_obj.read(conf_file)
            res = {}
            if prop_args.get('sectionKey'):
                section_name = prop_args.get('sectionKey')
                if not parser_obj.has_section(section_name):
                    # try to use "profile XYZ" format
                    if parser_obj.has_section(f'profile {section_name}'):
                        section_name = f'profile {section_name}'
                key_list = prop_args.get('keyList')
                for k in key_list:
                    if parser_obj.has_option(section_name, k):
                        res.update({k: parser_obj.get(section_name, k)})
        except (configparser.NoSectionError, configparser.NoOptionError, configparser.ParsingError) as conf_ex:
            self.logger.debug('Could not load properties from configuration file %s. Exception: %s',
                              conf_file, conf_ex)
        return res

    def _construct_cli_object(self) -> CMDDriverBase:
        raise NotImplementedError

    def _create_cli_instance(self) -> CMDDriverBase:
        cmd_driver_props = self._get_config_environment('cmdRunnerProperties')
        self.ctxt['cmdRunnerProperties'] = cmd_driver_props
        return self._construct_cli_object()

    def _install_storage_driver(self):
        raise NotImplementedError

    def _get_config_environment(self, *key_strs) -> Any:
        return self.configs.get_value('environment', *key_strs)

    def _load_config_environment_var_prop(self, prop_key: str):
        env_variables = self._get_config_environment('cliConfig', 'envVariables')
        # find the env_variable that maps to the property
        res = []
        for env_var in env_variables:
            if env_var['confProperty'] == prop_key:
                res.append(env_var)
        return res

    def _set_env_prop_from_env_var(self, prop_key: str) -> None:
        if self.ctxt.get(prop_key):
            # it is already set. do nothing
            return
        # find the env_variable that maps to the property
        for env_var in self._load_config_environment_var_prop(prop_key):
            env_var_key = env_var['envVariableKey']
            env_var_def_val = env_var.get('defaultValue')
            env_var_val = Utils.get_sys_env_var(env_var_key, env_var_def_val)
            if env_var_val is not None:
                if '/' in env_var_val:
                    # this is a file
                    env_var_val = FSUtil.expand_path(env_var_val)
                self.ctxt.update({prop_key: env_var_val})
                self.logger.warning('Property %s is not set. Setting default value %s'
                                    ' from environment variable', prop_key, env_var_val)
                break

    def _set_initial_configuration_list(self) -> None:
        # load the initial configurations list
        initial_conf_list = self._get_config_environment('initialConfigList')
        if initial_conf_list:
            for conf_prop_k in initial_conf_list:
                self._set_env_prop_from_env_var(conf_prop_k)

    def _set_remaining_configuration_list(self) -> None:
        remaining_props = self._get_config_environment('loadedConfigProps')
        if not remaining_props:
            return
        properties_map_arr = self._get_config_environment('cliConfig',
                                                          'confProperties',
                                                          'propertiesMap')
        if properties_map_arr:
            # We support multiple CLI configurations, the following two dictionaries
            # map config files to the corresponding property keys to be set, and section names respectively
            config_file_keys = defaultdict(list)
            config_file_section = {}
            for prop_elem in properties_map_arr:
                if prop_elem.get('confProperty') in remaining_props:
                    # The property uses the default value which is '_cliConfigFile_'
                    config_file = prop_elem.get('configFileProp', '_cliConfigFile_')
                    config_file_keys[config_file].append(prop_elem.get('propKey'))
                    if config_file not in config_file_section:
                        config_file_section[config_file] = prop_elem.get('section').strip('_')
            # The section names are loaded from dictionary 'config_file_section'
            # Example section names are awsProfile/profile
            loaded_conf_dict = {}
            for config_file in config_file_keys:
                loaded_conf_dict = \
                    self._load_props_from_sdk_conf_file(keyList=config_file_keys[config_file],
                                                        configFile=config_file.strip('_'),
                                                        sectionKey=self.ctxt.get(config_file_section[config_file]))
                if loaded_conf_dict:
                    self.ctxt.update(loaded_conf_dict)
            # If the property key is not already set, below code attempts to set the property
            # using an environment variable. This is a fallback mechanism to populate configuration
            # properties from the environment if they are not already set in the
            # loaded configuration.
            for prop_entry in properties_map_arr:
                prop_entry_key = prop_entry.get('propKey')
                # set it using environment variable if possible
                self._set_env_prop_from_env_var(prop_entry_key)

    def _set_credential_properties(self) -> None:
        properties_map_arr = self._get_config_environment('cliConfig',
                                                          'confProperties',
                                                          'credentialsMap')
        if not properties_map_arr:
            return
        # We support multiple CLI configurations, the following two dictionaries
        # map config files to the corresponding property keys to be set, and section names respectively
        credential_file_keys = defaultdict(list)
        credential_file_section = {}
        for prop_elem in properties_map_arr:
            credential_file = prop_elem.get('configFileProp', '_credentialFile_')
            credential_file_keys[credential_file].append(prop_elem.get('propKey'))
            if credential_file not in credential_file_section:
                credential_file_section[credential_file] = prop_elem.get('section').strip('_')
        # The section names are loaded from dictionary 'config_file_section'
        # Example section names are awsProfile/profile
        for credential_file in credential_file_keys:
            credential_file_value = self.ctxt.get(credential_file.strip('_'))
            if not credential_file_value:
                continue
            loaded_conf_dict = \
                self.load_from_config_parser(credential_file_value,
                                             keyList=credential_file_keys[credential_file],
                                             sectionKey=self.ctxt.get(credential_file_section[credential_file]))
            if loaded_conf_dict:
                self.ctxt.update(loaded_conf_dict)

    def _parse_arguments(self, ctxt_args: dict):
        # Get the possible parameters for that platform.
        # Arguments passed to the tool have more precedence than global env variables
        list_of_params = self._get_config_environment('envParams')
        if list_of_params:
            for param_elem in list_of_params:
                param_val = ctxt_args.get(param_elem)
                if param_val:
                    # add the argument to the context
                    self.ctxt.update({param_elem: param_val})
        self._set_initial_configuration_list()
        # load the remaining properties
        self._set_remaining_configuration_list()
        # load the credential properties
        self._set_credential_properties()

    def _load_props_from_sdk_conf_file(self, **prop_args) -> dict:
        if prop_args.get('configFile'):
            cli_conf_file = self.ctxt.get(prop_args.get('configFile'))
        else:
            cli_conf_file = self.ctxt.get('cliConfigFile')
        if cli_conf_file is None:
            return None
        return self.load_from_config_parser(cli_conf_file, **prop_args)

    def __post_init__(self):
        self.load_platform_configs()
        self.ctxt = {
            'platformType': self.type_id,
            'notes': {}
        }
        self._parse_arguments(self.ctxt_args)
        self.cli = self._create_cli_instance()
        self._install_storage_driver()

    def update_ctxt_notes(self, note_key, note_value):
        self.ctxt['notes'].update({note_key: note_value})

    def setup_and_validate_env(self):
        if self.cli is not None:
            self.cli.get_and_set_env_vars()
            self.cli.validate_env()
        # TODO we should fail if the CLI is None

    def _construct_cluster_from_props(self,
                                      cluster: str,
                                      props: str = None,
                                      is_inferred: bool = False,
                                      is_props_file: bool = False):
        raise NotImplementedError

    def set_offline_cluster(self, cluster_args: dict = None):
        raise NotImplementedError

    def load_cluster_by_prop(self, cluster_prop: JSONPropertiesContainer, is_inferred=False):
        cluster = cluster_prop.get_value_silent('cluster_id')
        return self._construct_cluster_from_props(cluster=cluster,
                                                  props=json.dumps(cluster_prop.props),
                                                  is_inferred=is_inferred,
                                                  is_props_file=True)

    def load_cluster_by_prop_file(self, cluster_prop_path: str):
        prop_container = JSONPropertiesContainer(prop_arg=cluster_prop_path)
        return self.load_cluster_by_prop(prop_container)

    def connect_cluster_by_name(self, cluster: str):
        """
        To be used to if the cluster can be found in the platform.
        :param cluster:
        :return:
        """
        cluster_props = self.cli.pull_cluster_props_by_args(args={'cluster': cluster})
        return self._construct_cluster_from_props(cluster=cluster,
                                                  props=cluster_props)

    def migrate_cluster_to_gpu(self, orig_cluster):
        """
        given a cluster, convert it to run NVIDIA Gpu based on mapping instance types
        :param orig_cluster: the cluster on which the application was executed and is running
                without acceleration.
        :return: a new object cluster that supports GPU
        """
        raise NotImplementedError

    def create_saving_estimator(self,
                                source_cluster: ClusterGetAccessor,
                                reshaped_cluster: ClusterGetAccessor,
                                target_cost: float = None,
                                source_cost: float = None):
        raise NotImplementedError

    def create_local_submission_job(self, job_prop, ctxt) -> Any:
        raise NotImplementedError

    def create_distributed_submission_job(self, job_prop, ctxt) -> Any:
        raise NotImplementedError

    def load_platform_configs(self):
        config_file_name = f'{CspEnv.tostring(self.type_id).lower()}-configs.json'
        config_path = Utils.resource_path(config_file_name)
        self.configs = JSONPropertiesContainer(prop_arg=config_path)

    def get_supported_gpus(self) -> dict:
        gpus_from_configs = self.configs.get_value('gpuConfigs', 'user-tools', 'supportedGpuInstances')
        gpu_scopes = {}
        for mc_prof, mc_info in gpus_from_configs.items():
            hw_info_json = mc_info['SysInfo']
            hw_info_ob = SysInfo(num_cpus=hw_info_json['num_cpus'], cpu_mem=hw_info_json['cpu_mem'])
            gpu_info_json = mc_info['GpuHWInfo']
            gpu_info_obj = GpuHWInfo(num_gpus=gpu_info_json['num_gpus'], gpu_mem=gpu_info_json['gpu_mem'])
            gpu_scopes[mc_prof] = NodeHWInfo(sys_info=hw_info_ob, gpu_info=gpu_info_obj)
        return gpu_scopes

    def validate_job_submission_args(self, submission_args: dict) -> dict:
        raise NotImplementedError

    def get_platform_name(self) -> str:
        """
        This used to get the lower case of the platform of the runtime.
        :return: the name of the platform of the runtime in lower_case.
        """
        return CspEnv.pretty_print(self.type_id)

    def _get_prediction_model_name(self) -> str:
        """
        Internal function to get the prediction model name.
        This should be overridden by subclasses
        """
        return CspEnv.pretty_print(self.type_id)

    def get_prediction_model_name(self) -> str:
        return self._get_prediction_model_name().replace('_', '-')

    def get_footer_message(self) -> str:
        return 'To support acceleration with T4 GPUs, switch the worker node instance types'

    def get_matching_worker_node_type(self, total_cores: int) -> Optional[str]:
        node_types_from_config = self.configs.get_value('clusterInference', 'defaultCpuInstances', 'executor')
        return next((node_type['name'] for node_type in node_types_from_config if node_type['vCPUs'] == total_cores),
                    None)

    def generate_cluster_configuration(self, render_args: dict):
        if not self.cluster_inference_supported:
            return None
        template_path = Utils.resource_path(f'templates/cluster_template/{CspEnv.pretty_print(self.type_id)}.ms')
        return TemplateGenerator.render_template_file(template_path, render_args)

    def get_platform_gpu_device_name(self, gpu_device_str: str) -> str:
        """
        Convert generic GPU device name to platform-specific GPU device name.
        By default, returns the GPU device string as-is.
        Override this method in platform-specific classes to provide
        platform-specific GPU device naming (e.g., "nvidia-l4" for Dataproc).

        :param gpu_device_str: Generic GPU device name (e.g., "T4", "L4")
        :return: Platform-specific GPU device name
        """
        return gpu_device_str


@dataclass
class ClusterBase(ClusterGetAccessor):
    """
    Represents an instance of a cluster on the platform.
    Cluster can be running/offline
    """
    platform: PlatformBase
    cli: CMDDriverBase = field(default=None, init=False)
    name: str = field(default=None, init=False)
    uuid: str = field(default=None, init=False)
    region: str = field(default=None, init=False)
    zone: str = field(default=None, init=False)
    state: ClusterState = field(default=ClusterState.RUNNING, init=False)
    nodes: dict = field(default_factory=dict, init=False)
    props: AbstractPropertiesContainer = field(default=None, init=False)
    logger: Logger = field(default=ToolLogging.get_and_setup_logger('rapids.tools.cluster'), init=False)
    is_inferred: bool = field(default=False, init=True)
    is_props_file: bool = field(default=False, init=True)  # indicates if the cluster is loaded from properties file

    @staticmethod
    def _verify_workers_exist(has_no_workers_cb: Callable[[], bool]):
        """
        Specifies how to handle cluster definitions that have no workers
        :param has_no_workers_cb: A callback that returns True if the cluster does not have any
               workers
        """
        if has_no_workers_cb():
            raise RuntimeError('Invalid cluster: The cluster has no worker nodes.\n\t'
                               'It is recommended to define a with (1 master, N workers).')

    def __post_init__(self):
        self.cli = self.platform.cli
        self.region = self.cli.get_region()

    def _init_connection(self, cluster_id: str = None,
                         props: str = None) -> dict:
        name = cluster_id
        if props is None:
            # we need to pull the properties from the platform
            props = self.cli.pull_cluster_props_by_args(args={'cluster': name, 'region': self.region})
        cluster_props = JSONPropertiesContainer(props, file_load=False)
        cluster_args = {
            'name': name,
            'props': cluster_props
        }
        return cluster_args

    def set_fields_from_dict(self, field_values: dict = None):
        """
        Given a dictionary, this function is to set the fields of the cluster
        :param field_values: the dictionary containing the key/value pair to initialize the cluster.
        :return:
        """
        if field_values is not None:
            for field_name in field_values:
                setattr(self, field_name, field_values.get(field_name))
        self._set_fields_from_props()

    def _process_loaded_props(self) -> None:
        """
        After loading the raw properties, perform any necessary processing to clean up the
        properties.
        """
        return None

    def _set_name_from_props(self) -> None:
        pass

    def _set_fields_from_props(self):
        self._process_loaded_props()
        if not self.name:
            self._set_name_from_props()

    def _init_nodes(self):
        pass

    def set_connection(self,
                       cluster_id: str = None,
                       props: str = None):
        """
        Setting a connection to an existing Connection to a cluster, then we need to pull properties
        :param cluster_id: the argument to be used to fetch the cluster
        :param props: optional argument that includes dictionary of the platform cluster's description.
        :return: a cluster
        """
        pre_init_args = self._init_connection(cluster_id, props)
        self.set_fields_from_dict(pre_init_args)
        self._init_nodes()
        # Verify that the cluster has defined workers
        self._verify_workers_exist(lambda: not self.nodes.get(SparkNodeType.WORKER))
        return self

    def is_cluster_running(self) -> bool:
        return self.state == ClusterState.RUNNING

    def is_gpu_cluster(self) -> bool:
        return self.get_gpu_per_worker()[0] > 0

    def get_eventlogs_from_config(self) -> List[str]:
        res_arr = []
        spark_props = self.get_all_spark_properties()
        if spark_props and 'spark.eventLog.dir' in spark_props:
            res_arr.append(spark_props.get('spark.eventLog.dir'))
        return res_arr

    def run_cmd_driver(self, ssh_cmd: str, cmd_input: str = None) -> str or None:
        """
        Execute command on the driver node
        :param ssh_cmd: the command to be executed on the remote node. Note that the quotes
                        surrounding the shell command should be included
        :param cmd_input: optional argument string used as an input to the command line.
                        i.e., writing to a file.
        :return:
        """
        # get the master node
        master_node: ClusterNode = self.get_master_node()
        return self.cli.ssh_cmd_node(master_node, ssh_cmd, cmd_input=cmd_input)

    def run_cmd_worker(self, ssh_cmd: str, cmd_input: str = None, ind: int = 0) -> str or None:
        """
        Execute command on the worker node
        :param ssh_cmd: the command to be executed on the remote node. Note that the quotes
                        surrounding the shell command should be included
        :param cmd_input: optional argument string used as an input to the command line.
                          i.e., writing to a file
        :param ind: the node index. By default, the command is executed on first worker node.
        """
        # get the worker node
        worker_node: ClusterNode = self.get_worker_node(ind)
        return self.cli.ssh_cmd_node(worker_node, ssh_cmd, cmd_input=cmd_input)

    def run_cmd_node(self, node: ClusterNode, ssh_cmd: str, cmd_input: str = None) -> str or None:
        """
        Execute command on the node
        :param node: the cluster node where the command to be executed on
        :param ssh_cmd: the command to be executed on the remote node. Note that the quotes
                        surrounding the shell command should be included
        :param cmd_input: optional argument string used as an input to the command line.
                          i.e., writing to a file
        """
        return self.cli.ssh_cmd_node(node, ssh_cmd, cmd_input=cmd_input)

    def scp_to_node(self, node: ClusterNode, src: str, dest: str) -> str or None:
        """
        Scp file to the node
        :param node: the cluster node to upload file to.
        :param src: the file path to be uploaded to the cluster node.
        :param dest: the file path where to store uploaded file on the cluster node.
        """
        return self.cli.scp_to_node(node, src, dest)

    def scp_from_node(self, node: ClusterNode, src: str, dest: str) -> str or None:
        """
        Scp file from the node
        :param node: the cluster node to download file from.
        :param src: the file path on the cluster node to be downloaded.
        :param dest: the file path where to store downloaded file.
        """
        return self.cli.scp_from_node(node, src, dest)

    def get_region(self) -> str:
        return self.cli.get_region()

    def get_worker_hw_info(self) -> NodeHWInfo:
        worker_node = self.get_worker_node()
        return worker_node.hw_info

    def _build_migrated_cluster(self, orig_cluster):
        """
        specific to the platform on how to build a cluster based on migration
        :param orig_cluster:
        """
        raise NotImplementedError

    def migrate_from_cluster(self, orig_cluster):
        self.name = orig_cluster.name
        self.uuid = orig_cluster.uuid
        self.zone = orig_cluster.zone
        self.state = orig_cluster.state
        # we need to copy the props in case we need to read a property
        self.props = orig_cluster.props
        self._build_migrated_cluster(orig_cluster)

    def find_matches_for_node(self) -> dict:
        """
        Maps the CPU instance types to GPU types
        :return: a map converting CPU machines to GPU ones.
        """
        mc_map = {}
        supported_gpus = self.platform.get_supported_gpus()
        for spark_node_type, node_list in self.nodes.items():
            if spark_node_type == SparkNodeType.MASTER:
                # skip
                self.cli.logger.debug('Skip converting Master nodes')
            else:
                for anode in node_list:
                    if anode.instance_type in supported_gpus:
                        continue
                    if anode.instance_type not in mc_map:
                        best_mc_match = anode.find_best_cpu_conversion(supported_gpus)
                        mc_map.update({anode.instance_type: best_mc_match})
        return mc_map

    def get_all_spark_properties(self) -> dict:
        """Returns a dictionary containing the spark configurations defined in the cluster properties"""
        raise NotImplementedError

    def get_nodes_cnt(self, node_type: SparkNodeType) -> int:
        node_values = self.nodes.get(node_type)
        if isinstance(node_values, list):
            res = len(node_values)
        else:
            res = 1
        return res

    def get_node(self, node_type: SparkNodeType) -> ClusterNode:
        node_values = self.nodes.get(node_type)
        if isinstance(node_values, list):
            res = node_values[0]
        else:
            res = node_values
        return res

    def get_all_nodes(self) -> list:
        nodes = []

        for value in self.nodes.values():
            if isinstance(value, list):
                nodes += value
            else:
                nodes += [value]

        return nodes

    def get_master_node(self) -> ClusterNode:
        return self.nodes.get(SparkNodeType.MASTER)

    def get_worker_node(self, ind: int = 0) -> ClusterNode:
        return self.nodes.get(SparkNodeType.WORKER)[ind]

    def get_name(self) -> str:
        return self.name

    def get_tmp_storage(self) -> str:
        raise NotImplementedError

    def get_image_version(self) -> str:
        raise NotImplementedError

    def _set_render_args_create_template(self) -> dict:
        cluster_config = self.get_cluster_configuration()
        return {
            'CLUSTER_NAME': self.get_name(),
            'REGION': self.region,
            'ZONE': self.zone,
            'MASTER_MACHINE': cluster_config.get('driverNodeType'),
            'WORKERS_COUNT': cluster_config.get('numWorkerNodes'),
            'WORKERS_MACHINE': cluster_config.get('workerNodeType')
        }

    @classmethod
    def _get_cluster_configuration(cls, driver_node_type: str, worker_node_type: str, num_worker_nodes: int) -> dict:
        return {
            'driverNodeType': driver_node_type,
            'workerNodeType': worker_node_type,
            'numWorkerNodes': num_worker_nodes
        }

    def get_cluster_configuration(self) -> dict:
        """
        Returns a dictionary containing the configuration of the cluster
        """
        return self._get_cluster_configuration(driver_node_type=self.get_master_node().instance_type,
                                               worker_node_type=self.get_worker_node().instance_type,
                                               num_worker_nodes=self.get_nodes_cnt(SparkNodeType.WORKER))

    def _get_gpu_configuration(self) -> dict:
        """
        Returns a dictionary containing the GPU configuration of the cluster
        """
        gpu_per_machine, gpu_device_str = self.get_gpu_per_worker()
        # Convert generic GPU device name to platform-specific name
        platform_gpu_name = self.platform.get_platform_gpu_device_name(gpu_device_str)
        # Need to handle case this was CPU event log and just make a recommendation
        if platform_gpu_name and gpu_per_machine > 0:
            return {
                'gpuInfo': {
                    'device': platform_gpu_name,
                    'gpuPerWorker': gpu_per_machine
                }
            }
        return {}

    def generate_create_script(self) -> str:
        platform_name = CspEnv.pretty_print(self.platform.type_id)
        template_path = Utils.resource_path(f'templates/{platform_name}-create_gpu_cluster_script.ms')
        render_args = self._set_render_args_create_template()
        return TemplateGenerator.render_template_file(template_path, render_args)

    def _set_render_args_bootstrap_template(self, overridden_args: dict = None) -> dict:
        res = {}
        if overridden_args:
            res.update(overridden_args)
        res.setdefault('CLUSTER_NAME', self.get_name())
        return res

    def generate_bootstrap_script(self, overridden_args: dict = None) -> str:
        platform_name = CspEnv.pretty_print(self.platform.type_id)
        template_path = Utils.resource_path(f'templates/{platform_name}-run_bootstrap.ms')
        render_args = self._set_render_args_bootstrap_template(overridden_args)
        return TemplateGenerator.render_template_file(template_path, render_args)

    def _generate_node_configuration(self, render_args: dict) -> Union[str, dict]:
        platform_name = CspEnv.pretty_print(self.platform.type_id)
        template_path = Utils.resource_path(f'templates/node_template/{platform_name}.ms')
        return json.loads(TemplateGenerator.render_template_file(template_path, render_args))

    def generate_node_configurations(self, num_executors: int, render_args: dict = None) -> list:
        """
        Generates a list of node configurations for the given number of executors.
        """
        node_config = self._generate_node_configuration(render_args)
        return [node_config for _ in range(num_executors)]

    def get_worker_conversion_str(self, include_gpu: bool = False) -> str:
        """
        Returns a string representing the worker node configuration.
        Example: '2 x g5.2xlarge'
        """
        num_workers = self.get_workers_count()
        worker_node_type = self.get_worker_node().instance_type
        worker_conversion_str = f'{num_workers} x {worker_node_type}'
        if include_gpu and self.is_gpu_cluster():
            gpu_str = self._get_gpu_conversion_str()
            return f'{worker_conversion_str} {gpu_str}'
        return worker_conversion_str

    def _get_gpu_conversion_str(self) -> str:
        """
        Returns a string representing the GPU configuration.
        Example: '(1 L4 each)'
        """
        gpu_per_machine, gpu_device_str = self.get_gpu_per_worker()
        return f'({gpu_per_machine} {gpu_device_str} each)'


@dataclass
class ClusterReshape(ClusterGetAccessor):
    """
    A class that handles reshaping of the given cluster.
    It takes argument a cluster object and callable methods that defines
    the way each cluster property is being reshaped.
    By default, the methods will have no effect on the properties.
    The caller can override the behavior by passing a callback method.
    The caller also can control which node type is affected by the reshap-methods.
    This can be done by setting the "node_types". By default, the reshaping
    is limited to the worker nodes of a cluster.
    """

    cluster_inst: ClusterBase
    node_types: List[SparkNodeType] = field(default_factory=lambda: [SparkNodeType.WORKER])
    reshape_workers_mc_type: Callable[[str], str] = field(default_factory=lambda: lambda x: x)
    reshape_workers_cnt: Callable[[int], int] = field(default_factory=lambda: lambda x: x)
    reshape_workers_cpus: Callable[[int], int] = field(default_factory=lambda: lambda x: x)
    reshape_workers_mem: Callable[[int], int] = field(default_factory=lambda: lambda x: x)
    reshape_workers_gpu_cnt: Callable[[int], int] = field(default_factory=lambda: lambda x: x)
    reshape_workers_gpu_device: Callable[[str], str] = field(default_factory=lambda: lambda x: x)

    def get_node(self, node_type: SparkNodeType) -> ClusterNode:
        if node_type == SparkNodeType.WORKER:
            return self.cluster_inst.get_worker_node()
        return self.cluster_inst.get_master_node()

    def get_all_nodes(self) -> list:
        raise NotImplementedError

    def get_node_instance_type(self, node_type: SparkNodeType) -> str:
        res = super().get_node_instance_type(node_type)
        if node_type in self.node_types:
            return self.reshape_workers_mc_type(res)
        return res

    def get_nodes_cnt(self, node_type: SparkNodeType) -> int:
        res = self.cluster_inst.get_nodes_cnt(node_type)
        if node_type in self.node_types:
            return self.reshape_workers_cnt(res)
        return res

    def get_node_core_count(self, node_type: SparkNodeType) -> int:
        res = super().get_node_core_count(node_type)
        if node_type in self.node_types:
            return self.reshape_workers_cpus(res)
        return res

    def get_node_mem_mb(self, node_type: SparkNodeType) -> int:
        res = super().get_node_mem_mb(node_type)
        if node_type in self.node_types:
            return self.reshape_workers_mem(res)
        return res

    def get_gpu_per_node(self, node_type: SparkNodeType) -> (int, str):
        num_gpus, gpu_device = super().get_gpu_per_node(node_type)
        if node_type in self.node_types:
            return self.reshape_workers_gpu_cnt(num_gpus), self.reshape_workers_gpu_device(gpu_device)
        return num_gpus, gpu_device

    def get_name(self) -> str:
        return self.cluster_inst.get_name()


def get_platform(platform_id: Enum) -> Type[PlatformBase]:
    platform_hash = {
        CspEnv.DATABRICKS_AWS: ('spark_rapids_pytools.cloud_api.databricks_aws', 'DBAWSPlatform'),
        CspEnv.DATABRICKS_AZURE: ('spark_rapids_pytools.cloud_api.databricks_azure', 'DBAzurePlatform'),
        CspEnv.DATAPROC: ('spark_rapids_pytools.cloud_api.dataproc', 'DataprocPlatform'),
        CspEnv.DATAPROC_GKE: ('spark_rapids_pytools.cloud_api.dataproc_gke', 'DataprocGkePlatform'),
        CspEnv.EMR: ('spark_rapids_pytools.cloud_api.emr', 'EMRPlatform'),
        CspEnv.ONPREM: ('spark_rapids_pytools.cloud_api.onprem', 'OnPremPlatform'),
    }
    if platform_id in platform_hash:
        mod_name, clz_name = platform_hash[platform_id]
        imported_mod = __import__(mod_name, globals(), locals(), [clz_name])
        return getattr(imported_mod, clz_name)
    raise AttributeError(f'Provider {platform_id} does not exist')
