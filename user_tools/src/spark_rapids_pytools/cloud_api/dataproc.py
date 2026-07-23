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


"""Implementation specific to Dataproc"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Union, Optional

from spark_rapids_tools import CspEnv
from spark_rapids_tools.utils import AbstractPropContainer
from spark_rapids_pytools.cloud_api.dataproc_job import DataprocLocalRapidsJob
from spark_rapids_pytools.cloud_api.gstorage import GStorageDriver
from spark_rapids_pytools.cloud_api.sp_types import PlatformBase, CMDDriverBase, \
    ClusterBase, ClusterNode, SysInfo, GpuHWInfo, SparkNodeType, ClusterState, GpuDevice, \
    NodeHWInfo, ClusterGetAccessor
from spark_rapids_pytools.common.prop_manager import is_valid_gpu_device
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_pytools.pricing.dataproc_pricing import DataprocPriceProvider
from spark_rapids_pytools.pricing.price_provider import SavingsEstimator


def _load_json_props(prop_arg: Any) -> Any:
    if not isinstance(prop_arg, str):
        return prop_arg
    try:
        return json.loads(prop_arg)
    except json.JSONDecodeError as exc:
        raise RuntimeError('Incorrect format of JSON File') from exc
    except TypeError as exc:
        raise RuntimeError('Incorrect Type of JSON content') from exc


@dataclass
class DataprocPlatform(PlatformBase):
    """
    Represents the interface and utilities required by Dataproc.
    Prerequisites:
    - install gcloud command lines (gcloud, gsutil)
    - configure the gcloud CLI.
    - dataproc has staging temporary storage. we can retrieve that from the cluster properties.
    """

    def __post_init__(self):
        self.type_id = CspEnv.DATAPROC
        self.cluster_inference_supported = True
        super().__post_init__()

    def _set_remaining_configuration_list(self) -> None:
        remaining_props = self._get_config_environment('loadedConfigProps')
        if not remaining_props:
            return
        properties_map_arr = self._get_config_environment('cliConfig',
                                                          'confProperties',
                                                          'propertiesMap')
        if properties_map_arr:
            # We support multiple gcloud CLI configurations, the following two dictionaries map
            # config sections to the corresponding property keys to be set, and config file name respectively
            config_section_keys = defaultdict(list)  # config section: keys
            config_section_file = {}  # config section: config file
            for prop_elem in properties_map_arr:
                if prop_elem.get('confProperty') in remaining_props:
                    config_section = prop_elem.get('section').strip('_')
                    # The property uses the default value which is '_cliConfigFile_'
                    config_file = prop_elem.get('configFileProp', '_cliConfigFile_').strip('_')
                    config_section_keys[config_section].append(prop_elem.get('propKey'))
                    if config_section not in config_section_file:
                        config_section_file[config_section] = config_file
            loaded_conf_dict = {}
            for config_section in config_section_keys:
                loaded_conf_dict = \
                    self._load_props_from_sdk_conf_file(keyList=config_section_keys[config_section],
                                                        configFile=config_section_file[config_section],
                                                        sectionKey=config_section)
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

    def _construct_cli_object(self) -> CMDDriverBase:
        return DataprocCMDDriver(timeout=0, cloud_ctxt=self.ctxt)

    def _install_storage_driver(self):
        self.storage = GStorageDriver(self.cli)

    def _construct_cluster_from_props(self, cluster: str, props: str = None, is_inferred: bool = False,
                                      is_props_file: bool = False):
        return DataprocCluster(self, is_inferred=is_inferred).set_connection(cluster_id=cluster, props=props)

    def set_offline_cluster(self, cluster_args: dict = None):
        pass

    def migrate_cluster_to_gpu(self, orig_cluster):
        """
        given a cluster, convert it to run NVIDIA Gpu based on mapping instance types
        :param orig_cluster: the original cluster to migrate from
        :return: a new object cluster that supports GPU.
        """
        gpu_cluster_ob = DataprocCluster(self)
        gpu_cluster_ob.migrate_from_cluster(orig_cluster)
        return gpu_cluster_ob

    def create_saving_estimator(self,
                                source_cluster: ClusterGetAccessor,
                                reshaped_cluster: ClusterGetAccessor,
                                target_cost: float = None,
                                source_cost: float = None):
        raw_pricing_config = self.configs.get_value_silent('pricing')
        pricing_config: Optional[AbstractPropContainer] = None
        if raw_pricing_config:
            pricing_config = AbstractPropContainer(props=_load_json_props(raw_pricing_config))
        pricing_provider = DataprocPriceProvider(region=self.cli.get_region(),
                                                 pricing_configs={'gcloud': pricing_config})
        saving_estimator = DataprocSavingsEstimator(price_provider=pricing_provider,
                                                    reshaped_cluster=reshaped_cluster,
                                                    source_cluster=source_cluster,
                                                    target_cost=target_cost,
                                                    source_cost=source_cost)
        return saving_estimator

    def create_local_submission_job(self, job_prop, ctxt) -> Any:
        return DataprocLocalRapidsJob(prop_container=job_prop, exec_ctxt=ctxt)

    def create_distributed_submission_job(self, job_prop, ctxt) -> Any:
        pass

    def validate_job_submission_args(self, submission_args: dict) -> dict:
        pass

    def get_supported_gpus(self) -> dict:
        def calc_num_gpus(gpus_criteria_conf: List[dict], num_cores: int) -> int:
            if gpus_criteria_conf:
                for c_conf in gpus_criteria_conf:
                    if c_conf.get('lowerBound') <= num_cores < c_conf.get('upperBound'):
                        return c_conf.get('gpuCount')
            # Use default if the configuration is not loaded. This should not happen anyway.
            return 2 if num_cpu >= 16 else 1

        gpus_from_configs = self.configs.get_value('gpuConfigs', 'user-tools', 'supportedGpuInstances')
        gpu_count_criteria = self.configs.get_value('gpuConfigs',
                                                    'user-tools',
                                                    'gpuPerMachine', 'criteria', 'numCores')
        gpu_scopes = {}
        for mc_prof, mc_info in gpus_from_configs.items():
            unit_info = mc_info['seriesInfo']
            for num_cpu in unit_info['vCPUs']:
                prof_name = f'{mc_prof}-{num_cpu}'
                # create the sys info
                memory_mb = num_cpu * unit_info['memPerCPU']
                sys_info_obj = SysInfo(num_cpus=num_cpu, cpu_mem=memory_mb)
                # create gpu_info
                gpu_cnt = calc_num_gpus(gpu_count_criteria, num_cpu)
                # default memory
                gpu_device = GpuDevice.get_default()
                gpu_mem = gpu_device.get_gpu_mem()[0]
                gpu_info_obj = GpuHWInfo(num_gpus=gpu_cnt, gpu_mem=gpu_mem, gpu_device=gpu_device)
                gpu_scopes[prof_name] = NodeHWInfo(sys_info=sys_info_obj, gpu_info=gpu_info_obj)
        return gpu_scopes

    def get_matching_worker_node_type(self, total_cores: int) -> Optional[str]:
        node_types_from_config = self.configs.get_value('clusterInference', 'defaultCpuInstances', 'executor')
        # TODO: Currently only single series is supported. Change this to a loop when using multiple series.
        series_name, unit_info = list(node_types_from_config.items())[0]
        if total_cores in unit_info['vCPUs']:
            return f'{series_name}-{total_cores}'
        return None

    def generate_cluster_configuration(self, render_args: dict):
        image_version = self.configs.get_value('clusterInference', 'defaultImage')
        render_args['IMAGE'] = f'"{image_version}"'
        render_args['ZONE'] = f'"{self.cli.get_zone()}"'
        return super().generate_cluster_configuration(render_args)

    def get_platform_gpu_device_name(self, gpu_device_str: str) -> str:
        """
        Convert generic GPU device name to Dataproc-specific accelerator name.
        Reference: https://docs.cloud.google.com/dataproc/docs/concepts/compute/gpus#types_of_gpus

        :param gpu_device_str: Generic GPU device name (e.g., "T4", "L4")
        :return: Dataproc-specific accelerator name (e.g., "nvidia-tesla-t4", "nvidia-l4")
        """
        # Mapping of generic GPU device names to Dataproc accelerator names
        gpu_device_map = {
            'L4': 'nvidia-l4',
            'A100': 'nvidia-a100-80gb',
            'P100': 'nvidia-tesla-p100',
            'V100': 'nvidia-tesla-v100',
            'P4': 'nvidia-tesla-p4',
            'T4': 'nvidia-tesla-t4'
        }
        return gpu_device_map.get(gpu_device_str, gpu_device_str)


@dataclass
class DataprocCMDDriver(CMDDriverBase):  # pylint: disable=abstract-method
    """Represents the command interface that will be used by Dataproc"""

    def _list_inconsistent_configurations(self) -> list:
        incorrect_envs = super()._list_inconsistent_configurations()
        required_props = self.get_required_props()
        if required_props:
            for prop_entry in required_props:
                prop_value = self.env_vars.get(prop_entry)
                if prop_value is None:
                    incorrect_envs.append(f'Property {prop_entry} is not set.')
        return incorrect_envs

    def _build_platform_list_cluster(self,
                                     cluster,
                                     query_args: dict = None) -> list:
        cmd_params = ['gcloud', 'dataproc', 'clusters', 'list',
                      f"--region='{self.get_region()}'"]
        filter_args = [f'clusterName = {cluster.name}']
        if query_args is not None:
            if 'state' in query_args:
                state_param = query_args.get('state')
                filter_args.append(f'status.state = {state_param}')
        filter_arg = Utils.gen_joined_str(' AND ', filter_args)
        cmd_params.append(f"--filter='{filter_arg}'")
        return cmd_params

    def pull_cluster_props_by_args(self, args: dict) -> str:
        cluster_name = args.get('cluster')
        # TODO: We should piggyback on the cmd so that we do not have to add region in each cmd
        # region is already set in the instance
        if 'region' in args:
            region_name = args.get('region')
        else:
            region_name = self.get_region()
        describe_cluster_cmd = ['gcloud',
                                'dataproc',
                                'clusters',
                                'describe',
                                cluster_name,
                                '--region',
                                region_name]
        return self.run_sys_cmd(describe_cluster_cmd)

    def exec_platform_describe_accelerator(self,
                                           accelerator_type: str,
                                           **cmd_args) -> str:
        cmd_args = ['gcloud', 'compute', 'accelerator-types', 'describe',
                    accelerator_type, '--zone',
                    self.get_env_var('zone')]
        return self.run_sys_cmd(cmd_args)

    def _build_cmd_ssh_prefix_for_node(self, node: ClusterNode) -> str:
        pref_args = ['gcloud',
                     'compute', 'ssh',
                     node.name,
                     '--zone',
                     self.get_env_var('zone'),
                     '--command=']
        return Utils.gen_joined_str(' ', pref_args)

    def _build_cmd_scp_to_node(self, node: ClusterNode, src: str, dest: str) -> str:
        pref_args = ['gcloud',
                     'compute', 'scp',
                     '--zone',
                     self.get_env_var('zone'),
                     src,
                     f'{node.name}:{dest}']
        return Utils.gen_joined_str(' ', pref_args)

    def _build_cmd_scp_from_node(self, node: ClusterNode, src: str, dest: str) -> str:
        pref_args = ['gcloud',
                     'compute', 'scp',
                     '--zone',
                     self.get_env_var('zone'),
                     f'{node.name}:{src}',
                     dest]
        return Utils.gen_joined_str(' ', pref_args)

    def _construct_ssh_cmd_with_prefix(self, prefix: str, remote_cmd: str) -> str:
        # for dataproc, the remote should not be preceded by ws
        return f'{prefix}{remote_cmd}'

    def get_submit_spark_job_cmd_for_cluster(self,
                                             cluster_name: str,
                                             submit_args: dict) -> List[str]:
        cmd = ['gcloud',
               'dataproc',
               'jobs',
               'submit',
               'spark',
               '--cluster',
               cluster_name,
               '--region',
               self.get_region()]
        # add the platform arguments: jars, class
        if 'platformSparkJobArgs' in submit_args:
            for arg_k, arg_val in submit_args.get('platformSparkJobArgs').items():
                if arg_val:
                    cmd.append(f'--{arg_k}={arg_val}')
        # add the jar arguments
        jar_args = submit_args.get('jarArgs')
        if jar_args:
            cmd.append('--')
            # expects a list of string
            cmd.extend(jar_args)
        return cmd

    def _process_instance_description(self, instance_descriptions: str) -> dict:
        def extract_gpu_name(gpu_description: str) -> str:
            gpu_name = ''
            for elem in gpu_description.split('-'):
                if is_valid_gpu_device(elem):
                    gpu_name = elem
                    break
            return gpu_name.upper()

        processed_instance_descriptions = {}
        raw_instances_descriptions = AbstractPropContainer(props=_load_json_props(instance_descriptions))
        for instance in raw_instances_descriptions.props:
            instance_content = {}
            instance_content['VCpuCount'] = int(instance.get('guestCpus', -1))
            instance_content['MemoryInMB'] = int(instance.get('memoryMb', -1))
            if 'accelerators' in instance:
                raw_accelerator_info = instance['accelerators'][0]
                gpu_name = extract_gpu_name(raw_accelerator_info.get('guestAcceleratorType'))
                if gpu_name != '':
                    gpu_count = int(raw_accelerator_info.get('guestAcceleratorCount', -1))
                    gpu_info = {'Name': gpu_name, 'Count': [gpu_count]}
                    instance_content['GpuInfo'] = [gpu_info]
            processed_instance_descriptions[instance.get('name')] = instance_content

        # for Dataproc, some instance types can attach customized GPU devices
        # Ref: https://cloud.google.com/compute/docs/gpus#n1-gpus
        for instance_name, instance_info in processed_instance_descriptions.items():
            if instance_name.startswith('n1-'):
                if 'GpuInfo' not in instance_info:
                    instance_info['GpuInfo'] = []
                # N1 + T4 GPUs
                if 1 <= instance_info['VCpuCount'] <= 48:
                    t4_gpu_info = {'Name': 'T4', 'Count': [1, 2, 4]}
                else:  # 48 < VCpuCount <= 96
                    t4_gpu_info = {'Name': 'T4', 'Count': [4]}
                instance_info['GpuInfo'].append(t4_gpu_info)
                # N1 + P4 GPUs
                if 1 <= instance_info['VCpuCount'] <= 24:
                    p4_gpu_info = {'Name': 'P4', 'Count': [1, 2, 4]}
                elif 24 < instance_info['VCpuCount'] <= 48:
                    p4_gpu_info = {'Name': 'P4', 'Count': [2, 4]}
                else:  # 48 < VCpuCount <= 96
                    p4_gpu_info = {'Name': 'P4', 'Count': [4]}
                instance_info['GpuInfo'].append(p4_gpu_info)
                # N1 + V100 GPUs
                if 1 <= instance_info['VCpuCount'] <= 12:
                    v100_gpu_info = {'Name': 'V100', 'Count': [1, 2, 4, 8]}
                elif 12 < instance_info['VCpuCount'] <= 24:
                    v100_gpu_info = {'Name': 'V100', 'Count': [2, 4, 8]}
                elif 24 < instance_info['VCpuCount'] <= 48:
                    v100_gpu_info = {'Name': 'V100', 'Count': [4, 8]}
                else:  # 48 < VCpuCount <= 96
                    v100_gpu_info = {'Name': 'V100', 'Count': [8]}
                instance_info['GpuInfo'].append(v100_gpu_info)
                # N1 + P100 GPUs
                if 1 <= instance_info['VCpuCount'] <= 16:
                    p100_gpu_info = {'Name': 'P100', 'Count': [1, 2, 4]}
                elif 16 < instance_info['VCpuCount'] <= 32:
                    p100_gpu_info = {'Name': 'P100', 'Count': [2, 4]}
                else:  # 32 < VCpuCount <= 96
                    p100_gpu_info = {'Name': 'P100', 'Count': [4]}
                instance_info['GpuInfo'].append(p100_gpu_info)
        return processed_instance_descriptions

    def get_instance_description_cli_params(self) -> list:
        return ['gcloud compute machine-types list', '--zones', f'{self.get_zone()}']


@dataclass
class DataprocNode(ClusterNode):
    """Implementation of Dataproc cluster node."""

    zone: str = field(default=None, init=False)

    @staticmethod
    def __extract_info_from_value(conf_val: str):
        if '/' in conf_val:
            # this is a valid url-path
            return FSUtil.get_resource_name(conf_val)
        # this is a value
        return conf_val

    def _pull_gpu_hw_info(self, cli=None) -> Optional[GpuHWInfo]:
        # gcloud GPU machines: https://cloud.google.com/compute/docs/gpus
        def get_gpu_device(accelerator_name: str) -> Optional[GpuDevice]:
            """
            return the GPU device given a accelerator full name (e.g. nvidia-tesla-t4)
            """
            accelerator_name_arr = accelerator_name.split('-')
            for elem in accelerator_name_arr:
                try:
                    return GpuDevice.fromstring(elem)
                except ValueError:
                    # Continue to next entry if the accelerator name is not a valid GPU device
                    continue
            return None

        # check if GPU info is already set
        if self.hw_info and self.hw_info.gpu_info:
            return self.hw_info.gpu_info

        # node has no 'accelerators' info, need to pull GPU info from instance catalog
        if self.props is None or self.props.get_value_silent('accelerators') is None:
            if self.instance_type.startswith('n1') or self.mc_props is None or \
                    self.mc_props.get_value_silent('GpuInfo') is None:
                # TODO: for n1-series, it can attach different type or number of GPU device
                # GPU info is only accurate when 'accelerators' is set, return None as default
                return None
            gpu_configs = self.mc_props.get_value('GpuInfo')[0]
            num_gpus = gpu_configs['Count'][0]
            gpu_device = GpuDevice.fromstring(gpu_configs['Name'])
            gpu_mem = gpu_device.get_gpu_mem()[0]
            return GpuHWInfo(num_gpus=num_gpus,
                             gpu_device=gpu_device,
                             gpu_mem=gpu_mem)

        accelerator_arr = self.props.get_value('accelerators')
        for defined_acc in accelerator_arr:
            # TODO: if the accelerator_arr has other non-gpu ones, then we need to loop until we
            #       find the gpu accelerators
            accelerator_type = defined_acc.get('acceleratorTypeUri') or defined_acc.get('acceleratorType')
            accelerator_full_name = self.__extract_info_from_value(accelerator_type)
            gpu_device = get_gpu_device(accelerator_full_name)
            if gpu_device is None:
                continue
            num_gpus = int(defined_acc.get('acceleratorCount'))
            gpu_mem = gpu_device.get_gpu_mem()[0]
            return GpuHWInfo(num_gpus=num_gpus,
                             gpu_device=gpu_device,
                             gpu_mem=gpu_mem)
        return None

    def _set_fields_from_props(self):
        # set the machine type
        if not self.props:
            return
        mc_type_uri = self.props.get_value_silent('machineTypeUri')
        if mc_type_uri:
            self.instance_type = self.__extract_info_from_value(mc_type_uri)
        else:
            # check if the machine type is  under a different name
            mc_type = self.props.get_value('machineType')
            if mc_type:
                self.instance_type = self.__extract_info_from_value(mc_type)


@dataclass
class DataprocCluster(ClusterBase):
    """
    Represents an instance of running cluster on Dataproc.
    """

    def _get_temp_gs_storage(self) -> str:
        temp_bucket = self.props.get_value_silent('config', 'tempBucket')
        if temp_bucket:
            return f'gs://{temp_bucket}/{self.uuid}'
        return None

    def get_eventlogs_from_config(self) -> List[str]:
        res_arr = super().get_eventlogs_from_config()
        if not res_arr:
            # The SHS was not set for the cluster. Use the tmp bucket storage as the default SHS log directory
            # append the temporary gstorage followed by the SHS folder
            tmp_gs = self._get_temp_gs_storage()
            res_arr.append(f'{tmp_gs}/spark-job-history')
        return res_arr

    def _set_fields_from_props(self):
        super()._set_fields_from_props()
        self.uuid = self.props.get_value('clusterUuid')
        self.state = ClusterState.fromstring(self.props.get_value('status', 'state'))

    def _set_name_from_props(self) -> None:
        self.name = self.props.get_value('clusterName')

    def _init_nodes(self):
        # assume that only one master node
        master_nodes_from_conf = self.props.get_value('config', 'masterConfig', 'instanceNames')
        raw_worker_prop = self.props.get_value_silent('config', 'workerConfig')
        worker_nodes: list = []
        if raw_worker_prop:
            worker_cnt = self.props.get_value('config', 'workerConfig', 'numInstances')
            worker_nodes_from_conf = self.props.get_value_silent('config', 'workerConfig', 'instanceNames')
            instance_names_cnt = len(worker_nodes_from_conf) if worker_nodes_from_conf else 0
            if worker_cnt != instance_names_cnt:
                if not self.is_inferred:
                    # this warning should be raised only when the cluster is not inferred, i.e. user has provided the
                    # cluster configuration with num_workers explicitly set
                    self.logger.warning('Cluster configuration: `instanceNames` count %d does not '
                                        'match the `numInstances` value %d. Using generated names.',
                                        instance_names_cnt, worker_cnt)
                worker_nodes_from_conf = self.generate_node_configurations(worker_cnt)
            # create workers array
            for worker_node in worker_nodes_from_conf:
                worker_props = {
                    'name': worker_node,
                    'props': AbstractPropContainer(props=raw_worker_prop),
                    # set the node zone based on the wrapper defined zone
                    'zone': self.zone
                }
                worker = DataprocNode.create_worker_node().set_fields_from_dict(worker_props)
                # TODO for optimization, we should set HW props for 1 worker
                worker.fetch_and_set_hw_info(self.cli)
                worker_nodes.append(worker)
        raw_master_props = self.props.get_value('config', 'masterConfig')
        master_props = {
            'name': master_nodes_from_conf[0],
            'props': AbstractPropContainer(props=raw_master_props),
            # set the node zone based on the wrapper defined zone
            'zone': self.zone
        }
        master_node = DataprocNode.create_master_node().set_fields_from_dict(master_props)
        master_node.fetch_and_set_hw_info(self.cli)
        self.nodes = {
            SparkNodeType.WORKER: worker_nodes,
            SparkNodeType.MASTER: master_node
        }

    def _set_zone_from_props(self, prop_container: AbstractPropContainer):
        """
        Extracts the 'zoneUri' from the properties container and updates the environment variable dictionary.
        """
        if prop_container:
            zone_uri = prop_container.get_value_silent('config', 'gceClusterConfig', 'zoneUri')
            if zone_uri:
                self.cli.env_vars['zone'] = FSUtil.get_resource_name(zone_uri)

    def _init_connection(self, cluster_id: str = None,
                         props: str = None) -> dict:
        cluster_args = super()._init_connection(cluster_id=cluster_id, props=props)
        # extract and update zone to the environment variable
        self._set_zone_from_props(cluster_args['props'])
        # propagate zone to the cluster
        cluster_args.setdefault('zone', self.cli.get_env_var('zone'))
        return cluster_args

    def _build_migrated_cluster(self, orig_cluster):
        """
        specific to the platform on how to build a cluster based on migration
        :param orig_cluster: the cpu_cluster that does not support the GPU devices.
        """
        # get the map of the instance types
        supported_mc_map = orig_cluster.platform.get_supported_gpus()
        mc_type_map = orig_cluster.find_matches_for_node()
        new_worker_nodes: list = []
        for anode in orig_cluster.nodes.get(SparkNodeType.WORKER):
            # loop on all worker nodes.
            # even if the node is the same type, we still need to set the hardware
            if anode.instance_type not in mc_type_map:
                # the node stays the same
                # skip converting the node
                new_instance_type = anode.instance_type
                self.logger.info('Node with %s supports GPU devices.',
                                 anode.instance_type)
            else:
                new_instance_type = mc_type_map.get(anode.instance_type)
                self.logger.info('Converting node %s into GPU supported instance-type %s',
                                 anode.instance_type,
                                 new_instance_type)
            worker_props = {
                'instance_type': new_instance_type,
                'name': anode.name,
                'zone': anode.zone,
            }
            new_node = DataprocNode.create_worker_node().set_fields_from_dict(worker_props)
            # we cannot rely on setting gpu info from the SDK because
            # dataproc does not bind machine types to GPUs
            # new_node.fetch_and_set_hw_info(self.cli)
            gpu_mc_hw: ClusterNode = supported_mc_map.get(new_instance_type)
            new_node.construct_hw_info(cli=None,
                                       gpu_info=gpu_mc_hw.gpu_info,
                                       sys_info=gpu_mc_hw.sys_info)
            new_worker_nodes.append(new_node)
        self.nodes = {
            SparkNodeType.WORKER: new_worker_nodes,
            SparkNodeType.MASTER: orig_cluster.nodes.get(SparkNodeType.MASTER)
        }
        if bool(mc_type_map):
            # update the platform notes
            self.platform.update_ctxt_notes('nodeConversions', mc_type_map)

    def get_all_spark_properties(self) -> dict:
        """Returns a dictionary containing the spark configurations defined in the cluster properties"""
        sw_props = self.props.get_value_silent('config', 'softwareConfig', 'properties')
        if sw_props:
            k_prefix = 'spark:'
            return {key[len(k_prefix):]: value for (key, value) in sw_props.items() if key.startswith(k_prefix)}
        return {}

    def get_tmp_storage(self) -> str:
        return self._get_temp_gs_storage()

    def get_image_version(self) -> str:
        return self.props.get_value_silent('config', 'softwareConfig', 'imageVersion')

    def _set_render_args_create_template(self) -> dict:
        render_args = super()._set_render_args_create_template()
        cluster_config = self.get_cluster_configuration()
        render_args['IMAGE'] = self.get_image_version()
        render_args['LOCAL_SSD'] = cluster_config.get('additionalConfig').get('localSsd')
        render_args['GPU_DEVICE'] = cluster_config.get('gpuInfo').get('device')
        render_args['GPU_PER_WORKER'] = cluster_config.get('gpuInfo').get('gpuPerWorker')
        return render_args

    def get_cluster_configuration(self) -> dict:
        """
        Overrides to provide the cluster configuration which is specific to Dataproc.
        """
        cluster_config = super().get_cluster_configuration()
        # If the cluster is GPU cluster, we need to add the GPU configuration
        if self.is_gpu_cluster():
            gpu_config = self._get_gpu_configuration()
            cluster_config.update(gpu_config)
            ssd_config = self._get_ssd_configuration()
            cluster_config.update(ssd_config)
        return cluster_config

    @classmethod
    def _get_ssd_configuration(cls) -> dict:
        """
        Currently, we recommend either `g2-standard` and `n1-standard` series which
        supports minimum 1 SSD per worker.

        TODO: We should recommend correct number of SSDs instead of a fixed number.
         See https://github.com/NVIDIA/spark-rapids-tools/issues/657
        """
        return {
            'ssdInfo': {
                'ssdPerWorker': 1
            }
        }

    def _generate_node_configuration(self, render_args: dict = None) -> Union[str, dict]:
        """
        Overrides to provide the cluster node configuration which is node name
        in case of Dataproc.
        """
        return 'test-node-e'

    def get_worker_conversion_str(self, include_gpu: bool = True) -> str:
        """
        Overrides to provide the worker conversion string which is specific to Dataproc.
        Example: '2 x n1-standard-32 (4 T4 each)'
        """
        return super().get_worker_conversion_str(include_gpu)


@dataclass
class DataprocSavingsEstimator(SavingsEstimator):
    """
    A class that calculates the savings based on Dataproc price provider
    """
    def _calculate_group_cost(self, cluster_inst: ClusterGetAccessor, node_type: SparkNodeType):
        nodes_cnt = cluster_inst.get_nodes_cnt(node_type)
        cores_count = cluster_inst.get_node_core_count(node_type)
        mem_mb = cluster_inst.get_node_mem_mb(node_type)
        node_mc_type = cluster_inst.get_node_instance_type(node_type)
        # memory here is in mb, we need to convert it to gb
        mem_gb = float(mem_mb) / 1024
        cores_cost = self.price_provider.get_cpu_price(node_mc_type) * int(cores_count)
        memory_cost = self.price_provider.get_ram_price(node_mc_type) * mem_gb
        # calculate the GPU cost
        gpu_per_machine, gpu_type = cluster_inst.get_gpu_per_node(node_type)
        gpu_cost = 0.0
        if gpu_per_machine > 0:
            gpu_unit_price = self.price_provider.get_gpu_price(gpu_type)
            gpu_cost = gpu_unit_price * gpu_per_machine
        return nodes_cnt * (cores_cost + memory_cost + gpu_cost)

    def _get_cost_per_cluster(self, cluster: ClusterGetAccessor):
        master_cost = self._calculate_group_cost(cluster, SparkNodeType.MASTER)
        workers_cost = self._calculate_group_cost(cluster, SparkNodeType.WORKER)
        master_cores = cluster.get_nodes_cnt(SparkNodeType.MASTER) * cluster.get_node_core_count(SparkNodeType.MASTER)
        worker_cores = cluster.get_nodes_cnt(SparkNodeType.WORKER) * cluster.get_node_core_count(SparkNodeType.WORKER)
        dataproc_cost = self.price_provider.get_container_cost() * (master_cores + worker_cores)
        return master_cost + workers_cost + dataproc_cost
