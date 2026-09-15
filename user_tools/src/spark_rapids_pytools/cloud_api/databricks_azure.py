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

"""Implementation specific to DATABRICKS_AZURE"""

import json
from dataclasses import dataclass, field
from typing import Any, List

from spark_rapids_tools import CspEnv
from spark_rapids_pytools.cloud_api.azurestorage import AzureStorageDriver
from spark_rapids_pytools.cloud_api.databricks_azure_job import DBAzureLocalRapidsJob
from spark_rapids_pytools.cloud_api.sp_types import CMDDriverBase, ClusterBase, ClusterNode, \
    PlatformBase, SysInfo, GpuHWInfo, ClusterState, SparkNodeType, ClusterGetAccessor, NodeHWInfo, GpuDevice
from spark_rapids_pytools.common.prop_manager import JSONPropertiesContainer
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_pytools.pricing.databricks_azure_pricing import DatabricksAzurePriceProvider
from spark_rapids_pytools.pricing.price_provider import SavingsEstimator


@dataclass
class DBAzurePlatform(PlatformBase):
    """
    Represents the interface and utilities required by DATABRICKS_AZURE.
    Prerequisites:
    - install databricks, azure command lines (databricks cli, azure cli)
    - configure the databricks cli (token, workspace, profile)
    - configure the azure cli
    """
    def __post_init__(self):
        self.type_id = CspEnv.DATABRICKS_AZURE
        self.cluster_inference_supported = True
        super().__post_init__()

    def _construct_cli_object(self) -> CMDDriverBase:
        return DBAzureCMDDriver(configs=self.configs, timeout=0, cloud_ctxt=self.ctxt)

    def _install_storage_driver(self):
        self.storage = AzureStorageDriver(self.cli)

    def _construct_cluster_from_props(self, cluster: str, props: str = None, is_inferred: bool = False,
                                      is_props_file: bool = False):
        return DatabricksAzureCluster(self, is_inferred=is_inferred).set_connection(cluster_id=cluster, props=props)

    def set_offline_cluster(self, cluster_args: dict = None):
        pass

    def migrate_cluster_to_gpu(self, orig_cluster):
        """
        given a cluster, convert it to run NVIDIA Gpu based on mapping instance types
        :param orig_cluster: the original cluster to migrate from
        :return: a new object cluster that supports GPU
        """
        gpu_cluster_ob = DatabricksAzureCluster(self)
        gpu_cluster_ob.migrate_from_cluster(orig_cluster)
        return gpu_cluster_ob

    def create_saving_estimator(self,
                                source_cluster: ClusterGetAccessor,
                                reshaped_cluster: ClusterGetAccessor,
                                target_cost: float = None,
                                source_cost: float = None):
        raw_pricing_config = self.configs.get_value_silent('pricing')
        if raw_pricing_config:
            pricing_config = JSONPropertiesContainer(prop_arg=raw_pricing_config, file_load=False)
        else:
            pricing_config: JSONPropertiesContainer = None
        db_azure_price_provider = DatabricksAzurePriceProvider(region=self.cli.get_region(),
                                                               pricing_configs={'databricks-azure': pricing_config})
        saving_estimator = DBAzureSavingsEstimator(price_provider=db_azure_price_provider,
                                                   reshaped_cluster=reshaped_cluster,
                                                   source_cluster=source_cluster,
                                                   target_cost=target_cost,
                                                   source_cost=source_cost)
        return saving_estimator

    def create_local_submission_job(self, job_prop, ctxt) -> Any:
        return DBAzureLocalRapidsJob(prop_container=job_prop, exec_ctxt=ctxt)

    def create_distributed_submission_job(self, job_prop, ctxt) -> Any:
        pass

    def validate_job_submission_args(self, submission_args: dict) -> dict:
        pass

    def get_supported_gpus(self) -> dict:
        gpus_from_configs = self.configs.get_value('gpuConfigs', 'user-tools', 'supportedGpuInstances')
        gpu_scopes = {}
        for mc_prof, mc_info in gpus_from_configs.items():
            hw_info_json = mc_info['SysInfo']
            hw_info_ob = SysInfo(num_cpus=hw_info_json['num_cpus'], cpu_mem=hw_info_json['cpu_mem'])
            gpu_info_json = mc_info['GpuInfo']['GPUs'][0]
            gpu_info_obj = GpuHWInfo(num_gpus=gpu_info_json['Count'], gpu_mem=gpu_info_json['MemoryInfo']['SizeInMiB'])
            gpu_scopes[mc_prof] = NodeHWInfo(sys_info=hw_info_ob, gpu_info=gpu_info_obj)
        return gpu_scopes


@dataclass
class DBAzureCMDDriver(CMDDriverBase):
    """Represents the command interface that will be used by DATABRICKS_AZURE"""

    configs: JSONPropertiesContainer = None

    def _list_inconsistent_configurations(self) -> list:
        incorrect_envs = super()._list_inconsistent_configurations()
        return incorrect_envs

    def _build_platform_list_cluster(self, cluster, query_args: dict = None) -> list:
        pass

    def pull_cluster_props_by_args(self, args: dict) -> str:
        get_cluster_cmd = ['databricks', 'clusters', 'get']
        if 'Id' in args:
            get_cluster_cmd.extend([args.get('Id')])
        elif 'cluster' in args:
            # TODO: currently, arguments '--cpu_cluster' or '--gpu_cluster' are processed and stored as
            # 'cluster' (as cluster names), while they are actually cluster ids for databricks platforms
            get_cluster_cmd.extend([args.get('cluster')])
        else:
            self.logger.error('Unable to pull cluster id or cluster name information')

        try:
            cluster_described = self.run_sys_cmd(get_cluster_cmd)
            if cluster_described is not None:
                raw_prop_container = JSONPropertiesContainer(prop_arg=cluster_described, file_load=False)
                return json.dumps(raw_prop_container.props)
        except Exception as ex:
            self.logger.error('Invalid arguments to pull the cluster properties: %s', ex)
            raise ex

        return None

    def _build_cmd_ssh_prefix_for_node(self, node: ClusterNode) -> str:
        port = self.env_vars.get('sshPort')
        key_file = self.env_vars.get('sshKeyFile')
        prefix_args = ['ssh',
                       '-o StrictHostKeyChecking=no',
                       f'-i {key_file} ' if key_file else '',
                       f'-p {port}',
                       f'ubuntu@{node.name}']
        return Utils.gen_joined_str(' ', prefix_args)

    def _build_cmd_scp_to_node(self, node: ClusterNode, src: str, dest: str) -> str:
        port = self.env_vars.get('sshPort')
        key_file = self.env_vars.get('sshKeyFile')
        prefix_args = ['scp',
                       '-o StrictHostKeyChecking=no',
                       f'-i {key_file} ' if key_file else '',
                       f'-P {port}',
                       src,
                       f'ubuntu@{node.name}:{dest}']
        return Utils.gen_joined_str(' ', prefix_args)

    def _build_cmd_scp_from_node(self, node: ClusterNode, src: str, dest: str) -> str:
        port = self.env_vars.get('sshPort')
        key_file = self.env_vars.get('sshKeyFile')
        prefix_args = ['scp',
                       '-o StrictHostKeyChecking=no',
                       f'-i {key_file} ' if key_file else '',
                       f'-P {port}',
                       f'ubuntu@{node.name}:{src}',
                       dest]
        return Utils.gen_joined_str(' ', prefix_args)

    def _process_instance_description(self, instance_descriptions: str) -> dict:
        processed_instance_descriptions = {}
        raw_instance_descriptions = JSONPropertiesContainer(prop_arg=instance_descriptions, file_load=False)
        for instance in raw_instance_descriptions.props:
            if not instance['capabilities']:
                continue
            instance_content = {}
            gpu_count = 0
            for elem in instance['capabilities']:
                if elem['name'] == 'vCPUs':
                    instance_content['VCpuCount'] = int(elem['value'])
                elif elem['name'] == 'MemoryGB':
                    instance_content['MemoryInMB'] = int(float(elem['value']) * 1024)
                elif elem['name'] == 'GPUs':
                    gpu_count = int(elem['value'])
            if gpu_count > 0:
                instance_content['GpuInfo'] = [{'Count': [gpu_count]}]
            processed_instance_descriptions[instance['name']] = instance_content
        return processed_instance_descriptions

    def get_instance_description_cli_params(self):
        return ['az vm list-skus', '--location', f'{self.get_region()}']

    def get_submit_spark_job_cmd_for_cluster(self, cluster_name: str, submit_args: dict) -> List[str]:
        raise NotImplementedError

    def get_region(self) -> str:
        if self.env_vars.get('location'):
            return self.env_vars.get('location')
        return self.env_vars.get('region')


@dataclass
class DatabricksAzureNode(ClusterNode):
    """Implementation of Databricks Azure cluster node."""

    region: str = field(default=None, init=False)

    def _set_fields_from_props(self):
        self.name = self.props.get_value_silent('public_dns')

    def _pull_gpu_hw_info(self, cli=None) -> GpuHWInfo or None:
        gpu_info = cli.configs.get_value('gpuConfigs', 'user-tools', 'supportedGpuInstances')
        if gpu_info is None:
            return None
        if self.instance_type not in gpu_info:
            return None
        gpu_instance = gpu_info[self.instance_type]['GpuInfo']['GPUs'][0]
        gpu_device = GpuDevice.fromstring(gpu_instance['Name'])
        return GpuHWInfo(num_gpus=gpu_instance['Count'],
                         gpu_device=gpu_device,
                         gpu_mem=gpu_instance['MemoryInfo']['SizeInMiB'])


@dataclass
class DatabricksAzureCluster(ClusterBase):
    """
    Represents an instance of running cluster on Databricks.
    """

    def _set_fields_from_props(self):
        super()._set_fields_from_props()
        self.uuid = self.props.get_value('cluster_id')
        self.state = ClusterState.fromstring(self.props.get_value('state'))

    def _set_name_from_props(self) -> None:
        self.name = self.props.get_value('cluster_name')

    def _init_nodes(self):
        # assume that only one driver node
        driver_nodes_from_conf = self.props.get_value_silent('driver')
        # a zero-worker (single-node) cluster has no `executors` entry at all
        worker_nodes_from_conf = self.props.get_value_silent('executors') or []
        num_workers = self.props.get_value_silent('num_workers')
        if num_workers is None and self.props.get_value_silent('autoscale') is not None:
            target_workers = self.props.get_value_silent('autoscale', 'target_workers')
            # use min_workers since it is usually the same as target_workers
            min_workers = self.props.get_value_silent('autoscale', 'min_workers')
            if target_workers is not None:
                num_workers = target_workers
                self.logger.info('Autoscaling cluster, will set number of workers to target_workers = %s',
                                 num_workers)
            elif min_workers is not None:
                num_workers = min_workers
                self.logger.info('Autoscaling cluster, will set number of workers to min_workers = %s',
                                 num_workers)
        if num_workers is None:
            self.logger.info('Unable to find number of workers for cluster, will default to 0')
            num_workers = 0
        # construct driver node info when cluster is inactive
        if driver_nodes_from_conf is None:
            driver_node_type_id = self.props.get_value('driver_node_type_id')
            if driver_node_type_id is None:
                raise RuntimeError('Failed to find driver node information from cluster properties')
            driver_nodes_from_conf = {'node_id': None}
        # construct worker nodes info when cluster is inactive
        executors_cnt = len(worker_nodes_from_conf) if worker_nodes_from_conf else 0
        if num_workers != executors_cnt:
            if not self.is_inferred:
                # this warning should be raised only when the cluster is not inferred, i.e. user has provided the
                # cluster configuration with num_workers explicitly set
                self.logger.warning('Cluster configuration: `executors` count %d does not match the '
                                    '`num_workers` value %d. Using generated names.', executors_cnt,
                                    num_workers)
            worker_nodes_from_conf = self.generate_node_configurations(num_workers)
        if num_workers == 0 and self.props.get_value('node_type_id') is None:
            # if there are no worker nodes and no node_type_id, then we cannot proceed
            raise RuntimeError('Failed to find worker node information from cluster properties')
        # create workers array
        worker_nodes: list = []
        for worker_node in worker_nodes_from_conf:
            worker_props = {
                'Id': worker_node['node_id'],
                'props': JSONPropertiesContainer(prop_arg=worker_node, file_load=False),
                # set the node region based on the wrapper defined region
                'region': self.region,
                'instance_type': self.props.get_value('node_type_id')
            }
            worker = DatabricksAzureNode.create_worker_node().set_fields_from_dict(worker_props)
            worker.fetch_and_set_hw_info(self.cli)
            worker_nodes.append(worker)
        driver_props = {
            'Id': driver_nodes_from_conf['node_id'],
            'props': JSONPropertiesContainer(prop_arg=driver_nodes_from_conf, file_load=False),
            # set the node region based on the wrapper defined region
            'region': self.region,
            'instance_type': self.props.get_value('driver_node_type_id')
        }
        driver_node = DatabricksAzureNode.create_master_node().set_fields_from_dict(driver_props)
        driver_node.fetch_and_set_hw_info(self.cli)
        self.nodes = {
            SparkNodeType.WORKER: worker_nodes,
            SparkNodeType.MASTER: driver_node
        }

    def _init_connection(self, cluster_id: str = None,
                         props: str = None) -> dict:
        cluster_args = super()._init_connection(cluster_id=cluster_id, props=props)
        # propagate region to the cluster
        cluster_args.setdefault('region', self.cli.get_env_var('region'))
        return cluster_args

    def get_all_spark_properties(self) -> dict:
        return self.props.get_value_silent('spark_conf')

    def _build_migrated_cluster(self, orig_cluster):
        """
        specific to the platform on how to build a cluster based on migration
        :param orig_cluster: the cpu_cluster that does not support the GPU devices.
        """
        # get the map of the instance types
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
                'Id': anode.Id,
                'region': anode.region,
                'props': anode.props,
            }
            new_node = DatabricksAzureNode.create_worker_node().set_fields_from_dict(worker_props)
            new_worker_nodes.append(new_node)
        self.nodes = {
            SparkNodeType.WORKER: new_worker_nodes,
            SparkNodeType.MASTER: orig_cluster.nodes.get(SparkNodeType.MASTER)
        }
        if bool(mc_type_map):
            # update the platform notes
            self.platform.update_ctxt_notes('nodeConversions', mc_type_map)

    def get_tmp_storage(self) -> str:
        raise NotImplementedError


@dataclass
class DBAzureSavingsEstimator(SavingsEstimator):
    """
    A class that calculates the savings based on a Databricks-Azure price provider
    """

    def _get_cost_per_cluster(self, cluster: ClusterGetAccessor):
        db_azure_cost = 0.0
        for node_type in [SparkNodeType.MASTER, SparkNodeType.WORKER]:
            instance_type = cluster.get_node_instance_type(node_type)
            nodes_cnt = cluster.get_nodes_cnt(node_type)
            cost = self.price_provider.get_instance_price(instance=instance_type)
            db_azure_cost += cost * nodes_cnt
        return db_azure_cost
