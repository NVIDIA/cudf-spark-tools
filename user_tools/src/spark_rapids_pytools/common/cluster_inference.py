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

"""This module provides functionality for cluster inference"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from logging import Logger

import pandas as pd

from spark_rapids_pytools.cloud_api.sp_types import PlatformBase, ClusterBase
from spark_rapids_pytools.common.utilities import ToolLogging
from spark_rapids_tools import CspEnv
from spark_rapids_tools.utils import AbstractPropContainer


class ClusterType(Enum):
    """
    Enum for cluster types
    """
    CPU = 'CPU'
    GPU = 'GPU'

    def __str__(self):
        return self.value


@dataclass
class ClusterInference:
    """
    Class for inferring cluster information and constructing CPU or GPU clusters.

    :param platform: The platform on which the cluster inference is performed.
    """
    platform: PlatformBase = field(default=None, init=True)
    cluster_type: ClusterType = field(default=ClusterType.CPU, init=True)
    logger: Logger = field(default=ToolLogging.get_and_setup_logger('rapids.tools.cluster_inference'), init=False)

    def _get_gpu_cluster_template_args(self, cluster_info_df: pd.Series) -> dict:
        """
        Extract GPU cluster properties from input json
        :param cluster_info_df: DataFrame containing cluster information
        """
        recommended_num_gpus = cluster_info_df.get('Recommended Num GPUs Per Node', 0)
        # The GPU device name is already platform-specific (e.g., "nvidia-l4" for Dataproc)
        # as it comes from the Scala tool's platform-specific output
        gpu_name = cluster_info_df.get('Recommended GPU Device')
        if gpu_name and recommended_num_gpus > 0:
            return {
                'GPU_NAME': f'"{gpu_name}"',
                'NUM_GPUS': int(recommended_num_gpus)
            }
        return {}

    def _log_inference_failure(self, app_id: str, reason: str) -> None:
        self.logger.info(f'For App ID: {app_id}, Unable to infer {self.cluster_type} cluster. Reason - {reason}')

    def _get_cluster_template_args(self, cluster_info_df: pd.Series) -> Optional[dict]:
        """
        Extract information about drivers and workers from input dataframe and return the template arguments
        """
        # Currently we support only single driver node for all CSPs
        num_driver_nodes = 1
        app_id = cluster_info_df.get('App ID')
        num_worker_nodes = cluster_info_df.get('Num Worker Nodes')
        # If number of worker nodes is invalid, log error and return
        if pd.isna(num_worker_nodes) or num_worker_nodes <= 0:
            self._log_inference_failure(app_id, 'Number of worker nodes cannot be determined. '
                                                'See logs for details.')
            return None

        cores_per_executor = cluster_info_df.get('Cores Per Executor')
        execs_per_node = cluster_info_df.get('Num Executors Per Node')
        total_cores_per_node = execs_per_node * cores_per_executor
        # 1. Construct template arguments based on raw cluster information and platform
        cluster_prop = {
            'NUM_DRIVER_NODES': int(num_driver_nodes),
            'NUM_WORKER_NODES': int(num_worker_nodes),
        }
        if self.platform.get_platform_name() == CspEnv.ONPREM:
            # For on-prem, if total cores per node is invalid, log error and return
            if pd.isna(total_cores_per_node) or total_cores_per_node <= 0:
                self._log_inference_failure(app_id, 'Total cores per node cannot be determined. '
                                                    'See logs for details.')
                return None
            # For on-prem, we need to include number of cores per worker node
            cluster_prop['NUM_WORKER_CORES'] = int(total_cores_per_node)
        else:
            # For CSPs, we need to include node types
            driver_node_type = cluster_info_df.get('Driver Node Type')
            # If driver instance is not set, use the default value from platform configurations
            if pd.isna(driver_node_type):
                driver_node_type = self.platform.configs.get_value('clusterInference', 'defaultCpuInstances', 'driver')
            worker_node_type = cluster_info_df.get('Worker Node Type')
            if pd.isna(worker_node_type):
                # For CSPs, if worker instance is not set and total cores per node is invalid, log error and return
                if pd.isna(total_cores_per_node) or total_cores_per_node <= 0:
                    self._log_inference_failure(app_id, 'Total cores per node cannot be determined. '
                                                        'See logs for details.')
                    return None
                # TODO - need to account for number of GPUs per executor
                worker_node_type = self.platform.get_matching_worker_node_type(total_cores_per_node)
                if worker_node_type is None:
                    self._log_inference_failure(app_id, f'No matching worker node '
                                                        f'found for num cores = {total_cores_per_node}')
                    return None
            cluster_prop.update({
                'DRIVER_NODE_TYPE': f'"{driver_node_type}"',
                'WORKER_NODE_TYPE': f'"{worker_node_type}"',
            })
        # 2. Now, if cluster type is GPU, include GPU properties
        if self.cluster_type == ClusterType.GPU:
            gpu_cluster_prop = self._get_gpu_cluster_template_args(cluster_info_df)
            cluster_prop.update(gpu_cluster_prop)
        return cluster_prop

    def infer_cluster(self, cluster_info_df: pd.DataFrame) -> Optional[ClusterBase]:
        """
        Infer CPU or GPU cluster configuration based input cluster df and return the constructed cluster object.
        """
        try:
            if len(cluster_info_df) != 1:
                self.logger.info('Cannot infer %s cluster from event logs. Only single cluster is supported.',
                                 self.cluster_type)
                return None

            # Extract cluster information from parsed logs. Above check ensures df contains single row.
            cluster_template_args = self._get_cluster_template_args(cluster_info_df.iloc[0])
            if cluster_template_args is None:
                return None
            # Construct cluster configuration using platform-specific logic
            cluster_conf = self.platform.generate_cluster_configuration(cluster_template_args)
            if cluster_conf is None:
                return None
            cluster_props_new = AbstractPropContainer(props=cluster_conf)
            return self.platform.load_cluster_by_prop(cluster_props_new, is_inferred=True)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('Error while inferring cluster: %s', str(e))
            return None
