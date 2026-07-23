# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""Tests for deprecated property container replacements."""

# pylint: disable=protected-access,too-few-public-methods,wrong-import-position

import json
import logging
import warnings

import pandas as pd


DEPRECATED_PROP_CONTAINER_MESSAGE = 'Deprecated: use AbstractPropContainer instead'


with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message=DEPRECATED_PROP_CONTAINER_MESSAGE, category=DeprecationWarning)
    from spark_rapids_pytools.cloud_api import dataproc as dataproc_mod
    from spark_rapids_pytools.cloud_api.dataproc import DataprocCluster, DataprocNode
    from spark_rapids_pytools.cloud_api.sp_types import SparkNodeType
    from spark_rapids_pytools.common.cluster_inference import ClusterInference
    from spark_rapids_tools import CspEnv
    from spark_rapids_tools.utils import AbstractPropContainer


class _InferencePlatform:
    """Minimal platform implementation for cluster inference tests."""

    def __init__(self):
        self.loaded_cluster_prop = None
        self.loaded_is_inferred = None

    @staticmethod
    def get_platform_name():
        return CspEnv.DATAPROC

    @staticmethod
    def generate_cluster_configuration(render_args):
        return {
            'cluster_id': 'inferred-cluster',
            'config': {
                'workerConfig': {
                    'numInstances': render_args['NUM_WORKER_NODES'],
                    'machineType': render_args['WORKER_NODE_TYPE'].strip('"')
                },
                'masterConfig': {
                    'machineType': render_args['DRIVER_NODE_TYPE'].strip('"')
                }
            }
        }

    def load_cluster_by_prop(self, cluster_prop, is_inferred=False):
        self.loaded_cluster_prop = cluster_prop
        self.loaded_is_inferred = is_inferred
        return cluster_prop


class _DataprocCli:
    """Minimal Dataproc CLI implementation for cluster initialization tests."""

    def __init__(self):
        self.env_vars = {'zone': 'us-central1-a'}
        self.logger = logging.getLogger(__name__)

    def get_region(self):
        return 'us-central1'

    def get_zone(self):
        return self.env_vars['zone']


class _DataprocPlatform:
    """Minimal Dataproc platform implementation for cluster initialization tests."""

    def __init__(self):
        self.cli = _DataprocCli()


class _DataprocConfigs:
    """Minimal Dataproc configs implementation for pricing tests."""

    def __init__(self, pricing):
        self.pricing = pricing

    def get_value_silent(self, *keys):
        if keys == ('pricing',):
            return self.pricing
        return None


class _DataprocPricingPlatform(_DataprocPlatform):
    """Minimal Dataproc platform implementation for savings estimator tests."""

    def __init__(self, pricing):
        super().__init__()
        self.configs = _DataprocConfigs(pricing)


def _deprecated_prop_container_warnings(caught_warnings):
    return [
        warning for warning in caught_warnings
        if issubclass(warning.category, DeprecationWarning)
        and DEPRECATED_PROP_CONTAINER_MESSAGE in str(warning.message)
    ]


def test_cluster_inference_uses_abstract_prop_container():
    platform = _InferencePlatform()
    cluster_info_df = pd.DataFrame([{
        'App ID': 'app-1',
        'Num Worker Nodes': 2,
        'Cores Per Executor': 4,
        'Num Executors Per Node': 1,
        'Driver Node Type': 'n1-standard-4',
        'Worker Node Type': 'n1-standard-8'
    }])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always', DeprecationWarning)
        inferred_cluster = ClusterInference(platform=platform).infer_cluster(cluster_info_df)

    assert inferred_cluster is platform.loaded_cluster_prop
    assert platform.loaded_is_inferred is True
    assert isinstance(platform.loaded_cluster_prop, AbstractPropContainer)
    assert not _deprecated_prop_container_warnings(caught_warnings)


def test_dataproc_init_nodes_uses_abstract_prop_container(monkeypatch):
    def skip_fetching_hw_info(self, cli):
        del self, cli

    monkeypatch.setattr(DataprocNode, 'fetch_and_set_hw_info', skip_fetching_hw_info)
    cluster = DataprocCluster(_DataprocPlatform(), is_inferred=True)
    cluster.zone = 'us-central1-a'
    cluster.props = AbstractPropContainer(props={
        'config': {
            'masterConfig': {
                'instanceNames': ['master-0'],
                'machineType': 'n1-standard-4'
            },
            'workerConfig': {
                'numInstances': 1,
                'instanceNames': ['worker-0'],
                'machineType': 'n1-standard-8'
            }
        }
    })

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always', DeprecationWarning)
        cluster._init_nodes()

    master_node = cluster.nodes[SparkNodeType.MASTER]
    worker_node = cluster.nodes[SparkNodeType.WORKER][0]
    assert isinstance(master_node.props, AbstractPropContainer)
    assert isinstance(worker_node.props, AbstractPropContainer)
    assert master_node.instance_type == 'n1-standard-4'
    assert worker_node.instance_type == 'n1-standard-8'
    assert not _deprecated_prop_container_warnings(caught_warnings)


def test_dataproc_pricing_config_uses_abstract_prop_container(monkeypatch):
    pricing_props = {
        'catalog': {
            'onlineResources': [{
                'resourceKey': 'gcloud-catalog',
                'localFile': 'gcloud-catalog.json',
                'onlineURL': 'https://example.com/gcloud-catalog.json'
            }]
        }
    }

    class RecordingDataprocPriceProvider:
        def __init__(self, region, pricing_configs):
            self.region = region
            self.pricing_configs = pricing_configs

    class RecordingDataprocSavingsEstimator:
        def __init__(self, price_provider, reshaped_cluster, source_cluster,
                     target_cost=None, source_cost=None):
            self.price_provider = price_provider
            self.reshaped_cluster = reshaped_cluster
            self.source_cluster = source_cluster
            self.target_cost = target_cost
            self.source_cost = source_cost

    monkeypatch.setattr(dataproc_mod, 'DataprocPriceProvider', RecordingDataprocPriceProvider)
    monkeypatch.setattr(dataproc_mod, 'DataprocSavingsEstimator', RecordingDataprocSavingsEstimator)
    platform = _DataprocPricingPlatform(json.dumps(pricing_props))

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always', DeprecationWarning)
        estimator = dataproc_mod.DataprocPlatform.create_saving_estimator(
            platform,
            source_cluster=object(),
            reshaped_cluster=object()
        )

    pricing_config = estimator.price_provider.pricing_configs['gcloud']
    assert estimator.price_provider.region == 'us-central1'
    assert isinstance(pricing_config, AbstractPropContainer)
    assert pricing_config.props == pricing_props
    assert not _deprecated_prop_container_warnings(caught_warnings)


def test_dataproc_instance_description_uses_abstract_prop_container(monkeypatch):
    instance_descriptions = [{
        'name': 'a2-highgpu-1g',
        'guestCpus': 12,
        'memoryMb': 87296,
        'accelerators': [{
            'guestAcceleratorType': 'nvidia-tesla-a100',
            'guestAcceleratorCount': 1
        }]
    }]
    captured_props = []

    class RecordingPropContainer(AbstractPropContainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured_props.append(self.props)

    monkeypatch.setattr(dataproc_mod, 'AbstractPropContainer', RecordingPropContainer)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always', DeprecationWarning)
        processed_instances = dataproc_mod.DataprocCMDDriver._process_instance_description(
            None,
            json.dumps(instance_descriptions)
        )

    assert captured_props == [instance_descriptions]
    assert processed_instances['a2-highgpu-1g'] == {
        'VCpuCount': 12,
        'MemoryInMB': 87296,
        'GpuInfo': [{'Name': 'A100', 'Count': [1]}]
    }
    assert not _deprecated_prop_container_warnings(caught_warnings)
