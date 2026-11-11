import json

import pandas as pd
import pytest

from spark_rapids_pytools.cloud_api.dataproc import DataprocCMDDriver
from spark_rapids_pytools.common.cluster_inference import ClusterInference
from spark_rapids_tools import CspEnv


@pytest.mark.parametrize('instance_descriptions', [
    json.dumps([{'name': 'n2-standard-4', 'guestCpus': 4, 'memoryMb': 16384}]),
    [{'name': 'n2-standard-4', 'guestCpus': 4, 'memoryMb': 16384}],
])
def test_dataproc_process_instance_description_decodes_json(instance_descriptions):
    driver = object.__new__(DataprocCMDDriver)

    assert driver._process_instance_description(instance_descriptions) == {
        'n2-standard-4': {'VCpuCount': 4, 'MemoryInMB': 16384}
    }


@pytest.mark.parametrize('cluster_conf', [
    json.dumps({'NUM_DRIVER_NODES': 1, 'NUM_WORKER_NODES': 2}),
    {'NUM_DRIVER_NODES': 1, 'NUM_WORKER_NODES': 2},
])
def test_cluster_inference_decodes_rendered_json(cluster_conf):
    captured = {}
    expected_cluster = object()

    class Platform:
        def get_platform_name(self):
            return CspEnv.ONPREM

        def generate_cluster_configuration(self, cluster_template_args):
            return cluster_conf

        def load_cluster_by_prop(self, prop_container, is_inferred=False):
            captured['props'] = prop_container.props
            captured['is_inferred'] = is_inferred
            return expected_cluster

    cluster_info = pd.DataFrame([{
        'App ID': 'app-1',
        'Num Worker Nodes': 2,
        'Cores Per Executor': 4,
        'Num Executors Per Node': 2,
    }])

    assert ClusterInference(platform=Platform()).infer_cluster(cluster_info) is expected_cluster
    assert captured == {'props': {'NUM_DRIVER_NODES': 1, 'NUM_WORKER_NODES': 2}, 'is_inferred': True}
