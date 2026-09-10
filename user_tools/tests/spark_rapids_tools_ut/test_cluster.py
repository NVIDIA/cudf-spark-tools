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

"""Test Identifying cluster from properties"""

import json
from types import SimpleNamespace

import pytest

from spark_rapids_tools import CspPath
from spark_rapids_tools.cloud import ClientCluster
from spark_rapids_tools.exceptions import InvalidPropertiesSchema
from spark_rapids_pytools.cloud_api.databricks_aws import DatabricksCluster, DatabricksNode
from spark_rapids_pytools.cloud_api.databricks_azure import DatabricksAzureCluster, DatabricksAzureNode
from .conftest import SparkRapidsToolsUT, all_cpu_cluster_props


class TestClusterCSP(SparkRapidsToolsUT):  # pylint: disable=too-few-public-methods
    """
    Class testing identifying the cluster type by comparing the properties to
    the defined Schema
    """
    def test_cluster_invalid_path(self, get_ut_data_dir):
        with pytest.raises(InvalidPropertiesSchema) as ex_schema:
            ClientCluster(CspPath(f'{get_ut_data_dir}/non_existing_file.json'))
        assert 'Incorrect properties files:' in ex_schema.value.message

    @pytest.mark.parametrize('csp,prop_path', all_cpu_cluster_props)
    def test_define_cluster_type_from_schema(self, csp, prop_path, get_ut_data_dir):
        client_cluster = ClientCluster(CspPath(f'{get_ut_data_dir}/{prop_path}'))
        assert client_cluster.platform_name == csp

    @pytest.mark.parametrize(
        ('cluster_type', 'node_type', 'prop_path'),
        [
            (DatabricksCluster, DatabricksNode, 'cluster/databricks/aws-cpu-00.json'),
            (DatabricksAzureCluster, DatabricksAzureNode, 'cluster/databricks/azure-cpu-00.json'),
        ],
    )
    def test_databricks_zero_worker_cluster_reports_validation_error(
            self, monkeypatch, get_ut_data_dir, cluster_type, node_type, prop_path):
        cluster_props = json.loads((get_ut_data_dir / prop_path).read_text(encoding='utf-8'))
        cluster_props.pop('executors')
        cluster_props['num_workers'] = 0

        platform = SimpleNamespace(cli=SimpleNamespace(
            get_region=lambda: 'test-region',
            get_env_var=lambda _: 'test-region',
        ))
        monkeypatch.setattr(node_type, 'fetch_and_set_hw_info', lambda *_: None)

        with pytest.raises(RuntimeError, match='The cluster has no worker nodes'):
            cluster_type(platform).set_connection(props=json.dumps(cluster_props))
