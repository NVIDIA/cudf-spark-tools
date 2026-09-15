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

import pytest

from spark_rapids_pytools.cloud_api.sp_types import CspEnv, get_platform
from spark_rapids_pytools.common.prop_manager import JSONPropertiesContainer
from spark_rapids_tools import CspPath
from spark_rapids_tools.cloud import ClientCluster
from spark_rapids_tools.exceptions import InvalidPropertiesSchema
from .conftest import SparkRapidsToolsUT, all_cpu_cluster_props

databricks_cluster_props = [
    pytest.param(CspEnv.DATABRICKS_AWS, 'cluster/databricks/aws-cpu-00.json', id='databricks_aws'),
    pytest.param(CspEnv.DATABRICKS_AZURE, 'cluster/databricks/azure-cpu-00.json', id='databricks_azure')
]


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


class TestDatabricksClusterWorkers(SparkRapidsToolsUT):  # pylint: disable=too-few-public-methods
    """
    Class testing how the Databricks platforms build the worker nodes of a cluster from its
    `clusters get` properties when the `executors` entry is absent, which is what the API
    returns for a zero-worker (single-node) cluster and for a terminated cluster
    """
    @staticmethod
    def _load_cluster(csp_enum, prop_path, data_dir, num_workers):
        with open(f'{data_dir}/{prop_path}', encoding='utf8') as prop_file:
            props = json.load(prop_file)
        props.pop('executors', None)
        props['num_workers'] = num_workers
        platform = get_platform(csp_enum)(ctxt_args={})
        return platform.load_cluster_by_prop(JSONPropertiesContainer(prop_arg=props, file_load=False))

    @pytest.mark.parametrize('csp_enum,prop_path', databricks_cluster_props)
    def test_zero_worker_cluster_raises_no_workers_error(self, csp_enum, prop_path, get_ut_data_dir):
        # a single-node cluster has num_workers 0 and no executors entry; the platform's own
        # "no worker nodes" error is the expected answer, not a TypeError from iterating None
        with pytest.raises(RuntimeError, match='The cluster has no worker nodes'):
            self._load_cluster(csp_enum, prop_path, get_ut_data_dir, num_workers=0)

    @pytest.mark.parametrize('csp_enum,prop_path', databricks_cluster_props)
    def test_terminated_cluster_generates_workers(self, csp_enum, prop_path, get_ut_data_dir):
        # a terminated multi-node cluster has no executors entry either; its workers are generated
        cluster = self._load_cluster(csp_enum, prop_path, get_ut_data_dir, num_workers=2)
        assert cluster.get_workers_count() == 2
