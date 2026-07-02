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

"""Tests for Maven URL overrides and Maven metadata authentication."""

import base64
from unittest.mock import patch

from spark_rapids_tools.utils import Utilities
from spark_rapids_pytools.resources.dev.prepackage_mgr import PrepackageMgr


def test_resolve_maven_url_keeps_default_when_env_missing(monkeypatch):
    monkeypatch.delenv('RAPIDS_TOOLS_MAVEN_BASE_URL', raising=False)
    assert Utilities.resolve_maven_url(
        'https://repo1.maven.org/maven2/com/nvidia/artifact') == (
        'https://repo1.maven.org/maven2/com/nvidia/artifact')


def test_resolve_maven_url_uses_env_for_maven_central(monkeypatch):
    monkeypatch.setenv('RAPIDS_TOOLS_MAVEN_BASE_URL', 'https://mirror.example/maven/')
    assert Utilities.resolve_maven_url(
        'https://repo1.maven.org/maven2/com/nvidia/artifact') == (
        'https://mirror.example/maven/com/nvidia/artifact')


def test_resolve_maven_url_ignores_non_maven_urls(monkeypatch):
    monkeypatch.setenv('RAPIDS_TOOLS_MAVEN_BASE_URL', 'https://mirror.example/maven')
    assert Utilities.resolve_maven_url(
        'https://archive.apache.org/dist/spark/spark.tgz') == (
        'https://archive.apache.org/dist/spark/spark.tgz')


def test_prepackage_tools_jar_url_uses_maven_base_env(monkeypatch, tmp_path):
    monkeypatch.setenv('RAPIDS_TOOLS_MAVEN_BASE_URL', 'https://mirror.example/maven/')
    monkeypatch.setattr(Utilities, 'get_latest_mvn_jar_from_metadata', lambda url_base: '26.04.0')
    mgr = PrepackageMgr(resource_dir=str(tmp_path), archive_enabled=False)

    assert mgr._get_spark_rapids_jar_url() == (  # pylint: disable=protected-access
        'https://mirror.example/maven/com/nvidia/rapids-4-spark-tools_2.12/'
        '26.04.0/rapids-4-spark-tools_2.12-26.04.0.jar')


class _FakeMetadataResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        return False

    def read(self):
        return b'''
            <metadata>
                <versioning>
                    <versions>
                        <version>26.02.0</version>
                        <version>26.04.0</version>
                    </versions>
                </versioning>
            </metadata>
        '''


def test_latest_mvn_jar_metadata_uses_configured_basic_auth(monkeypatch):
    user = 'maven-user'
    password = 'maven-password'
    maven_base_url = 'https://mirror.example/maven'
    monkeypatch.setenv('RAPIDS_TOOLS_MAVEN_BASE_URL', maven_base_url)
    monkeypatch.setenv('RAPIDS_TOOLS_MAVEN_USERNAME', user)
    monkeypatch.setenv('RAPIDS_TOOLS_MAVEN_PASSWORD', password)
    expected_auth = 'Basic ' + base64.b64encode(f'{user}:{password}'.encode()).decode()

    def fake_urlopen(request, context):  # pylint: disable=unused-argument
        assert request.full_url == f'{maven_base_url}/com/nvidia/rapids-4-spark-tools_2.12/maven-metadata.xml'
        assert request.get_header('Authorization') == expected_auth
        return _FakeMetadataResponse()

    with patch('urllib.request.urlopen', side_effect=fake_urlopen):
        assert Utilities.get_latest_mvn_jar_from_metadata(
            f'{maven_base_url}/com/nvidia/rapids-4-spark-tools_2.12',
            loaded_version='26.04.3') == '26.04.0'
