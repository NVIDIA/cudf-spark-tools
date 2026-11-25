# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Test qualx_config module"""
import pytest  # pylint: disable=import-error
from spark_rapids_tools.tools.qualx.config import (
    default_config_path,
    get_config,
    get_cache_dir,
    get_label,
)
from ..conftest import SparkRapidsToolsUT

# a config with the required settings, to which a test adds the settings that it is about
BASE_CONFIG = """
datasets: datasets
featurizers: ['default.py']
split_functions: split_train_val.py
model_type: xgboost
xgboost: {}
"""


def write_config(path, settings: str = '') -> str:
    """Write a config with the given settings to the path, and return the path as get_config takes it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BASE_CONFIG + settings, encoding='utf-8')
    return str(path)


class TestConfig(SparkRapidsToolsUT):
    """Test class for qualx_config module"""
    def test_get_cache_dir(self, monkeypatch):
        # Test with mock environment variable
        monkeypatch.setenv('QUALX_CACHE_DIR', 'test_cache')
        get_config(reload=True)  # reload config
        assert get_cache_dir() == 'test_cache'

        # Test without environment variable (should use default)
        monkeypatch.delenv('QUALX_CACHE_DIR')
        get_config(reload=True)  # reload config
        assert get_cache_dir() == 'qualx_cache'

    def test_get_label(self, monkeypatch):
        # Test with duration_sum
        monkeypatch.setenv('QUALX_LABEL', 'duration_sum')
        get_config(reload=True)  # reload config
        assert get_label() == 'duration_sum'

        # Test with unsupported label
        with pytest.raises(ValueError):
            monkeypatch.setenv('QUALX_LABEL', 'duration')
            get_config(reload=True)  # reload config
            get_label()

        # Test without environment variable (should use default)
        monkeypatch.delenv('QUALX_LABEL')
        get_config(reload=True)  # reload config
        assert get_label() == 'Duration'

    def test_get_config_follows_path(self, tmp_path):
        """Test that get_config loads a config again only for another path, or on reload"""
        first_path = write_config(tmp_path / 'first' / 'qualx-conf.yaml', 'model_id: first/1\n')
        second_path = write_config(tmp_path / 'second' / 'qualx-conf.yaml', 'model_id: second/2\n')
        try:
            default = get_config(reload=True)
            assert default.file_path == default_config_path()
            assert default.model_id is None
            # the default config by its path is the loaded config
            assert get_config(default_config_path()) is default

            # a path other than the loaded one loads its config
            first = get_config(first_path)
            assert first is not default
            assert first.file_path == first_path
            assert first.model_id == 'first/1'

            # the same path, or no path, keeps the loaded config
            assert get_config(first_path) is first
            assert get_config(tmp_path / 'first' / 'qualx-conf.yaml') is first
            assert get_config() is first

            second = get_config(second_path)
            assert second.file_path == second_path
            assert second.model_id == 'second/2'
            assert get_config() is second

            # a config that fails to load leaves the loaded one in place
            invalid_path = write_config(tmp_path / 'invalid' / 'qualx-conf.yaml', 'aggregation: stage\n')
            with pytest.raises(ValueError):
                get_config(invalid_path)
            assert get_config() is second

            # reload loads the given path again, and the default config when there is no path
            reloaded = get_config(second_path, reload=True)
            assert reloaded is not second
            assert reloaded.model_id == 'second/2'
            assert get_config(reload=True).file_path == default_config_path()
        finally:
            get_config(reload=True)  # reload config

    def test_label_follows_config(self, tmp_path):
        """Test that the label is the loaded config's, without a reload"""
        task_time_path = write_config(tmp_path / 'qualx-conf.yaml', 'label: duration_sum\n')
        try:
            get_config(reload=True)
            assert get_label() == 'Duration'

            get_config(task_time_path)
            assert get_label() == 'duration_sum'
        finally:
            get_config(reload=True)  # reload config
        assert get_label() == 'Duration'
