# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
"""
Config module for Qualx, controlled by environment variables.

Environment variables:
- QUALX_CACHE_DIR: cache directory for saving Profiler output.
- QUALX_DATA_DIR: data directory containing eventlogs, primarily used in dataset JSON files.
- QUALX_DIR: root directory for Qualx execution, primarily used in dataset JSON files to locate
    dataset-specific plugins.
- QUALX_LABEL: targeted label column for XGBoost model, which sets its built-in cost function.
- SPARK_RAPIDS_TOOLS_JAR: path to the cuDF plugin tools JAR file.
"""
from typing import TYPE_CHECKING, Type
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_tools.tools.qualx.qualx_config import QualxConfig

if TYPE_CHECKING:
    from spark_rapids_tools.tools.qualx.rollup import CostFunction


_config = None
_cost_function = None


def get_config(path: str = None, *, cls: Type[QualxConfig] = QualxConfig, reload: bool = False) -> QualxConfig:
    global _config, _cost_function  # pylint: disable=global-statement

    if _config is None or reload:
        # get path to resources/qualx-conf.yaml
        config_path = path if path else str(Utils.resource_path('qualx-conf.yaml'))
        _config = cls.load_from_file(config_path)
        if _config is None:
            raise ValueError(f'Failed to load Qualx configuration from: {config_path}')
        _cost_function = None
    return _config


def get_cache_dir() -> str:
    """Get cache directory to save Profiler output."""
    return get_config().cache_dir


def get_cost_function() -> 'CostFunction':
    """Get the cost function that describes the cost of an app."""
    global _cost_function  # pylint: disable=global-statement

    if _cost_function is None:
        from spark_rapids_tools.tools.qualx.rollup import CostFunction  # pylint: disable=import-outside-toplevel
        config = get_config()
        cost_function = config.cost_function
        if not isinstance(cost_function, dict):
            cost_function = {'path': cost_function}

        # Load the CPU / GPU price ratio, if supplied, otherwise leave as identity.
        price_ratio = config.cpu_cost / config.gpu_cost if config.cpu_cost else 1.0
        _cost_function = CostFunction.load(cost_function['path'], cost_function.get('args'), price_ratio)

    return _cost_function


def get_label() -> str:
    """Get targeted label column for XGBoost model: the LABEL of the cost function."""
    return get_cost_function().label
