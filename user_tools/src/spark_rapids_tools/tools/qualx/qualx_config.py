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

"""Configuration module for Qualx tool."""

import os
from typing import List, Optional, Union, Literal

from pydantic import Field, model_validator

from spark_rapids_tools import CspPathT
from spark_rapids_tools.configuration.common import BaseConfig
from spark_rapids_tools.utils import AbstractPropContainer


# These labels are aliases for built-in cost functions.
LABEL_COST_FUNCTIONS = {
    'Duration': 'duration.py',
    'duration_sum': 'duration_sum.py',
}


class QualxConfig(BaseConfig):
    """Main container for Qualx configuration."""
    # path to the config file, for internal use only
    file_path: Optional[str] = None

    cache_dir: str = Field(
        default='qualx_cache',
        description=(
            'Cache directory for saving Profiler output. '
            'Can be overridden by QUALX_CACHE_DIR environment variable.'
        ),
        examples=['qualx_cache'],
    )

    datasets: str = Field(
        description='Directory containing dataset definitions.',
        examples=['datasets'])

    featurizers: List[str] = Field(
        description='List of featurizer modules to use.',
        examples=[['default.py', 'hash_plan.py', 'velox.py', 'photon.py']])

    modifiers: Optional[List[str]] = Field(
        default=[],
        description='List of modifier modules to use.',
        examples=[['align_hash.py']])

    label: Optional[Literal['Duration', 'duration_sum']] = Field(
        default='Duration',
        description=(
            'Targeted label column for XGBoost model, as a shorthand for its built-in cost function: '
            'Duration for duration.py, and duration_sum for duration_sum.py. '
            'See cost_function to provide a custom cost function with optional args. '
            'Can be overridden by QUALX_LABEL environment variable.'
        ),
        examples=['Duration', 'duration_sum'])

    cost_function: Optional[Union[dict, str]] = Field(
        default=None,
        description=(
            'OPTIONAL: Path to the cost function that describes app costs, or dictionary of path and args. '
            'This describes the cost of each SQL, the app-level cost, and how SQLs are rolled up to an '
            'app-level speedup. By default, the model is expected to predict the cost of each SQL as '
            'specified by the LABEL. The cost function can optionally override how model predictions '
            'are translated into GPU SQL and non-SQL costs for custom post-processing.'
        ),
        examples=[
            'my_cost_function.py',
            {
                'path': 'my_cost_function.py',
                'args': {'non_sql_speedup': 1.0},
            },
        ])

    cpu_cost: Optional[float] = Field(
        default=None,
        description=(
            'OPTIONAL: Price of one unit of the cost function\'s cost in a CPU run. This must be set with '
            'gpu_cost. The app\'s predicted speedup is multiplied by cpu_cost / gpu_cost. '
            'This is not used in training. '
        ),
        examples=[14.60])

    gpu_cost: Optional[float] = Field(
        default=None,
        description=(
            'OPTIONAL: Price of one unit of the cost function\'s cost in a GPU run. This must be set with '
            'cpu_cost. The app\'s predicted speedup is multiplied by cpu_cost / gpu_cost. '
            'This is not used in training. '
        ),
        examples=[15.96])

    split_functions: Union[dict, str] = Field(
        description='Path to split function, or dictionary of path and args',
        examples=[{
            'train': {
                'path': 'split_stratified.py',
                'args': {'threshold': 1.0, 'test_pct': 0.2, 'val_pct': 0.2}
            },
            'test': 'split_all_test.py',
        }])

    model_type: str = Field(
        description='Type of model to use.',
        examples=['xgboost'])

    xgboost: dict = Field(
        description='XGBoost-specific configuration.',
        examples=[{
            'model_name': 'xgb_model.json',
            'n_trials': 200,
            'qual_tool_filter': 'stage'
        }])

    sample_weight: Optional[dict] = Field(
        default={},
        description='OPTIONAL: Sample weight configuration.',
        examples=[{
            'threshold': 1.0,
            'positive': 1.0,
            'negative': 1.0
        }])

    calib: Optional[bool] = Field(
        default=False,
        description='Flag to perform model calibration',
        examples=[True, False])

    alignment_dir: Optional[str] = Field(
        default=None,
        description='OPTIONAL: Path to alignment directory.',
        examples=['alignment'])

    tools_config: Optional[str] = Field(
        default=None,
        description='OPTIONAL: Path to tools configuration file for Profiler and Qualification tools.',
        examples=['tools-config.yaml'])

    @model_validator(mode='after')
    def check_cost_function(self):
        """Resolve the cost function from the label, unless it is set by the config."""
        if self.cost_function is None:
            self.label = self.label or 'Duration'
            self.cost_function = LABEL_COST_FUNCTIONS[self.label]
            return self

        if 'label' in self.model_fields_set:
            raise ValueError('Set label or cost_function, not both')

        # the label comes from the cost function, through get_label
        self.label = None
        if isinstance(self.cost_function, dict) and not {'path'} <= set(self.cost_function) <= {'path', 'args'}:
            raise ValueError(f'Invalid cost_function: {self.cost_function}')
        return self

    @model_validator(mode='after')
    def check_prices(self):
        """The prices are set together, since only their ratio is used."""
        prices = [self.cpu_cost, self.gpu_cost]
        if prices != [None, None] and not all(price is not None and price > 0 for price in prices):
            raise ValueError(f'Set both cpu_cost and gpu_cost to positive prices, got: {prices}')
        return self

    @model_validator(mode='after')
    def check_env_overrides(self):
        """Check for environment variable overrides after model initialization."""
        # Check for QUALX_CACHE_DIR environment variable
        env_cache_dir = os.environ.get('QUALX_CACHE_DIR')
        if env_cache_dir:
            self.cache_dir = env_cache_dir

        # Check for QUALX_LABEL environment variable
        env_label = os.environ.get('QUALX_LABEL')
        if env_label:
            if env_label not in LABEL_COST_FUNCTIONS:
                raise ValueError(
                    f"QUALX_LABEL environment variable must be one of {list(LABEL_COST_FUNCTIONS)}, got '{env_label}'"
                )
            self.label = env_label
            self.cost_function = LABEL_COST_FUNCTIONS[env_label]

        return self

    @classmethod
    def load_from_file(cls, file_path: Union[str, CspPathT]) -> Optional['QualxConfig']:
        """Load the Qualx configuration from a file."""
        prop_container = AbstractPropContainer.load_from_file(file_path)
        obj = cls(**prop_container.props)
        obj.file_path = file_path
        return obj

    @classmethod
    def get_schema(cls) -> str:
        """Returns a JSON schema of the Qualx configuration."""
        return cls.model_json_schema()


class QualxPipelineConfig(QualxConfig):
    """Configuration for Qualx training and evaluation pipeline."""
    dataset_name: str = Field(
        description='Name of the dataset to use.',
        examples=['mydataset'])

    platform: str = Field(
        description='Platform supported by Profiler and Qualification tools.',
        examples=['onprem'])

    alignment_dir: str = Field(
        description=(
            'Path to a directory containing CSV files with CPU to GPU appId alignments '
            '(and optional sqlID alignments).'
        ),
        examples=['alignment'])

    eventlogs: dict = Field(
        description='Paths to CPU and GPU eventlogs.',
        examples=[{'cpu': ['/path/to/cpu/eventlogs'], 'gpu': ['/path/to/gpu/eventlogs']}])

    output_dir: str = Field(
        description='Path to output directory.',
        examples=['pipeline'])
