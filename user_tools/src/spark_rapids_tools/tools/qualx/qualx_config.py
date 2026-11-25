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

"""Configuration module for Qualx tool."""

import os
from typing import List, Optional, Union, Literal

from pydantic import Field, model_validator

from spark_rapids_tools import CspPathT
from spark_rapids_tools.configuration.common import BaseConfig
from spark_rapids_tools.utils import AbstractPropContainer


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

    label: Literal['Duration', 'duration_sum'] = Field(
        default='Duration',
        description=(
            'Targeted label column for XGBoost model. '
            'Can be overridden by QUALX_LABEL environment variable.'
        ),
        examples=['Duration', 'duration_sum'])

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

    calibration: Optional[dict] = Field(
        default=None,
        description=(
            'OPTIONAL: Calibration of the predicted app speedups, as a dictionary of path and args. '
            'path is a plugin with a calibrate(speedup, **args) function, such as the built-in power.py. '
            'The calibration is applied to the speedup of each app, after any rollup of its SQLs. '
            'This argument does not apply to training.'
        ),
        examples=[{
            'path': 'power.py',
            'args': {'c0': 0.51, 'c1': 1.0, 'c2': 2.52, 'direction': 'inverse', 'floor': 0.5},
        }])

    aggregation: Literal['sql', 'app'] = Field(
        default='sql',
        description=(
            'Aggregation level of the rows that the model predicts from. sql predicts each SQL and rolls the '
            'SQLs up to their app. app predicts from one row per app, and the prediction is the app speedup. '
            'qual_tool_filter applies to the rows per SQL only. This argument does not apply to training.'
        ),
        examples=['sql', 'app'])

    feature_levels: Optional[List[Literal['sql', 'app']]] = Field(
        default=None,
        description=(
            'OPTIONAL: Levels of raw features to write beside the predictions, in addition to the level of '
            'the aggregation. Each level is extracted from the profiler output separately.'
        ),
        examples=[['sql', 'app']])

    model_id: Optional[str] = Field(
        default=None,
        description=(
            'OPTIONAL: Identifier of the model and config, chosen by the caller. For instance, it may specify '
            'the platform, cost basis, and version of a model. It is copied to the modelId column of the '
            'per-app predictions.'
        ),
        examples=['emr/yarn_cost/3'])

    alignment_dir: Optional[str] = Field(
        default=None,
        description='OPTIONAL: Path to alignment directory.',
        examples=['alignment'])

    tools_config: Optional[str] = Field(
        default=None,
        description='OPTIONAL: Path to tools configuration file for Profiler and Qualification tools.',
        examples=['tools-config.yaml'])

    @model_validator(mode='after')
    def check_calibration(self):
        if self.calibration is None:
            return self
        calibration = dict(self.calibration)
        if not {'path'} <= set(calibration) <= {'path', 'args'}:
            raise ValueError(f'calibration takes a path and optional args, got: {calibration}')
        if not isinstance(calibration['path'], str) or not isinstance(calibration.get('args') or {}, dict):
            raise ValueError(f'calibration takes a path and a dictionary of args, got: {calibration}')
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
            if env_label not in ['Duration', 'duration_sum']:
                raise ValueError(
                    f"QUALX_LABEL environment variable must be either 'Duration' or 'duration_sum', got '{env_label}'"
                )
            self.label = env_label

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
