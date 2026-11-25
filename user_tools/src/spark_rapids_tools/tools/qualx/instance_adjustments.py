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

"""Instance adjustments: speedup estimates for GPU instance types, derived from a model's estimate.

A QualX model's speedup estimate describes the GPU instance type its training runs used. An instance
adjustments file gives, for each GPU instance type, how to derive that type's estimate from the
model's estimate. The instance type the model was trained on has a factor of 1.0.
"""

from typing import Dict, List, Literal, Union

from pydantic import Field, model_validator

from spark_rapids_tools import CspPathT
from spark_rapids_tools.configuration.common import BaseConfig
from spark_rapids_tools.utils import AbstractPropContainer


class LinearAdjustment(BaseConfig):
    """Multiplies the model's speedup estimate by a factor."""

    type: Literal['linear'] = Field(description='Adjustment type.')
    level: Literal['sql', 'app'] = Field(
        description='Where the adjustment applies. "sql" adjusts each SQL query\'s predicted speedup of its '
                    'supported part, after calibration and before the per-app rollup. "app" scales the per-app '
                    'speedup. Use the level the factor was measured at.')
    factor: float = Field(
        gt=0,
        description='Multiplier from the model\'s speedup estimate to this instance type\'s; 1.0 for the '
                    'instance type the model was trained on.')

    @model_validator(mode='after')
    def _no_app_level_scale_up(self) -> 'LinearAdjustment':
        if self.level == 'app' and self.factor > 1:
            raise ValueError('an app-level factor above 1 would also speed up time outside supported SQL, '
                             'which no GPU speeds up; apply it with level: sql')
        return self

    @property
    def is_identity(self) -> bool:
        """Whether the adjustment leaves the model's estimate unchanged."""
        return self.factor == 1.0

    def apply(self, speedup):
        """Adjust a speedup, or a Series of per-SQL speedups."""
        return speedup * self.factor

    def describe(self) -> str:
        return f'linear({self.factor!r}, {self.level})'


class InstanceAdjustments(BaseConfig):
    """How to derive each GPU instance type's speedup estimate from a model's estimate."""

    # A discriminated union on `type` once a second adjustment type exists.
    adjustments: Dict[str, LinearAdjustment] = Field(
        description='Adjustment from the model\'s speedup estimate, per GPU instance type.',
        examples=[{
            'g6.4xlarge': {'type': 'linear', 'level': 'sql', 'factor': 1.0},
            'g4dn.4xlarge': {'type': 'linear', 'level': 'sql', 'factor': 0.7},
        }])

    @classmethod
    def load_from_file(cls, file_path: Union[str, CspPathT]) -> 'InstanceAdjustments':
        """Load and validate an instance adjustments file (YAML or JSON)."""
        prop_container = AbstractPropContainer.load_from_file(file_path)
        return cls(**prop_container.props)

    def describe(self, instance_type: str) -> str:
        """How the estimate for instance_type is derived from the model's estimate."""
        return self.adjustments[instance_type].describe()

    def check_targets(self, instance_types: List[str]) -> None:
        """Raise ValueError unless every target instance type has an adjustment."""
        missing = ', '.join(instance_type for instance_type in instance_types if instance_type not in self.adjustments)
        if missing:
            raise ValueError(f'No adjustment for {missing}')
