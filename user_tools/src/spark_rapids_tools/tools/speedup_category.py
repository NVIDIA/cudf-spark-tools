# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

"""Implementation class for Speedup Category logic."""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

import pandas as pd


def validate_speedup_threshold(value: Any) -> float:
    """Validate that a speedup threshold is a positive non-boolean finite number."""
    if isinstance(value, bool):
        raise ValueError(f'speedup_threshold must be positive and finite, got: {value}')
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'speedup_threshold must be positive and finite, got: {value}') from exc
    if not 0 < threshold < math.inf:
        raise ValueError(f'speedup_threshold must be positive and finite, got: {value}')
    return threshold


def with_speedup_threshold(props: dict, threshold: float) -> dict:
    """
    Return a copy of the speedup category properties, with the given speedup threshold in place of the
    configured one. Eligibility for qualification uses this threshold as the lower bound. The 'Not Applicable'
    category applies to any speedup below the threshold. The other bounds remain as configured.

    Example:
    with_speedup_threshold(props, 1.5)['strategies']['spark']['categories'] = [
        {'title': 'Not Applicable', 'lowerBound': -1000000.0, 'upperBound': 1.5},
        {'title': 'Small',          'lowerBound': 1.5,        'upperBound': 2.0},
        {'title': 'Medium',         'lowerBound': 2.0,        'upperBound': 3.0},
        {'title': 'Large',          'lowerBound': 3.0,        'upperBound': 1000000.0}
    ]
    """
    props = copy.deepcopy(props)
    speedup_col_name = props.get('speedupColumnName')
    for strategy in props.get('strategies', {}).values():
        for condition in strategy.get('eligibilityConditions', []):
            if condition.get('columnName') == speedup_col_name:
                condition['lowerBound'] = threshold
        # The categories are in ascending order, and each one starts where the one before it ends.
        categories = strategy.get('categories', [])
        for index in range(1, len(categories)):
            bound = threshold if index == 1 else max(categories[index]['lowerBound'], threshold)
            categories[index - 1]['upperBound'] = bound
            categories[index]['lowerBound'] = bound
    return props


class SpeedupStrategy:
    """
    Wrapper class for speedup strategy properties.
    """
    _categories: list
    _eligibility_conditions: list

    def __init__(self, props: dict):
        self._categories = props.get('categories', [])
        self._eligibility_conditions = props.get('eligibilityConditions', [])

    def get_categories(self) -> list:
        return self._categories

    def get_eligibility_conditions(self) -> list:
        return self._eligibility_conditions


@dataclass
class SpeedupCategory:
    """
    Encapsulates the logic to categorize the speedup values based on the range values.
    """
    props: dict = field(default=None, init=True)
    speedup_strategies: Dict[str, SpeedupStrategy] = field(default_factory=dict, init=False)

    def __post_init__(self):
        strategy_properties = self.props.get('strategies', {})
        # Create a SpeedupStrategy for each runtime type.
        for spark_runtime, properties in strategy_properties.items():  # type: str, dict
            self.speedup_strategies[spark_runtime] = SpeedupStrategy(properties)

    def _build_category_column(self, all_apps: pd.DataFrame) -> pd.DataFrame:
        """
        Build the category column based on the range values of the speedup column.
        Example:
        props['categories'] = [
            {'title': 'Not Recommended', 'lowerBound': -100000, 'upperBound': 1.3},
            {'title': 'Small',           'lowerBound': 1.3,     'upperBound': 2},
            {'title': 'Medium',          'lowerBound': 2,       'upperBound': 3},
            {'title': 'Large',           'lowerBound': 3,       'upperBound': 100000}
        ]
        1. input: row_1 = pd.Series({'speedup': 1.8})
           output: row_1 = pd.Series({'speedup': 1.8, 'speedup category': 'Small'})
           reason: Speedup Category will be 'Small' because the speedup is within the range (1.3-2).
        2. input: row_2 = pd.Series({'speedup': 3.5})
           output: row_2 = pd.Series({'speedup': 3.5, 'speedup category': 'Large'})
           reason: Speedup Category will be 'Large' because the speedup is within the range (3-100000).
        """
        category_col_name = self.props.get('categoryColumnName')
        speedup_col_name = self.props.get('speedupColumnName')
        spark_runtime_col_name = self.props.get('sparkRuntimeColumnName')

        # Calculate the category based on the speedup value
        def calculate_category(single_row: pd.Series) -> Optional[str]:
            spark_runtime = single_row.get(spark_runtime_col_name).lower()
            # Get the speedup strategy and its categories for the given runtime type.
            categories = self.speedup_strategies.get(spark_runtime).get_categories()
            col_value = single_row.get(speedup_col_name)
            for category in categories:
                if category.get('lowerBound') <= col_value < category.get('upperBound'):
                    return category.get('title')
            return None
        all_apps[category_col_name] = all_apps.apply(calculate_category, axis=1)
        return all_apps

    def _process_category(self, all_apps: pd.DataFrame) -> pd.DataFrame:
        """
        Process the speedup category column based on the eligibility criteria. If the row does not match
        the criteria, the category column will be set to the `Not Recommended` category.
        Example:
        self.props['eligibilityConditions'] = [
            {'columnName': 'criteriaCol1', 'lowerBound': 18, 'upperBound': 30},
            {'columnName': 'criteriaCol2', 'lowerBound': 70, 'upperBound': 100}
        ]
        1. input: row_1 = pd.Series({'criteriaCol1': 25, 'criteriaCol2': 85, 'speedup category': 'Large'})
           output: row_1 = pd.Series({'criteriaCol1': 25, 'criteriaCol2': 85, 'speedup category': 'Large'})
           reason: Category will remain 'Large' because the criteriaCol1 is within the range (18-30) and
            the criteriaCol2 (85) is within the range (70-100).
        2. input: row_2 = pd.Series({'criteriaCol1': 15, 'criteriaCol2': 85, 'speedup category': 'Medium'})
           output: row_2 = pd.Series({'criteriaCol1': 15, 'criteriaCol2': 85, 'speedup category': 'Not Recommended'})
           reason: Category will be set to 'Not Recommended' because the criteriaCol1 is not within the range (18-30)
        """
        category_col_name = self.props.get('categoryColumnName')
        heuristics_col_name = self.props.get('heuristicsColumnName')
        spark_runtime_col_name = self.props.get('sparkRuntimeColumnName')

        def process_row(single_row: pd.Series) -> pd.Series:
            spark_runtime = single_row.get(spark_runtime_col_name).lower()
            # Get the speedup strategy and its eligibility conditions for the given runtime type.
            eligibility_conditions = self.speedup_strategies.get(spark_runtime).get_eligibility_conditions()
            for entry in eligibility_conditions:
                col_value = single_row[entry.get('columnName')]
                # Have to convert the values to float because the input data is in string format
                lower_bound = float(entry.get('lowerBound'))
                upper_bound = float(entry.get('upperBound'))
                # If the row does not match the eligibility criteria, set the category to `Not Recommended`.
                # The reason for 'Not Recommended' will be added to the `Not Recommended Reason` column.
                if not lower_bound <= col_value <= upper_bound:
                    existing_reason = single_row.get('Not Recommended Reason', '')
                    heuristic_skipping_reason = entry.get('skippingReason')
                    if existing_reason and heuristic_skipping_reason:
                        single_row['Not Recommended Reason'] = existing_reason + f', {heuristic_skipping_reason}'
                    elif heuristic_skipping_reason:
                        single_row['Not Recommended Reason'] = heuristic_skipping_reason
                    single_row[category_col_name] = self.props.get('defaultCategory')
                # If the row was already marked to be skipped due to heuristics, set the category to `Not Recommended`
                # in case not already set. The reason to be skipped due to heuristic will already be there
                if single_row.get(heuristics_col_name) is True:
                    single_row[category_col_name] = self.props.get('defaultCategory')

            return single_row
        all_apps = all_apps.apply(process_row, axis=1)
        return all_apps

    def build_category_column(self, all_apps: pd.DataFrame) -> pd.DataFrame:
        apps_with_category = self._build_category_column(all_apps)
        processed_apps = self._process_category(apps_with_category)
        return processed_apps
