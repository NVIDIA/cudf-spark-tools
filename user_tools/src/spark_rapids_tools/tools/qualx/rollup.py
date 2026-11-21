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

"""
Roll up the per-SQL predictions into app-level speedups, under a cost function.

A cost function is a plugin that describes the cost of an app. An app's speedup is its cost in the CPU run over
its predicted cost on GPU. Each is summed over the app's SQLs, and the time outside SQLs ('rest'):

    speedup = (sum_i K_cpu_i + K_cpu_rest) / (sum_i K_gpu_i + K_gpu_rest)

A cost function defines:

    LABEL: str
        The name of what the model predicts for each SQL: the ratio of its cost, in the CPU run over the GPU run.
        Training reads that cost from sql_cost, and writes it under this name. If LABEL is the label of a built-in
        cost function (Duration or duration_sum), the model uses that built-in sql_cost. gpu_sql_cost describes what
        the sql_cost ratio means for this cost.
    INPUT_COLUMNS: List[str]
        The columns that the cost reads. The columns must be written by the featurizers.
    sql_cost(sqls, **args) -> pd.Series
        What each SQL cost in its run. sqls is a dataframe with one row per SQL of each app, from a CPU or a GPU run.
    app_cost(sqls, **args) -> pd.Series
        What the app of each SQL cost in its run. The non-SQL cost is this app_cost minus the sum over sql_cost.
    gpu_sql_cost(sqls, speedup, **args) -> pd.Series
        Optional. What each SQL of a CPU app is predicted to cost in the GPU app. sqls has each SQL's cost as
        cost. speedup is the model's predicted ratio, adjusted for its unsupported operators. By default, the
        gpu_sql_cost is the sql_cost / speedup. This function can be overridden to add custom post-processing
        for cases where the model's predicted ratio does not directly correspond to the sql_cost basis.
    gpu_app_cost(sqls, **args) -> pd.Series
        Optional. What the app of each SQL of a CPU app is predicted to cost in the GPU app, like app_cost. sqls
        has each SQL's cost as cost, its app's cost as appCost, its predicted GPU cost as gpuCost, and its app as
        app. By default, gpu_app_cost is the sum of the app's gpu_sql_cost, plus its observed CPU non-SQL cost.
        This function can be overridden for cases where the non-SQL time for CPU and GPU are charged differently
        (e.g., due to differences in dynamic allocation).

Each function can take args from the config's cost_function args, as named parameters after its data. A parameter
without a default is required. Loading fails on a missing or an unknown arg.

Costs themselves do not include prices. If the price of a unit of cost for CPU and GPU are provided, each app's
speedup is multiplied by their ratio. Training doesn't read prices.
"""

import inspect
import os
import types
from functools import partial
from typing import Callable, List, Optional

import pandas as pd

from spark_rapids_tools.tools.qualx.qualx_config import LABEL_COST_FUNCTIONS
from spark_rapids_tools.tools.qualx.util import get_abs_path, load_plugin

# the data params passed to each plugin function, ahead of the config's args
DATA_PARAMS = {
    'sql_cost': ['sqls'],
    'app_cost': ['sqls'],
    'gpu_sql_cost': ['sqls', 'speedup'],
    'gpu_app_cost': ['sqls'],
}


def _gpu_sql_cost(sqls: pd.DataFrame, speedup: pd.Series) -> pd.Series:
    """By default, a SQL costs its CPU cost over the model's ratio."""
    return sqls['cost'] / speedup


def _gpu_app_cost(sqls: pd.DataFrame) -> pd.Series:
    """By default, an app costs its SQLs' GPU costs, and its non-SQL time costs as much as in the CPU app."""
    app = sqls.groupby('app')
    # SQLs can overlap and add up to more than the app cost. In such cases, we assume non-SQL time costs 0.
    # TODO: to properly model concurrent SQLs and non-SQL time, we would need to know how SQLs overlapped.
    return app['gpuCost'].transform('sum') + (sqls['appCost'] - app['cost'].transform('sum')).clip(lower=0)


def _load_plugin(path: str) -> types.ModuleType:
    """Load the plugin at path, which can be absolute or relative to the qualx or the config directory."""
    return load_plugin(get_abs_path(os.path.expandvars(path), ['cost_functions', 'plugins']))


def _arg_names(name: str, fn: Callable) -> List[str]:
    """The args that the plugin function fn takes after its data params."""
    return list(inspect.signature(fn).parameters)[len(DATA_PARAMS[name]):]


def _bind_args(name: str, fn: Callable, args: dict) -> Callable:
    """Bind the args that fn takes, and raise if one without a default is missing."""
    params = inspect.signature(fn).parameters
    missing = [arg for arg in _arg_names(name, fn) if params[arg].default is params[arg].empty and arg not in args]
    if missing:
        raise ValueError(f'Cost function {fn.__module__} is missing args for {name}: {missing}')
    return partial(fn, **{arg: args[arg] for arg in _arg_names(name, fn) if arg in args})


class CostFunction:
    """A cost function plugin."""

    def __init__(self, plugin: types.ModuleType, args: Optional[dict] = None, price_ratio: float = 1.0):
        name = plugin.__name__
        missing = [attr for attr in ['LABEL', 'INPUT_COLUMNS', 'sql_cost', 'app_cost'] if not hasattr(plugin, attr)]
        if missing:
            raise ValueError(f'Cost function {name} does not define: {missing}')

        args = args or {}
        functions = {
            'sql_cost': plugin.sql_cost,
            'app_cost': plugin.app_cost,
            'gpu_sql_cost': getattr(plugin, 'gpu_sql_cost', _gpu_sql_cost),
            'gpu_app_cost': getattr(plugin, 'gpu_app_cost', _gpu_app_cost),
        }
        unknown = sorted(set(args) - {arg for fn_name, fn in functions.items() for arg in _arg_names(fn_name, fn)})
        if unknown:
            raise ValueError(f'Cost function {name} takes no args named: {unknown}')

        self._sql_cost, self._app_cost, self._gpu_sql_cost, self._gpu_app_cost = (
            _bind_args(fn_name, fn, args) for fn_name, fn in functions.items())

        self.label: str = plugin.LABEL  # name of the cost that the model predicts the ratio of
        self._label_cost: Callable = self._sql_cost  # the cost function: either the plugin, or a built-in cost function
        if self.label in LABEL_COST_FUNCTIONS:
            self._label_cost = _load_plugin(LABEL_COST_FUNCTIONS[self.label]).sql_cost

        self.input_columns: List[str] = list(plugin.INPUT_COLUMNS)
        self.price_ratio = price_ratio

    @classmethod
    def load(cls, path: str, args: Optional[dict] = None, price_ratio: float = 1.0) -> 'CostFunction':
        """Load the plugin at path, which can be absolute or relative to the qualx or the config directory."""
        return cls(_load_plugin(path), args, price_ratio)

    @staticmethod
    def _check_columns(sqls: pd.DataFrame, columns: List[str]) -> None:
        """Raise if any column in columns is not present in sqls."""
        missing = [col for col in columns if col not in sqls]
        if missing:
            # e.g. features that were cached before the featurizer wrote these columns
            raise ValueError(
                f'The cost function is missing columns: {missing}. Preprocess the dataset again if it is cached.')

    @staticmethod
    def _check_costs(costs: pd.DataFrame, where: str) -> None:
        """Raise if the cost function left a cost or a predicted GPU cost missing or negative."""
        for col in ['cost', 'gpuCost']:
            invalid = costs[col].isna() | (costs[col] < 0)
            if invalid.any():
                raise ValueError(f'The cost function gave {invalid.sum()} {where} a missing or negative {col}')

    def sql_cost(self, sqls: pd.DataFrame) -> pd.Series:
        """What each SQL cost, in the CPU or GPU run that its row of sqls describes."""
        self._check_columns(sqls, self.input_columns)
        return self._sql_cost(sqls)

    def app_cost(self, sqls: pd.DataFrame) -> pd.Series:
        """What the app of each SQL cost, in the CPU or GPU run that its row of sqls describes."""
        self._check_columns(sqls, self.input_columns)
        return self._app_cost(sqls)

    def with_label(self, sqls: pd.DataFrame) -> pd.DataFrame:
        """sqls with the label: what the model predicts the ratio of, for each SQL in its CPU or GPU run."""
        return sqls.assign(**{self.label: self._label_cost(sqls)})

    def rollup(self, sqls: pd.DataFrame, by: List[str]) -> pd.DataFrame:
        """Each CPU app's cost, predicted GPU cost, supported fraction and speedup, from its per-SQL predictions.

        sqls has one row per SQL of each CPU app, with its fraction_supported and the model's y_pred. The columns
        in `by` identify an app. The final speedup includes the price ratio, and the costs do not.
        """
        self._check_columns(sqls, ['fraction_supported', 'y_pred'])
        costs = sqls.assign(app=sqls.groupby(by, sort=False).ngroup(), cost=self.sql_cost(sqls),
                            appCost=self.app_cost(sqls))
        # a SQL with no task time has no fraction, and none of it is supported
        supported = costs['fraction_supported'].fillna(0.0)
        # the model predicts the ratio of the supported part of a SQL, and the rest of it has a ratio of 1
        speedup = 1.0 / (1.0 - supported + supported / costs['y_pred'])
        costs['gpuCost'] = self._gpu_sql_cost(costs, speedup)
        self._check_costs(costs, 'SQL(s)')
        costs = costs.assign(supportedCost=costs['cost'] * supported, gpuAppCost=self._gpu_app_cost(costs))

        groups = costs.groupby(by, sort=False)
        for col, fn_name in [('appCost', 'app_cost'), ('gpuAppCost', 'gpu_app_cost')]:
            if (groups[col].nunique(dropna=False) > 1).any():
                raise ValueError(f'The cost function gave the SQLs of one app different values of {fn_name}')

        in_sqls = groups[['cost', 'supportedCost']].sum()
        app_costs = groups[['appCost', 'gpuAppCost']].first()
        apps = pd.DataFrame({
            # SQLs can overlap, so they can add up to more than their app
            'cost': in_sqls['cost'] + (app_costs['appCost'] - in_sqls['cost']).clip(lower=0),
            'gpuCost': app_costs['gpuAppCost'],
        })
        self._check_costs(apps, 'app(s)')
        return apps.assign(
            # the share of the app's cost that is in supported operators
            fraction_supported=in_sqls['supportedCost'] / apps['cost'],
            speedup=apps['cost'] / apps['gpuCost'] * self.price_ratio,
        ).reset_index()
