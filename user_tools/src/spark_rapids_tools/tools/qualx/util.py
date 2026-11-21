# Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

""" Utility functions for QualX """

import glob
import importlib
import logging
import os
import re
import secrets
import string
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tabulate import tabulate

from spark_rapids_tools.api_v1 import QualWrapper, QualCore
from spark_rapids_tools.cmdli.dev_cli import DevCLI
from spark_rapids_tools.tools.qualx.config import get_config, get_cost_function, get_label
from spark_rapids_tools.utils.util import temp_file_with_contents


INTERMEDIATE_DATA_ENABLED = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class RegexPattern:
    app_id = re.compile(r'app.*[_-][0-9]+[_-][0-9]+')
    qual_tool_metrics = re.compile(r'raw_metrics')


def ensure_directory(path: str, parent: bool = False) -> None:
    """Ensure that a directory exists for a given path.

    Parameters
    ----------
    path: str
        Path to directory.
    parent: bool
        If true, ensure that the parent directory exists.
    """
    ensure_path = Path(path)
    if parent:
        ensure_path = ensure_path.parent
    os.makedirs(ensure_path, exist_ok=True)


def find_paths(directory: str, filter_fn: Callable = None, return_directories: bool = False) -> List[str]:
    """Find all files or subdirectories in a directory that match a filter function.

    Parameters
    ----------
    directory: str
        Path to directory to search.
    filter_fn: Callable
        Filter function that selects files/directories.
    return_directories: bool
        If true, returns matching directories, otherwise returns matching files
    """
    paths = []
    if directory and os.path.isdir(directory):
        for root, directories, files in os.walk(directory):
            if return_directories:
                filtered_dirs = filter(filter_fn, directories)
                paths.extend([os.path.join(root, filtered_dir) for filtered_dir in filtered_dirs])
            else:
                filtered_files = filter(filter_fn, files)
                paths.extend([os.path.join(root, file) for file in filtered_files])
    return sorted(paths)


def find_eventlogs(path: str) -> List[str]:
    """Find all eventlogs given a root directory."""
    if '*' in path:
        # handle paths w/ glob patterns
        eventlogs = [os.path.join(path, f) for f in glob.glob(path, recursive=True)]
    elif Path(path).is_dir():
        # find all 'eventlog' files in directory
        eventlogs = find_paths(path, lambda f: f == 'eventlog')
        if eventlogs:
            # and return their parent directories
            eventlogs = [Path(p).parent for p in eventlogs]
        else:
            # otherwise, find all 'app-XXXX-YYYY' files
            eventlogs = find_paths(path, RegexPattern.app_id.match(path))
            if not eventlogs:
                raise ValueError(f'No event logs found in: {path}')
    else:
        # if file, just return as list
        eventlogs = [path]

    return eventlogs


def get_abs_path(path: str, subdir: Optional[Union[str, List[str]]] = None) -> str:
    """Get absolute path for a given path, using an order of precedence and optional subdirectory.

    Order of precedence:
    1. absolute path, if provided
    2. qualx source directory
    3. config directory
    4. current working directory

    Parameters
    ----------
    path: str
        Path to get absolute path for.
    subdir: Optional[Union[str, List[str]]]
        Subdirectory (or list of subdirectories) to look for path in.

    Returns
    -------
    str
        Absolute path.
    """
    if os.path.isabs(path) and os.path.exists(path):
        # absolute path, return as-is
        return path

    # otherwise, search for in qualx source directory, config directory, and current working directory
    qualx_dir = os.path.dirname(__file__)
    config_dir = os.path.dirname(get_config().file_path)
    cwd = os.getcwd()

    search_paths = [qualx_dir, config_dir, cwd]

    # make list of subdirectories to search, including '' for no subdirectory
    if not subdir:
        subdir_paths = ['']
    elif isinstance(subdir, str):
        subdir_paths = [subdir, '']
    elif isinstance(subdir, list):
        subdir_paths = subdir
        if '' not in subdir_paths:
            subdir_paths.append('')
    else:
        raise ValueError(f'Invalid subdirectory: {subdir}')

    for subdir_path in subdir_paths:
        search_paths_with_subdir = [os.path.join(search_path, subdir_path) for search_path in search_paths]

        for search_path in search_paths_with_subdir:
            abs_path = os.path.join(search_path, path)
            if os.path.exists(abs_path):
                return abs_path

    raise ValueError(f'{path} not found')


def get_dataset_platforms(dataset: str) -> Tuple[List[str], str]:
    """Return list of platforms and dataset parent directory from a string path.

    Note: the '_' character is reserved to delimit variants of a platform, e.g. different execution engines
    like Photon or Velox.  Example: `databricks-aws_photon`.

    Platform variants may be used when separate qualx models are trained for the base platform and the variant.
    The platform variant will be used to identify the qualx model, while the base platform will be used when
    invoking the Profiler/Qualification tools.

    Parameters
    ----------
    dataset: str
        Path to datasets directory, datasets/platform directory, or datasets/platform/dataset.json file.

    Returns
    -------
    platforms: List[str]
        List of platforms associated with the dataset path.
    dataset_base: str
        Parent directory of datasets directory.
    """
    supported_platforms = [
        'databricks-aws',
        'databricks-aws_photon',
        'databricks-azure',
        'databricks-azure_photon',
        'dataproc',
        'emr',
        'onprem'
    ]

    splits = Path(dataset).parts
    basename = splits[-1]
    if basename.endswith('.json'):
        # single dataset JSON
        platforms = [splits[-2]]
        dataset_base = os.path.join(*splits[:-2])
    elif basename in supported_platforms:
        # single platform directory
        platforms = [basename]
        dataset_base = os.path.join(*splits[:-1])
    else:
        # otherwise, assume "datasets" directory
        platforms = os.listdir(dataset)
        dataset_base = dataset

    # validate platforms
    for platform in platforms:
        if platform not in supported_platforms:
            raise ValueError(f'Unsupported platform: {platform}')

    return platforms, dataset_base


def compute_accuracy(
    results: pd.DataFrame, y: str, y_preds: Dict[str, str], weight: str = None
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy scores on a pandas DataFrame.

    Parameters
    ----------
    results: pd.DataFrame
        Pandas dataframe containing the label column and one or more prediction columns.
    y: str
        Label column name.
    y_preds: Dict[str, str]
        Dictionary of display name of metric -> prediction column name.
    weight: str
        Name of column to use as weight, e.g. 'Duration' or 'appDuration'.

    Returns
    -------
    scores: Dict[str, Dict[str, float]]
        Dictionary of different scoring metrics per prediction column,
        e.g. {'QXS': {'MAPE"; 1.31, 'dMAPE': 1.25}}
    """
    scores = {}
    for name, y_pred in y_preds.items():
        scores[name] = {
            'MAPE': np.average(np.abs(results[y] - results[y_pred]) / results[y]),
            'wMAPE': np.sum(np.abs(results[y] - results[y_pred])) / np.sum(results[y]),
        }
        if weight:
            # MAPE w/ custom weighting by duration
            scores[name]['dMAPE'] = np.sum(
                results[weight] * (np.abs(results[y] - results[y_pred]) / results[y])
            ) / np.sum(results[weight])

            # MAPE w/ custom weighting by log(1 + duration)
            scores[name]['ldMAPE'] = np.sum(
                np.log(1 + results[weight])
                * (np.abs(results[y] - results[y_pred]) / results[y])
            ) / np.sum(np.log(1 + results[weight]))

            # wMAPE w/ custom weighting by duration
            scores[name]['dwMAPE'] = np.sum(
                results[weight] * np.abs(results[y] - results[y_pred])
            ) / (np.sum(results[weight] * results[y]))

            # wMAPE w/ custom weighting by log(1 + duration)
            scores[name]['ldwMAPE'] = np.sum(
                np.log(1 + results[weight]) * np.abs(results[y] - results[y_pred])
            ) / (np.sum(np.log(1 + results[weight]) * results[y]))
    return scores


def compute_precision_recall(
    results: pd.DataFrame, y: str, y_preds: Dict[str, str], threshold: float
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute precision and recall from a dataframe using a threshold for identifying true positives.

    Parameters
    ----------
    results: pd.DataFrame
        Pandas dataframe containing the label column and one or more prediction columns.
    y: str
        Label column name.
    y_preds: Dict[str, str]
        Dictionary of display name of prediction -> prediction column name.
    threshold: float
        Threshold separating positives (inclusive of threshold) from negatives.

    Returns
    -------
    precision: Dict[str, float]
        Dictionary of precision metric per prediction column,
        e.g. {'QX': 0.90, 'QXS': 0.92}
    recall: Dict[str, float]
        Dictionary of recall metric per prediction column,
        e.g. {'QX': 0.78, 'QXS': 0.82}
    """
    precision = {}
    recall = {}
    for name, y_pred in y_preds.items():
        tp = sum((results[y_pred] >= threshold) & (results[y] >= threshold))
        fp = sum((results[y_pred] >= threshold) & (results[y] < threshold))
        fn = sum((results[y_pred] < threshold) & (results[y] >= threshold))

        precision[name] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall[name] = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return precision, recall


def load_plugin(plugin_path: str) -> types.ModuleType:
    """Dynamically load plugin modules with helper functions for dataset-specific code."""
    plugin_path = os.path.expandvars(plugin_path)
    plugin_name = Path(plugin_path).name.split('.')[0]
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f'Plugin not found: {plugin_path}')

    spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.debug('Loaded plugin: %s', plugin_path)
    return module


def random_string(length: int) -> str:
    """Return a random hexadecimal string of a specified length."""
    return ''.join(secrets.choice(string.hexdigits) for _ in range(length))


def run_profiler_tool(platform: str, eventlogs: List[str], output_dir: str, tools_config: str = None) -> None:
    logger.info('Running profiling on: %s', eventlogs if len(eventlogs) < 5 else f'{len(eventlogs)} eventlogs')
    logger.info('Saving output to: %s', output_dir)

    # Write eventlogs list to a file to avoid long command lines
    ensure_directory(output_dir)
    eventlogs_text = ''.join(os.path.expandvars(e) + '\n' for e in eventlogs)
    with temp_file_with_contents(eventlogs_text, suffix='.txt') as eventlogs_file:
        logger.info('Triggering profiling with %d eventlogs via file: %s', len(eventlogs), eventlogs_file)

        dev_cli = DevCLI()
        dev_cli.profiling_core(
            eventlogs=eventlogs_file,
            platform=platform,
            output_folder=output_dir,
            tools_jar=None,
            tools_config_file=tools_config,
            verbose=True
        )


def run_qualification_tool(
        platform: str,
        eventlogs: List[str],
        output_dir: str,
        skip_run: bool = False,
        tools_config: str = None
) -> List[QualCore]:
    logger.info('Running qualification on: %s', eventlogs if len(eventlogs) < 5 else f'{len(eventlogs)} eventlogs')
    logger.info('Saving output to: %s', output_dir)

    if skip_run:
        output_dirs = QualWrapper.find_report_paths(root_path=output_dir)
    else:
        output_dirs = []
        # Filter out GPU logs early and process all qualifying eventlogs in a single call
        filtered_eventlogs = [eventlog for eventlog in eventlogs if '/gpu' not in str(eventlog).lower()]
        if filtered_eventlogs:
            # Write eventlogs list to a file to avoid long command lines
            ensure_directory(output_dir)
            filtered_text = ''.join(os.path.expandvars(e) + '\n' for e in filtered_eventlogs)
            with temp_file_with_contents(filtered_text, suffix='.txt') as eventlogs_file:
                logger.info('Triggering qualification with %d eventlogs via file: %s',
                            len(filtered_eventlogs), eventlogs_file)

                dev_cli = DevCLI()
                run_out_path = dev_cli.qualification_core(
                    eventlogs=eventlogs_file,
                    platform=platform,
                    output_folder=output_dir,
                    tools_jar=None,
                    tools_config_file=tools_config,
                    verbose=False
                )
                if run_out_path is not None:
                    output_dirs.append(run_out_path)
                else:
                    logger.warning(
                        'Qualification tool on %s did not produce any output.', eventlogs_file)

    qual_handlers: List[QualCore] = []
    for output_path in output_dirs:
        try:
            handler = QualCore(output_path)
            qual_handlers.append(handler)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning('Failed to create QualCoreHandler for %s: %s', output_path, e)

    if not qual_handlers:
        logger.warning('No valid qualification handlers were created from eventlogs')

    return qual_handlers


def print_summary(summary: pd.DataFrame) -> None:
    # print summary as formatted table
    display_cols = {
        # qualx
        'appName': 'App Name',
        'appId': 'App ID',
        'appDuration': 'App Duration',
        'Duration': 'SQL Duration',
        'Duration_supported': 'Estimated Supported\nSQL Duration',
        'Duration_pred': 'Estimated GPU\nSQL Duration',
        'appDuration_pred': 'Estimated GPU\nApp Duration',
        'fraction_supported': 'Estimated Supported\nSQL Duration Fraction',
        'speedup': 'Estimated GPU\nSpeedup',
        # actual
        'gpuDuration': 'Actual GPU\nSQL Duration',
        'appDuration_actual': 'Actual GPU\nApp Duration',
        'speedup_actual': 'Actual GPU\nSpeedup',
    }
    col_map = {k: v for k, v in display_cols.items() if k in summary.columns}
    formatted = summary[col_map.keys()].rename(col_map, axis=1)
    print(tabulate(formatted, headers='keys', tablefmt='psql', floatfmt='.2f'))


def print_speedup_summary(dataset_summary: pd.DataFrame) -> None:
    overall_speedup = (
            dataset_summary['appDuration'].sum()
            / dataset_summary['appDuration_pred'].sum()
            * get_cost_function().price_ratio
    )
    total_applications = dataset_summary.shape[0]
    summary = {
        'Total applications': total_applications,
        'Overall estimated speedup': overall_speedup,
    }
    summary_df = pd.DataFrame(summary, index=[0]).transpose()
    print('\nReport Summary:')
    print(tabulate(summary_df, colalign=('left', 'right')))
    print()


def create_row_with_default_speedup(app: pd.Series) -> pd.Series:
    """
    Create a default row for an app with no speedup prediction.
    """
    label = get_label()
    return pd.Series({
        'appName': app['App Name'],
        'appId': app['App ID'],
        'appDuration': app['App Duration'],
        label: 0,
        f'{label}_pred': 0,
        f'{label}_supported': 0,
        'fraction_supported': 0.0,
        'appDuration_pred': app['App Duration'],
        'speedup': 1.0,
    })


def write_csv_reports(per_sql: pd.DataFrame, per_app: pd.DataFrame, output_info: dict) -> None:
    """
    Write per-SQL and per-application predictions to CSV files
    """
    try:
        if per_sql is not None:
            sql_predictions_path = output_info['perSql']['path']
            logger.debug('Writing per-SQL predictions to: %s', sql_predictions_path)
            per_sql.to_csv(sql_predictions_path)
    except Exception as e:  # pylint: disable=broad-except
        logger.error('Error writing per-SQL predictions. Reason: %s', e)

    try:
        if per_app is not None:
            app_predictions_path = output_info['perApp']['path']
            logger.debug('Writing per-application predictions to: %s', app_predictions_path)
            per_app.to_csv(app_predictions_path)
    except Exception as e:  # pylint: disable=broad-except
        logger.error('Error writing per-app predictions. Reason: %s', e)


def log_fallback(logger_obj: logging.Logger, app_ids: List[str], fallback_reason: str) -> None:
    """
    Log a warning message for a fallback during preprocessing.
    This function expects logger object to log the source module of the warning.
    """
    app_id_str = ', '.join(app_ids)
    logger_obj.warning(f'Predicted speedup will be 1.0 for {app_id_str}. Reason: {fallback_reason}.')
