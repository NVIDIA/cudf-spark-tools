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

"""Utility functions for preprocessing for QualX"""

from itertools import chain
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple
import json
import glob
import os
import re

import pandas as pd

from spark_rapids_tools.api_v1 import ProfWrapper
from spark_rapids_tools.tools.qualx.config import get_config
from spark_rapids_tools.tools.qualx.util import (
    ensure_directory,
    find_eventlogs,
    find_paths,
    get_abs_path,
    get_logger,
    get_dataset_platforms,
    load_plugin,
    run_profiler_tool,
    RegexPattern,
)

PREPROCESSED_FILE = 'preprocessed.parquet'
FEATURE_LEVELS = ('sql', 'app')  # one row of raw features per-SQL or per-app

logger = get_logger(__name__)

_featurizers = None  # global cache of featurizers
_featurizers_config = None  # the config that the featurizers were loaded for
_modifiers = None  # global cache of modifiers
_modifiers_config = None  # the config that the modifiers were loaded for


def get_alignment() -> pd.DataFrame:
    """Get alignment_df from alignment_file, including historical alignments."""
    alignment_dir = get_config().alignment_dir

    if alignment_dir is None:
        return pd.DataFrame()

    abs_path = get_abs_path(alignment_dir)
    if not abs_path:
        raise ValueError(f'Alignment directory not found: {alignment_dir}')

    # list all CSV files in alignment_file directory
    csv_files = sorted(glob.glob(f'{abs_path}/*.csv'))
    # move basename.csv (latest) file to end of list
    if len(csv_files) > 1 and not re.search(r'.*_[0-9]+.csv$', csv_files[0]):
        base_file = csv_files.pop(0)
        csv_files.append(base_file)

    # load and concatenate CSV files
    chunk_size = 100
    alignment_dfs = []
    for i in range(0, len(csv_files), chunk_size):
        chunk_files = csv_files[i:i + chunk_size]
        chunk_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
        alignment_dfs.append(chunk_df)

    if alignment_dfs:
        alignment_df = pd.concat(alignment_dfs, ignore_index=True)
        # remove duplicate CPU runs, keeping only the most recent GPU run
        subset = [col for col in ['appId_cpu', 'sqlID_cpu'] if col in alignment_df.columns]
        alignment_df.drop_duplicates(subset=subset, inplace=True, keep='last')
    else:
        # create empty alignment_df with minimal set of columns
        alignment_df = pd.DataFrame(columns=['appId_cpu', 'appId_gpu'])

    return alignment_df


def get_featurizers(reload: bool = False) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    """Lazily load featurizer modules, using global cache. The cache is rebuilt when the config is reloaded."""
    global _featurizers, _featurizers_config  # pylint: disable=global-statement
    config = get_config()
    if _featurizers is not None and _featurizers_config is config and not reload:
        return _featurizers

    # get absolute paths for featurizers
    featurizer_paths = config.featurizers
    abs_paths = [get_abs_path(f, 'featurizers') for f in featurizer_paths]

    # load featurizers
    _featurizers = [load_plugin(f) for f in abs_paths]
    _featurizers_config = config
    return _featurizers


def get_modifiers(reload: bool = False) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    """Lazily load modifier modules, using global cache. The cache is rebuilt when the config is reloaded."""
    global _modifiers, _modifiers_config  # pylint: disable=global-statement
    config = get_config()
    if _modifiers is not None and _modifiers_config is config and not reload:
        return _modifiers

    modifier_paths = config.modifiers
    if not modifier_paths:
        return []

    abs_paths = [get_abs_path(f, 'modifiers') for f in modifier_paths]

    # load modifiers
    _modifiers = [load_plugin(f) for f in abs_paths]
    _modifiers_config = config
    return _modifiers


def _supports_level(featurizer, level: str = 'app') -> bool:
    """Whether a featurizer declares support for the given aggregation level (SQL-only by default)."""
    return level in getattr(featurizer, 'supported_levels', ('sql',))


def expected_raw_features() -> Set[str]:
    """Get set of expected raw features from all featurizers."""
    featurizers = get_featurizers()
    return set(chain(*[f.expected_raw_features for f in featurizers]))


def load_datasets(
    dataset: str, ignore_test=True
) -> Tuple[Mapping[str, Any], pd.DataFrame]:
    """Load datasets as JSON and return a pd.DataFrame from the profiler CSV files.

    Parameters
    ----------
    dataset:
        Path to a folder containing one or more dataset JSON files.
    ignore_test:
        If True, ignore datasets with 'test' in the name.
    """
    config = get_config()
    cache_dir = config.cache_dir

    platforms, dataset_base = get_dataset_platforms(dataset)

    if ignore_test:

        def filter_fn(f):
            return f.endswith('.json') and 'test' not in f

    else:

        def filter_fn(f):
            return f.endswith('.json')

    ds_count = 0
    all_datasets = {}
    profile_dfs = []
    for platform in platforms:
        # load json files
        datasets = {}
        json_files = find_paths(f'{dataset_base}/{platform}', filter_fn)
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                ds = json.load(file)
                ds_count += len(ds)
                for ds_name in ds.keys():
                    # inject platform into dataset metadata
                    ds[ds_name]['platform'] = platform
                datasets.update(ds)
        all_datasets.update(datasets)

        # check cache
        platform_cache = f'{cache_dir}/{platform}'
        ensure_directory(platform_cache)
        if os.path.isfile(f'{platform_cache}/{PREPROCESSED_FILE}'):
            # load preprocessed input, if cached
            logger.info(
                'Using cached profile_df: %s/%s', platform_cache, PREPROCESSED_FILE
            )
            profile_df = pd.read_parquet(f'{platform_cache}/{PREPROCESSED_FILE}')
            if ignore_test:
                # remove any 'test' datasets from cached data by filtering
                # only appNames found in datasets structure
                dataset_keys = list(datasets.keys())
                profile_df['appName_base'] = profile_df['appName'].str.split(':').str[0]
                profile_df = profile_df.loc[
                    profile_df['appName_base'].isin(dataset_keys)
                ]
                profile_df.drop(columns='appName_base', inplace=True)
        else:
            # otherwise, check for cached profiler output
            profile_dir = f'{platform_cache}/profile'
            ensure_directory(profile_dir)
            profiles = os.listdir(profile_dir)

            # run the profiler on eventlogs, if not already cached
            for ds_name, ds_meta in datasets.items():
                if ds_name not in profiles:
                    eventlogs = ds_meta['eventlogs']
                    eventlogs = [os.path.expandvars(eventlog) for eventlog in eventlogs]
                    run_profiler_tool(platform,
                                      eventlogs,
                                      output_dir=f'{profile_dir}/{ds_name}',
                                      tools_config=config.tools_config)

            # load/preprocess profiler data
            profile_df = load_profiles(datasets, profile_dir=profile_dir)
            # save preprocessed dataframe to cache
            profile_df.to_parquet(f'{platform_cache}/{PREPROCESSED_FILE}')

        profile_dfs.append(profile_df)

    profile_df = pd.concat(profile_dfs)

    # sanity check
    if ds_count != len(all_datasets):
        logger.warning(
            'Duplicate dataset key detected, got %d datasets, but read %d datasets.',
            len(all_datasets),
            ds_count,
        )

    return all_datasets, profile_df


def infer_app_meta(eventlogs: List[str]) -> Mapping[str, Mapping]:
    """Given a list of paths to eventlogs, infer the app_meta from the path for each appId.

    Expected eventlog path format: <eventlogs>/**/<jobName>/eventlogs/<runType>/<appId>
    """
    eventlog_list = [find_eventlogs(os.path.expandvars(e)) for e in eventlogs]
    eventlog_list = list(chain(*eventlog_list))
    app_meta = {}
    for e in eventlog_list:
        parts = Path(e).parts
        app_id_part = parts[-1]
        match = RegexPattern.app_id.search(app_id_part)
        app_id = match.group() if match else app_id_part
        run_type = parts[-2].upper()
        job_name = parts[-4]
        app_meta[app_id] = {
            'jobName': job_name,
            'runType': run_type,
            'scaleFactor': 1,
        }
    return app_meta


def load_profiles(
    datasets: Mapping[str, Mapping],
    *,
    profile_dir: Optional[str] = None,
    node_level_supp: Optional[pd.DataFrame] = None,
    qual_tool_filter: Optional[str] = None,
    qual_tool_output: Optional[pd.DataFrame] = None,
    remove_failed_sql: bool = True,
    level: str = 'sql',
) -> pd.DataFrame:
    """
    Load dataset profiler CSV files as a pd.DataFrame, with one row per SQL or, for level 'app', one per app.

    qual_tool_filter applies to the rows of level 'sql' only. The rows of level 'app' always cover the whole app:
    they are currently extracted as with the filter 'none'.
    """
    if level not in FEATURE_LEVELS:
        raise ValueError(f'level must be one of {FEATURE_LEVELS}, got: {level}')

    plugins = {}
    modifiers = {}
    all_raw_features = []
    config = get_config()
    alignment_df = get_alignment()
    featurizers = get_featurizers()
    modifiers = get_modifiers()

    if level == 'app':
        # App-level rows cover the whole app and currently do not support filtering.
        # TODO: in the future, app-level rows can be aggregated from filtered SQL-level rows.
        qual_tool_filter = 'none'
        # Skip featurizers that do not produce app-level rows.
        skipped = [f.__name__ for f in featurizers if not _supports_level(f)]
        if skipped:
            logger.debug('Skipping featurizers without app-level features: %s', skipped)
        featurizers = [f for f in featurizers if _supports_level(f)]
        if not featurizers:
            raise ValueError('None of the featurizers can extract app-level features')

    feature_keys = ['appId', 'sqlID'] if level == 'sql' else ['appId']

    for ds_name, ds_meta in datasets.items():
        # get platform from dataset metadata, or use onprem if not provided
        platform = ds_meta.get('platform', 'onprem')

        if 'load_profiles_hook' in ds_meta:
            plugins[ds_name] = ds_meta['load_profiles_hook']

        # TODO: There is a difference in the way qualification tool and qualx train/predict consume profiling output.
        # We should clean this up in the future.
        if 'profiles' in ds_meta:
            # during prediction, we provide a list of profile paths from qual tool output
            profile_paths = ds_meta['profiles']
            app_meta = ds_meta['app_meta']
        elif profile_dir is not None:
            # during training/evaluation, we expect profile_dir to point to the qualx_cache
            profile_paths = ProfWrapper.find_report_paths(f'{profile_dir}/{ds_name}')
            # get app_meta, or infer from directory structure of eventlogs
            app_meta = ds_meta.get('app_meta', infer_app_meta(ds_meta['eventlogs']))
        else:
            logger.error('No profile_dir specified.')
            continue

        # get default app_meta
        if 'default' in app_meta:
            app_meta_default = app_meta['default']
            if len(app_meta) != 1:
                raise ValueError(f'Default app_meta for {ds_name} cannot be used with additional entries.')
        else:
            app_meta_default = {'runType': 'CPU', 'scaleFactor': 1}

        # get list of csv files and json files for all profiles, filtering out profiling_status.csv
        profile_files = []
        for path in profile_paths:
            csv_files = glob.glob(f'{path}/**/*.csv', recursive=True)
            csv_files = [f for f in csv_files if 'profiling_status.csv' not in f]
            json_files = glob.glob(f'{path}/**/*.json', recursive=True)
            profile_files.extend(csv_files + json_files)

        # filter files by app_ids in app_meta
        toc_list = []
        app_meta_list = []
        for app_id, meta in app_meta.items():
            # get jobName from app_meta
            job_name = meta.get('jobName', None)

            # filter profiler files by app_id and attach ds_name, appId, table_name
            # convert glob pattern to regex pattern
            if app_id == 'default':
                app_id_files = profile_files
            else:
                app_id = app_id.replace('*', '.*')
                app_id_files = [f for f in profile_files if re.search(app_id, f)]

            if app_id_files:
                tmp = pd.DataFrame({'filepath': app_id_files})
                fp_split = tmp['filepath'].str.split(r'/')
                tmp['ds_name'] = f'{ds_name}:{job_name}' if job_name else ds_name
                tmp['appId'] = fp_split.str[-2]
                tmp['table_name'] = fp_split.str[-1].str.split('.').str[0]

                # collect mapping of appId (after globbing) to meta
                tmp_app_meta = tmp[['appId']].drop_duplicates()
                for key, value in meta.items():
                    tmp_app_meta[key] = value
                app_meta_list.append(tmp_app_meta)

                toc_list.append(tmp)
            else:
                logger.warning('No CSV/JSON files found for: %s: %s', ds_name, app_id)

        if toc_list:
            toc = pd.concat(toc_list)
        else:
            logger.warning('No CSV/JSON files found for: %s', ds_name)
            continue

        # invoke featurizers to extract raw features from profiler output
        raw_features = pd.DataFrame()
        for i, featurizer in enumerate(featurizers):
            # a featurizer takes the level only if it has app-level features
            level_args = {'level': level} if level != 'sql' else {}
            features = featurizer.extract_raw_features(
                toc,
                node_level_supp,
                qual_tool_filter,
                qual_tool_output,
                remove_failed_sql,
                **level_args,
            )
            if features.empty:
                logger.warning(
                    'Featurizer (%s) returned an empty DataFrame for: %s',
                    featurizer.__name__,
                    ds_name,
                )
                if i == 0:
                    # main/first featurizer returned empty, skip other featurizers
                    break
                # non-main featurizer returned empty, continue to next featurizer
                continue

            # append features from each featurizer
            if raw_features.empty:
                # first featurizer, use as is
                raw_features = features
            else:
                # otherwise merge on appId and sqlID, or on appId alone for app-level features
                raw_features = raw_features.merge(
                    features, on=feature_keys, how='left'
                )

        # skip if no raw features
        if raw_features.empty:
            logger.warning('No raw features found for: %s', ds_name)
            continue

        # append columns from ds_meta
        raw_features[f'platform_{platform}'] = 1

        # append columns from app_meta
        if app_meta_list:
            app_meta_df = pd.concat(app_meta_list)
            raw_features = raw_features.merge(app_meta_df, on='appId', how='left')
        raw_features.fillna(app_meta_default, inplace=True)  # fill nans with default values

        # impute missing features
        raw_features = impute(raw_features, expected_raw_features())

        # append to list of all raw features
        all_raw_features.append(raw_features)

    profile_df = (
        pd.concat(all_raw_features).reset_index(drop=True)
        if all_raw_features
        else pd.DataFrame()
    )

    if not profile_df.empty:
        # run any dataset-specificload_profiles_hook plugins on profile_df
        for ds_name, plugin_path in plugins.items():
            plugin = load_plugin(plugin_path)
            if plugin:
                df_schema = profile_df.dtypes
                dataset_df = profile_df.loc[
                    (profile_df.appName == ds_name)
                    | (profile_df.appName.str.startswith(f'{ds_name}:'))
                ]
                modified_dataset_df = plugin.load_profiles_hook(dataset_df)
                if modified_dataset_df.index.equals(dataset_df.index):
                    profile_df.update(modified_dataset_df)
                    profile_df.astype(df_schema)
                else:
                    raise ValueError(
                        f'Plugin: load_profiles_hook for {ds_name} unexpectedly modified row indices.'
                    )

        # run any modifiers on profile_df
        for modifier in modifiers:
            df_schema = profile_df.dtypes
            modified_df = modifier.modify(profile_df, config=config, alignment_df=alignment_df)
            if modified_df.index.equals(profile_df.index):
                profile_df.update(modified_df)
                profile_df.astype(df_schema)
            else:
                raise ValueError(
                    f'Modifier: {modifier.__name__} unexpectedly modified row indices.'
                )

    return profile_df


def impute(full_tbl: pd.DataFrame, expected_features: set) -> pd.DataFrame:
    """Impute missing columns and delete extra columns."""
    actual_features = set(full_tbl.columns)
    if actual_features == expected_features:
        logger.debug('Dataset has all expected features')
    else:
        missing = sorted(expected_features - actual_features)
        extra = sorted(actual_features - expected_features)
        if missing:
            logger.debug('Imputing missing features: %s', missing)
            if 'fraction_supported' in missing:
                full_tbl['fraction_supported'] = 1.0
                missing.remove('fraction_supported')
            full_tbl.loc[:, missing] = 0

        if extra:
            logger.debug('Removing extra features: %s', extra)
            full_tbl = full_tbl.drop(columns=extra)

        # one last check after modifications (update expected_features if needed)
        if set(full_tbl.columns) != expected_features:
            raise ValueError('Dataset does not have expected features')

    return full_tbl


def load_qtool_execs(exec_info: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Load supported stage info from combined exec DataFrame in a form that can be merged with profiler data
    to aggregate features and durations only over supported stages.
    """
    node_level_supp = None

    def _is_ignore_no_perf(action: str) -> bool:
        return action == 'IgnoreNoPerf'

    if exec_info is not None and not exec_info.empty:
        node_level_supp = exec_info.copy()
        # TODO: Revisit the need to check for 'WholeStageCodegen' in Exec Name.
        #       We used to consider execs like 'WholeStageCodegen' )
        node_level_supp['Exec Is Supported'] = (
            node_level_supp['Exec Is Supported']
            | node_level_supp['Action'].apply(_is_ignore_no_perf)
            | node_level_supp['Exec Name']
            .astype(str)
            .apply(lambda x: x.startswith('WholeStageCodegen'))
        )
        # TODO: the composite key should be unique. We can consider simplifying this expression
        #       if there is no chance of having same duplicate App-IDs from different tools' reports.
        #       In that case, the expression can be simplified to:
        #       node_level_supp[['App ID', 'SQL ID', 'SQL Node Id', 'Exec Is Supported']]
        node_level_supp = (
            node_level_supp[['App ID', 'SQL ID', 'SQL Node Id', 'Exec Is Supported']]
            .groupby(['App ID', 'SQL ID', 'SQL Node Id'])
            .agg('all')
            .reset_index(level=[0, 1, 2])
        )
    return node_level_supp
