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

""" Main module for QualX related commands """

from typing import Callable, List, Optional, Set, Tuple, Union, Dict
import glob
import json
import os
import traceback
from pathlib import Path
from tabulate import tabulate
from xgboost.core import XGBoostError, Booster
import numpy as np
import pandas as pd
import xgboost as xgb
import fire

from spark_rapids_tools import CspPath
from spark_rapids_tools.api_v1 import (
    CombinedCSVBuilder,
    QualCore
)
from spark_rapids_tools.tools.qualx.config import (
    get_cache_dir,
    get_config,
    get_label,
)
from spark_rapids_tools.tools.qualx.preprocess import (
    load_datasets,
    load_profiles,
    load_qtool_execs,
    PREPROCESSED_FILE
)
from spark_rapids_tools.tools.qualx.model import (
    extract_model_features,
    compute_shapley_values,
)
from spark_rapids_tools.tools.qualx.model import train as train_model, calibrate as calibrate_model, predict as predict_model
from spark_rapids_tools.tools.qualx.util import (
    compute_accuracy,
    ensure_directory,
    find_paths,
    get_abs_path,
    get_logger,
    get_dataset_platforms,
    load_plugin,
    print_summary,
    print_speedup_summary,
    run_qualification_tool,
    INTERMEDIATE_DATA_ENABLED,
    create_row_with_default_speedup,
    write_csv_reports
)
from spark_rapids_pytools.common.utilities import Utils


logger = get_logger(__name__)


def _get_model_path(platform: str, model: Optional[str], variant: Optional[str] = None) -> Path:
    if model is not None:
        # if "model" is provided
        if CspPath.is_file_path(model, raise_on_error=False):
            # and "model" is actually represents a file path
            # check that it is valid json file
            if not CspPath.is_file_path(model,
                                        extensions=['json'],
                                        raise_on_error=False):
                raise ValueError(
                    f'Custom model file [{model}] is invalid. Please specify a valid JSON file.')
            # TODO: If the path is remote, we need to copy it locally in order to successfully
            #       load it with xgboost.
            model_path = Path(CspPath(model).no_scheme)
            if not model_path.exists():
                raise FileNotFoundError(f'Model JSON file not found: {model_path}')
        else:
            # otherwise, try loading pre-trained model by "short name", e.g. "onprem"
            model_path = Path(Utils.resource_path(f'qualx/models/xgboost/{model}.json'))
            if not model_path.exists():
                raise FileNotFoundError(f'Model JSON file for {model} not found at: {model_path}')
    else:
        # if "model" not provided, try loading pre-trained model by "platform" (and "variant")
        variant_suffixes = {
            'SPARK': '',
            'SPARK_RAPIDS': '',
            'PHOTON': '_photon'
        }
        variant_suffix = variant_suffixes.get(variant, '')
        model_path = Path(Utils.resource_path(f'qualx/models/xgboost/{platform}{variant_suffix}.json'))
        if not model_path.exists():
            if not variant_suffix:
                raise ValueError(
                    f'Platform [{platform}] does not have a pre-trained model, '
                    'please specify --model or choose another platform.'
                )
            # variant model not found, try base platform model
            model_path = Path(Utils.resource_path(f'qualx/models/xgboost/{platform}.json'))
            if model_path.exists():
                logger.warning(
                    'Platform [%s%s] does not have a pretrained model, using [%s] model',
                    platform,
                    variant_suffix,
                    platform,
                )
            else:
                raise ValueError(
                    f'Platform [{platform}{variant_suffix}] does not have a pre-trained model, '
                    'please specify --model or choose another platform.'
                )
    return model_path


def _get_model(platform: str,
               model: Optional[str] = None,
               variant: Optional[str] = None) -> Booster:
    """
    Load the XGBoost model from a specified path or use the pre-trained model for the platform.

    The "model" argument has a precedence over the other input options. If it is defined, this function checks
    if it is a valid path to an XGBoost model or a string literal matching a pre-trained model name.
    If "model" is not defined, then the pre-trained model matching "platform" will be used.

    Parameters
    ----------
    platform: str
        Name of the platform used to define the path of the pre-trained model.
        The platform should match a JSON file in the "resources" directory.
    model: str
        Either a path to a model file or the name of a pre-trained model.
        If the input is a string literal that cannot be a file path, then it is assumed that the file
        is located under the resources directory.
    variant: str
        Value of the `sparkRuntime` column from the `application_information.csv` file from the profiler tool.
        If set, will be used to load a pre-trained model matching the platform and variant.

    Returns
    -------
    xgb.Booster model file.
    """
    model_path = _get_model_path(platform, model, variant)
    logger.debug('Loading model from: %s', model_path)
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)
    return xgb_model


def _get_calib_params(platform: str,
                      model: Optional[str] = None,
                      variant: Optional[str] = None) -> Dict[str, float]:
    """
    Load calibration params for the model, to be used similar to `_get_model()`.
    Corresponding to model path `/path/to/{model_name}.json`, load calib params
    from `/path/to/{model_name}_calib.cfg`

    The "model" argument has a precedence over the other input options. If it is defined, this function checks
    if it is a valid path to an XGBoost model or a string literal matching a pre-trained model name.
    If "model" is not defined, then the pre-trained model matching "platform" will be used.

    Parameters
    ----------
    platform: str
        Name of the platform used to define the path of the pre-trained model.
        The platform should match a JSON file in the "resources" directory.
    model: str
        Either a path to a model file or the name of a pre-trained model.
        If the input is a string literal that cannot be a file path, then it is assumed that the file
        is located under the resources directory.
    variant: str
        Value of the `sparkRuntime` column from the `application_information.csv` file from the profiler tool.
        If set, it will be used to load a pre-trained model matching the platform and variant.

    Returns
    -------
    Calibration parameters dict
    """
    model_path = _get_model_path(platform, model, variant)
    calib_path = model_path.parent / Path(model_path.stem + '_calib.cfg')
    calib_params = {'c1': 1.0, 'c2': 1.0, 'c3': 0.0}
    if calib_path.exists():
        logger.debug('Loading calib params from: %s', calib_path)
        with open(calib_path, 'r', encoding='utf-8') as f:
            calib_params = json.load(f)
    else:
        logger.debug('Calib params file not found: %s', calib_path)
    return calib_params


def _get_combined_qual_data(
        qual_handlers: List[QualCore]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    """
    Combine qualification data from multiple QualCore objects.

    This function aggregates node-level support data and qualification tool outputs from multiple
    qualification runs, aligning them by application ID to ensure consistency across datasets.
    It processes the raw execution results using a CSV report combiner, building reports from
    each handler and merging them into unified DataFrames.

    :param qual_handlers: List of QualCore instances.
    :return: A tuple containing:
           - Combined node-level support DataFrame,
           - Combined qualification tool output DataFrame,
           - List of paths to the rawMetrics folder.
    """
    if not qual_handlers:
        return None, None, []
    # Combine node-level support data from multiple qualification handlers.
    # This processes the raw execution results using a CSV report combiner, building reports
    # from each handler and aligning on the "App ID" field to ensure consistency across datasets.
    with CombinedCSVBuilder(
            table='execCSVReport',
            handlers=qual_handlers
    ).suppress_failure() as c_builder:
        # use "App ID" to fit with the remaining qualx code.
        c_builder.combiner.on_app_fields({'app_id': 'App ID'})
        comb_node_level_supp_df = c_builder.build()
    # process the node-level-support
    processed_node_level_supp_df = load_qtool_execs(comb_node_level_supp_df)
    # Get path to the raw metrics directory which has the per-app raw_metrics files
    combined_raw_metric_paths = []
    for q_c_h in qual_handlers:
        raw_metrics_path = q_c_h.get_raw_metrics_path()
        if raw_metrics_path:
            # For now, append the path without the scheme to be compatible with the remaining code
            # that does not expect a URI value.
            combined_raw_metric_paths.append(raw_metrics_path.no_scheme)

    # TODO: TO_REMOVE: app_summary_csv was loaded to get the app_duration because it was missing in
    #      raw csv files. Consider removing that.
    with CombinedCSVBuilder(
            'qualCoreCSVSummary',
            qual_handlers,
    ).suppress_failure() as c_builder:
        # use-only those columns
        c_builder.apply_on_report(lambda x: x.pd_args({'usecols': ['App Name', 'App ID', 'App Duration']}))
        # No need to inject appIDs since "App ID" column is included in the report.
        c_builder.combiner.disable_apps_injection()
        comb_apps_summary_df = c_builder.build()

    return processed_node_level_supp_df, comb_apps_summary_df, combined_raw_metric_paths


def _get_split_fn(split_fn: Union[str, dict]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    if isinstance(split_fn, dict):
        if all(key in split_fn for key in ['path', 'args']):
            plugin_path = split_fn['path']
            plugin_kwargs = split_fn['args']
        else:
            # should never get here, otherwise fix split_fn config
            raise ValueError(f'Invalid split_function: {split_fn}')
    else:  # split_fn is a string
        plugin_path = split_fn
        plugin_kwargs = {}

    plugin = load_plugin(get_abs_path(os.path.expandvars(plugin_path), ['split_functions', 'plugins']))
    if plugin_kwargs:
        return lambda df: plugin.split_function(df, **plugin_kwargs)
    return plugin.split_function


def _compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    # summarize speedups per appId
    label = get_label()
    result_cols = [
        # qualx
        'appName',
        'appId',
        'appDuration',
        'description',
        label,
        f'{label}_pred',
        f'{label}_supported',
        'scaleFactor',
    ]
    cols = [col for col in result_cols if col in results.columns]
    # compute per-app stats
    group_by_cols = ['appName', 'appId', 'appDuration', 'scaleFactor']
    summary: pd.DataFrame = (
        results[cols]
        .groupby(group_by_cols)
        .agg(
            {
                label: 'sum',
                f'{label}_pred': 'sum',
                f'{label}_supported': 'sum',
                'description': 'first',
            }
        )
        .reset_index()
    )

    # compute the fraction of app duration w/ supported ops
    # without qual tool output, this is the entire SQL duration
    # with qual tool output, this is the fraction of SQL w/ supported ops
    summary['appDuration'] = summary['appDuration'].clip(lower=summary[label])
    summary['fraction_supported'] = (
        summary[f'{label}_supported'] / summary['appDuration']
    )

    # compute the predicted app duration from original app duration and predicted SQL duration
    # note: this assumes a non-SQL speedup of 1.0
    summary['appDuration_pred'] = (
        summary['appDuration'] - summary[label] + summary[f'{label}_pred']
    )
    # compute the per-app speedup
    summary['speedup'] = summary['appDuration'] / summary['appDuration_pred']

    # fix dtypes
    long_cols = [
        'appDuration',
        'appDuration_actual',
        'appDuration_pred',
        label,
        f'{label}_pred',
        f'{label}_supported',
        f'gpu_{label}',
    ]
    cols = [col for col in long_cols if col in summary.columns]
    summary[cols] = summary[cols].fillna(-1).astype('long')
    return summary


def _predict(
    xgb_model,
    dataset: str,
    input_df: pd.DataFrame,
    *,
    split_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
    qual_tool_filter: Optional[str] = 'stage',
    calib_params: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label = get_label()
    results = pd.DataFrame(
        columns=[
            'appId',
            'appDuration',
            'sqlID',
            'scaleFactor',
            label,
            f'{label}_supported',
            f'{label}_pred',
            'speedup_pred',
        ]
    )
    summary = pd.DataFrame(
        columns=[
            'appId',
            'appDuration',
            f'{label}_pred',
            f'{label}_supported',
            'fraction_supported',
            'appDuration_pred',
            'speedup',
        ]
    )
    if not input_df.empty:
        filter_str = (
            f'with {qual_tool_filter} filtering'
            if any(input_df['fraction_supported'] != 1.0)
            else 'raw'
        )
        logger.debug('Predicting dataset (%s): %s', filter_str, dataset)
        features, feature_cols, label_col = extract_model_features(input_df, {'default': split_fn})
        # note: dataset name is already stored in the 'appName' field
        try:
            results = predict_model(xgb_model, features, feature_cols, label_col, calib_params)

            # compute per-app speedups
            summary = _compute_summary(results)

            if 'y' in results:
                results = results.loc[~results.y.isna()].reset_index(drop=True)
        except XGBoostError as e:
            # ignore and continue
            logger.error(e)
        except Exception as e:  # pylint: disable=broad-except
            # ignore and continue
            logger.error(e)
            traceback.print_exc()
    return results, summary


def _read_dataset_scores(
    eval_dir: str, score: str, granularity: str, split: str
) -> pd.DataFrame:
    """Load accuracy scores per dataset.

    Parameters
    ----------
    eval_dir: str
        Path to the output directory of a `qualx evaluate` run.
    score: str
        Type of metric to report.
    granularity: str
        Aggregation level for metric: sql, app.
    split: str
        Name of data split to report: train, test, val, all.
        Note: for 'app' granularity, only 'all' is supported.
    """
    mape_csv = glob.glob(f'{eval_dir}/**/*_mape.csv', recursive=True)
    df = pd.concat([pd.read_csv(f) for f in mape_csv])
    df = (
        df.loc[
            (df.score == score) & (df.split == split) & (df.granularity == granularity)
        ]
        .sort_values(['model', 'platform', 'dataset'])
        .reset_index(drop=True)
    )

    nan_df = df[df.isna().any(axis=1)].copy()
    if not nan_df.empty:
        df.dropna(inplace=True)
        nan_df['key'] = (
            nan_df['model'] + '/' + nan_df['platform'] + '/' + nan_df['dataset']
        )
        keys = list(nan_df['key'].unique())
        logger.debug('Dropped rows w/ NaN values from: %s: %s', eval_dir, keys)

    return df


def _read_platform_scores(
    eval_dir: str, score: str, split: str, dataset_filter: List[str] = None
) -> Tuple[pd.DataFrame, Set[str]]:
    """Load accuracy scores per platform.

    Per-app predictions are aggregated by platform to produce the scores.

    Parameters
    ----------
    eval_dir: str
        Path to the output directory of a `qualx evaluate` run.
    score: str
        Type of metric to report
    split: str
        Name of data split to report: train, test, all.
    """
    pred_csv = glob.glob(f'{eval_dir}/**/*_app.csv', recursive=True)

    if split == 'train':
        pred_csv = [f for f in pred_csv if 'test' not in f]
    elif split == 'test':
        pred_csv = [f for f in pred_csv if 'test' in f]

    dfs = []
    for f in pred_csv:
        splits = Path(f).parts
        model = splits[1]
        platform = splits[2] if model == 'combined' else model
        dataset = splits[3] if model == 'combined' else splits[2]
        if dataset_filter and dataset not in dataset_filter:
            continue
        df = pd.read_csv(f)
        df['model'] = model
        df['platform'] = platform
        df['dataset'] = dataset
        dfs.append(df)
    preds_df = pd.concat(dfs)

    nan_df = preds_df[preds_df.isna().any(axis=1)].copy()
    if not nan_df.empty:
        preds_df.dropna(inplace=True)
        nan_df['key'] = (
            nan_df['model'] + '/' + nan_df['platform'] + '/' + nan_df['dataset']
        )
        keys = list(nan_df['key'].unique())
        logger.debug('Dropped rows w/ NaN values from: %s: %s', eval_dir, keys)

    # compute accuracy by platform
    scores = {}
    grouped = preds_df.groupby(['model', 'platform'])
    for (model, platform), group in grouped:
        acc = compute_accuracy(
            group,
            y='Actual speedup',
            y_preds={'QX': 'QX speedup', 'QXS': 'QXS speedup'},
            weight='appDuration',
        )
        acc_score = {k: v[score] for k, v in acc.items()}
        acc_score['count'] = len(group)
        scores[(model, platform)] = acc_score
    scores_df = pd.DataFrame(scores).transpose().reset_index()
    scores_df.rename(columns={'level_0': 'model', 'level_1': 'platform'}, inplace=True)
    scores_df['count'] = scores_df['count'].astype(int)
    return scores_df, set(preds_df.dataset.unique())


def _add_entries_for_missing_apps(all_default_preds: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entry with default speedup prediction for apps that are missing in the summary.
    """
    # identify apps that do not have predictions in the summary
    missing_apps = all_default_preds[~all_default_preds['appId'].isin(summary_df['appId'])].copy()
    missing_apps['wasPredicted'] = False
    summary_df['wasPredicted'] = True
    # add missing applications with default predictions to the summary
    return pd.concat([summary_df, missing_apps], ignore_index=True)


def models() -> None:
    """Show available pre-trained models."""
    available_models = [
        model.replace('.json', '')
        for model in os.listdir(
            Path(Utils.resource_path('qualx/models/xgboost'))
        )
    ]
    for model in sorted(available_models):
        print(model)


def preprocess(dataset: str, config: Optional[str] = None) -> None:
    """Extract raw features from profiler logs.

    Extract raw features from one or more profiler logs and save the resulting dataframe as a
    parquet file.  This is primarily used to cache preprocessed input data for training.

    Parameters
    ----------
    dataset: str
        Path to a datasets directory for a given platform, e.g. 'datasets/onprem'
    config_path: str
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    get_config(config)

    platforms, _ = get_dataset_platforms(dataset)

    # invalidate preprocessed.parquet files
    cache_dir = get_cache_dir()
    for platform in platforms:
        preprocessed_data = f'{cache_dir}/{platform}/{PREPROCESSED_FILE}'
        if os.path.exists(preprocessed_data):
            logger.info('Invalidating cached profile_df: %s', preprocessed_data)
            os.remove(preprocessed_data)

    load_datasets(dataset, ignore_test=False)


def train(
    dataset: str,
    model: Optional[str] = 'xgb_model.json',
    output_dir: Optional[str] = 'train',
    n_trials: Optional[int] = None,
    base_model: Optional[str] = None,
    features_csv_dir: Optional[str] = None,
    config: Optional[str] = None,
) -> None:
    """Train an XGBoost model.

    Train a model on the input data specified by either `--preprocessed` or `--dataset`.

    Parameters
    ----------
    dataset:
        Path to a folder containing one or more dataset JSON files.
    model:
        Path to save the trained XGBoost model.
    output_dir:
        Path to save the output files, e.g. SHAP values, feature importance.
    n_trials:
        Number of trials for hyperparameter search.
    base_model:
        Path to an existing pre-trained model to continue training from.
    features_csv_dir:
        Path to a directory containing one or more features.csv files.  These files are produced during prediction,
        and must be manually edited to provide a label column (Duration_speedup) and value.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    cfg = get_config(config)
    model_type = cfg.model_type
    model_config = cfg.__dict__.get(model_type, {})
    trials = n_trials if n_trials else model_config.get('n_trials', 200)
    calib = cfg.calib

    datasets, profile_df = load_datasets(dataset)
    dataset_list = sorted(list(datasets.keys()))
    profile_datasets = sorted(list(profile_df['appName'].unique()))
    logger.info('Training on: %s', dataset_list)

    # sanity check
    if set(dataset_list) != set(profile_datasets):
        logger.warning(
            'Training data contained datasets: %s, expected: %s', profile_datasets, dataset_list
        )

    # default split function for training
    split_fn = load_plugin(get_abs_path('split_train_val.py', 'split_functions')).split_function
    split_functions = {'default': split_fn}

    # per-dataset split functions, if specified
    for ds_name, ds_meta in datasets.items():
        if 'split_function' in ds_meta:
            split_fn = _get_split_fn(ds_meta['split_function'])
            split_functions[ds_name] = split_fn

    features, feature_cols, label_col = extract_model_features(profile_df, split_functions)

    if features_csv_dir:
        if not Path(features_csv_dir).exists():
            raise FileNotFoundError(f'Features directory not found: {features_csv_dir}')
        features_csv_files = glob.glob(f'{features_csv_dir}/**/*.csv', recursive=True)
        df = pd.concat([pd.read_csv(f) for f in features_csv_files])
        if df[label_col].isnull().any():
            raise ValueError(
                'Additional features contained an empty/null label, '
                f'please add a label column ({label_col}) and value.'
            )
        df['split'] = 'train'
        logger.info(
            'Concatenating %s row(s) of additional features to training set from: %s', len(df), features_csv_dir
        )
        features = pd.concat([features, df])

    xgb_base_model = None
    if base_model:
        if os.path.exists(base_model):
            logger.info('Fine-tuning on base model from: %s', base_model)
            xgb_base_model = xgb.Booster()
            xgb_base_model.load_model(base_model)
            base_model_cfg = Path(base_model).with_suffix('.cfg')
            if os.path.exists(base_model_cfg):
                with open(base_model_cfg, 'r', encoding='utf-8') as f:
                    cfg = f.read()
                xgb_base_model.load_config(cfg)
            else:
                raise ValueError(f'Existing model config not found: {base_model_cfg}')
            feature_cols = xgb_base_model.feature_names   # align features to base model
        else:
            raise ValueError(f'Existing model not found for fine-tuning: {base_model}')
    xgb_model = train_model(features, feature_cols, label_col, n_trials=trials, base_model=xgb_base_model)

    # save model and params
    ensure_directory(model, parent=True)
    logger.info('Saving model to: %s', model)
    xgb_model.save_model(model)
    cfg = xgb_model.save_config()
    base_model_cfg = Path(model).with_suffix('.cfg')
    with open(base_model_cfg, 'w', encoding='utf-8') as f:
        f.write(cfg)

    # calibrate model
    if calib:
        calib_params = calibrate_model(features, feature_cols, label_col, xgb_model)
        calib_path = Path(model).parent / Path(Path(model).stem + '_calib.cfg')
        with open(calib_path, 'w', encoding='utf-8') as f:
            json.dump(calib_params, f)

    ensure_directory(output_dir)

    for split in ['train', 'test']:
        if split == 'train':
            features_split = features[feature_cols].loc[features['split'].isin(['train', 'val'])]
        else:
            features_split = features[feature_cols].loc[features['split'] == split]
        if features_split.empty:
            continue

        feature_importance, _ = compute_shapley_values(xgb_model, features_split)
        feature_cols = feature_importance.feature.values
        feature_stats = features_split[feature_cols].describe().transpose().drop(columns='count')
        feature_stats = feature_stats.reset_index().rename(columns={'index': 'feature'})

        feature_importance = feature_importance.merge(feature_stats, how='left', on='feature')
        feature_importance.to_csv(f'{output_dir}/shap_{split}.csv')
        if split == 'train':
            # save training shap values and feature distribution metrics with model
            metrics_file = Path(model).with_suffix('.metrics')
            feature_importance.to_csv(metrics_file)

        print(f'Shapley feature importance and statistics ({split}):')
        print(tabulate(feature_importance, headers='keys', tablefmt='psql', floatfmt='.2f'))


def predict(
    platform: str,
    qual: str,
    output_info: dict,
    *,
    model: Optional[str] = None,
    qual_tool_filter: Optional[str] = None,
    config: Optional[str] = None,
    qual_handlers: List[QualCore]
) -> pd.DataFrame:
    """Predict GPU speedup given CPU logs.

    Parameters
    ----------
    platform: str
        Platform name supported by Profiler tool, e.g. 'onprem'
    qual: str
        Path to a directory containing one or more qual output directories.
    output_info: dict
        Dictionary containing information about the output files.
    model:
        Name of a pre-trained model or path to a model on disk to use for prediction.
    qual_tool_filter:
        Filter to apply to the qualification tool output: 'stage' (default), 'sqlId', or 'none'.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    qual_handlers:
        List of QualCoreResultHandler instances for reading qualification data.
    """
    # load config from command line argument, or use default
    cfg = get_config(config)
    model_type = cfg.model_type
    model_config = cfg.__dict__.get(model_type, {})
    qual_filter = qual_tool_filter if qual_tool_filter else model_config.get('qual_tool_filter', 'stage')

    if qual_handlers:
        # filter out empty handlers
        qual_handlers = [q_core_h for q_core_h in qual_handlers if not q_core_h.is_empty()]
    # log a warning and return an empty DataFrame if no qual handlers can be processed.
    if not qual_handlers:
        logger.warning('All qualification handlers are empty - no apps to predict')
        return pd.DataFrame()
    node_level_supp, qual_tool_output, qual_metrics = _get_combined_qual_data(qual_handlers)
    # create a DataFrame with default predictions for all app IDs.
    # this will be used for apps without predictions.
    default_preds_df = qual_tool_output.apply(create_row_with_default_speedup, axis=1)

    if len(qual_metrics) == 0:
        logger.warning('Qualification tool metrics are missing. Speedup predictions will be skipped.')
        return pd.DataFrame()

    # construct mapping of appIds to original appNames
    app_id_name_map = default_preds_df.set_index('appId')['appName'].to_dict()

    # if qualification metrics are provided, load metrics and apply filtering
    datasets = {}                       # create a dummy dataset
    dataset_name = Path(qual).name      # use qual directory name as dataset name
    datasets[dataset_name] = {
        'profiles': qual_metrics,
        'app_meta': {'default': {'runType': 'CPU', 'scaleFactor': 1}},
        'platform': platform,
    }

    logger.debug('Loading dataset: %s', dataset_name)
    profile_df = load_profiles(
        datasets=datasets,
        node_level_supp=node_level_supp,
        qual_tool_filter=qual_filter,
        qual_tool_output=qual_tool_output,
        remove_failed_sql=False,
    )
    if profile_df.empty:
        raise ValueError('Data preprocessing resulted in an empty dataset. Speedup predictions will default to 1.0.')

    # reset appName to original
    profile_df['appName'] = profile_df['appId'].map(app_id_name_map)

    # handle platform variants (for CPU eventlogs)
    variants = profile_df.loc[profile_df.runType == 'CPU']['sparkRuntime'].unique().tolist()
    if len(variants) > 1:
        # platform variants found in dataset
        if model and model != platform:
            # mixing platform and model args w/ variants is ambiguous
            raise ValueError(
                f'Platform variants found in data, but model ({model}) does not match platform ({platform}).'
            )
        # group input data by model variant
        prediction_groups = {}
        for variant in variants:
            xgb_model = _get_model(platform, None, variant)  # variants only supported for pre-trained platform models
            calib_params = _get_calib_params(platform, None, variant)
            variant_profile_df = profile_df.loc[
                (profile_df['sparkRuntime'] == variant) | (profile_df['sparkRuntime'] == 'SPARK_RAPIDS')
            ]
            prediction_groups[xgb_model] = (variant_profile_df, calib_params)
    else:
        # single platform
        xgb_model = _get_model(platform, model)
        calib_params = _get_calib_params(platform, model)
        prediction_groups = {
            xgb_model: (profile_df, calib_params)
        }

    filter_str = (
        f'with {qual_tool_filter} filtering'
        if node_level_supp is not None and any(profile_df['fraction_supported'] != 1.0)
        else 'raw'
    )
    logger.debug('Predicting dataset (%s): %s', filter_str, dataset_name)

    try:
        features_list = []
        predictions_list = []
        for xgb_model, (prof_df, calib_params) in prediction_groups.items():
            features, feature_cols, label_col = extract_model_features(prof_df)
            features_index = features.index
            features_list.append(features)
            predictions = predict_model(xgb_model, features, feature_cols, label_col, calib_params)
            predictions.index = features_index
            predictions_list.append(predictions)
        features = pd.concat(features_list)
        features.sort_index(inplace=True)
        predictions = pd.concat(predictions_list)
        predictions.sort_index(inplace=True)
    except XGBoostError as e:
        # ignore and continue
        logger.error(e)

    # write output files
    per_app_summary = None
    try:
        per_sql_summary = predictions

        # save features, feature importance, and shapley values per dataset name
        if output_info:
            # save features for troubleshooting
            output_file = output_info['features']['path']
            logger.debug('Writing features to: %s', output_file)
            features.to_csv(output_file, index=False)

            feature_importance, shapley_values = compute_shapley_values(xgb_model, features)

            output_file = output_info['featureImportance']['path']
            logger.debug('Writing shapley feature importances to: %s', output_file)
            feature_importance.to_csv(output_file)

            output_file = output_info['shapValues']['path']
            logger.debug('Writing shapley values to: %s', output_file)
            shapley_values.to_csv(output_file, index=False)

        # compute per-app speedups
        summary = _compute_summary(per_sql_summary)
        # combine calculated summary with default predictions for missing apps
        per_app_summary = _add_entries_for_missing_apps(default_preds_df, summary)
        if INTERMEDIATE_DATA_ENABLED:
            print_summary(per_app_summary)
            # compute speedup for the entire dataset
            dataset_speedup = (
                per_app_summary['appDuration'].sum() / per_app_summary['appDuration_pred'].sum()
            )
            print(f'Dataset estimated speedup: {dataset_speedup:.2f}')
    except Exception as e:  # pylint: disable=broad-except
        # ignore and continue
        logger.error(e)
        traceback.print_exc()

    write_csv_reports(per_sql_summary, per_app_summary, output_info)
    if INTERMEDIATE_DATA_ENABLED:
        print_speedup_summary(per_app_summary)
    return per_app_summary


def _predict_cli(
    platform: str,
    output_dir: str,
    *,
    eventlogs: Optional[str] = None,
    qual_output: Optional[str] = None,
    model: Optional[str] = None,
    qual_tool_filter: Optional[str] = None,
    config: Optional[str] = None,
) -> None:
    """Predict GPU speedup given CPU logs.

    Predict the speedup of running a Spark application with the cuDF plugin on GPUs (vs. CPUs).
    This uses an XGBoost model trained on matching CPU and GPU runs of various Spark applications.

    Note: this provides a 'best guess' based on an ML model, which may show incorrect results when
    compared against actual performance.

    Parameters
    ----------
    platform: str
        Name of platform for spark_rapids_user_tools, e.g. `onprem`, `dataproc`, etc.  This will
        be used as the model name, if --model is not provided.
    output_dir:
        Path to save predictions as CSV files.
    eventlogs: str
        Path to a single event log, or a directory containing multiple event logs.
    qual_output: str
        Path to qualification tool output.
    model: str
        Either a model name corresponding to a platform/pre-trained model, or the path to an XGBoost
        model on disk.
    qual_tool_filter: str
        Set to either 'stage' (default), 'sqlId', or 'none', where 'stage' applies model to supported stages,
        'sqlId' applies model to supported sqlIDs, and 'none' applies model to all sqlIDs and stages.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    cfg = get_config(config)
    model_type = cfg.model_type
    model_config = cfg.__dict__.get(model_type, {})
    qual_filter = qual_tool_filter if qual_tool_filter else model_config.get('qual_tool_filter', 'stage')

    # Note this function is for internal usage only. `spark_rapids predict` cmd is the public interface.
    assert eventlogs or qual_output, 'Please specify either --eventlogs or --qual_output.'

    # Construct output paths
    ensure_directory(output_dir)
    qual_handlers = []

    if eventlogs:
        # --eventlogs takes priority over --qual_output
        if not any(f.startswith('qual_') for f in os.listdir(output_dir)):
            # run qual tool if no existing qual output found
            qual_handlers.extend(run_qualification_tool(platform,
                                                        [eventlogs],
                                                        output_dir,
                                                        tools_config=cfg.tools_config))
        qual = output_dir
    else:
        qual = qual_output
        qual_handlers.append(QualCore(qual_output))

    output_info = {
        'perSql': {'path': os.path.join(output_dir, 'per_sql.csv')},
        'perApp': {'path': os.path.join(output_dir, 'per_app.csv')},
        'shapValues': {'path': os.path.join(output_dir, 'shap_values.csv')},
        'features': {'path': os.path.join(output_dir, 'features.csv')},
        'featureImportance': {'path': os.path.join(output_dir, 'feature_importance.csv')},
    }

    if not qual_handlers:
        raise ValueError('No qualification handlers available - unable to proceed with prediction')

    predict(
        platform,
        output_info=output_info,
        qual=qual,
        model=model,
        qual_tool_filter=qual_filter,
        qual_handlers=qual_handlers
    )


def evaluate(
    platform: str,
    dataset: str,
    output_dir: str,
    *,
    model: Optional[str] = None,
    qual_tool_filter: Optional[str] = None,
    config: Optional[str] = None,
) -> None:
    """Evaluate model predictions against actual GPU speedup.

    For training datasets with GPU event logs, this returns the actual GPU speedup along with the
    predictions of:
    - QX: qualx (raw)
    - QXS: qualx (stage filtered)

    Parameters
    ----------
    platform: str
        Name of platform for spark_rapids_user_tools, e.g. `onprem`, `dataproc`, etc.  This will
        be used as the model name, if --model is not provided.
    dataset: str
        Path to a JSON file describing the input dataset.
    output_dir: str
        Path to a directory where the processed files will be saved.
    model: str
        Either a model name corresponding to a platform/pre-trained model, or the path to an XGBoost
        model on disk.
    qual_tool_filter: str
        Filter out unsupported sql queries ('sql') or unsupported stages ('stage', default) before
        applying the model, or use 'off' to apply the model without any consideration of unsupported
        operators.  When filtering, a sql or stage will be considered unsupported if any of its
        contained operators is unsupported.  That sql or stage will be assigned a 1.0 speedup, and
        the model will be applied to the remaining sqls or stages.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    cfg = get_config(config)
    qual_filter = qual_tool_filter if qual_tool_filter else cfg.xgboost.get('qual_tool_filter', 'stage')

    with open(dataset, 'r', encoding='utf-8') as f:
        datasets = json.load(f)
        for ds_name in datasets.keys():
            datasets[ds_name]['platform'] = platform

    splits = os.path.split(dataset)
    dataset_name = splits[-1].replace('.json', '')

    ensure_directory(output_dir)

    xgb_model = _get_model(platform, model)
    calib_params = _get_calib_params(platform, model)

    cache_dir = get_cache_dir()

    platform_cache = f'{cache_dir}/{platform}'
    profile_dir = f'{platform_cache}/profile'
    qual_dir = f'{platform_cache}/qual'
    ensure_directory(qual_dir)

    split_fn = None
    quals = os.listdir(qual_dir)
    qual_handlers: List[QualCore] = []
    for ds_name, ds_meta in datasets.items():
        eventlogs = ds_meta['eventlogs']
        eventlogs = [os.path.expandvars(eventlog) for eventlog in eventlogs]
        skip_run = ds_name in quals
        if qual_filter in ['sql', 'stage']:
            qual_handlers.extend(run_qualification_tool(platform,
                                                        eventlogs,
                                                        f'{qual_dir}/{ds_name}',
                                                        skip_run=skip_run,
                                                        tools_config=cfg.tools_config))
        if 'split_function' in ds_meta:
            split_fn = _get_split_fn(ds_meta['split_function'])

    node_level_supp = None
    qual_tool_output = None
    if qual_filter in ['sql', 'stage']:
        logger.debug('Loading qualification tool CSV files.')
        if not qual_handlers:
            raise ValueError('No qualification handlers available for evaluation')
        node_level_supp, qual_tool_output, _ = _get_combined_qual_data(qual_handlers)

    logger.debug('Loading profiler tool CSV files.')
    profile_df = load_profiles(datasets, profile_dir=profile_dir)  # w/ GPU rows
    filtered_profile_df = load_profiles(
        datasets,
        profile_dir=profile_dir,
        node_level_supp=node_level_supp,
        qual_tool_filter=qual_filter,
        qual_tool_output=qual_tool_output,
    )  # w/o GPU rows
    if profile_df.empty:
        raise ValueError(f'Warning: No profile data found for {dataset}')

    if not split_fn:
        # use default split_function, if not specified
        if 'test' in dataset_name:
            split_fn = load_plugin(get_abs_path('split_all_test.py', 'split_functions')).split_function
        else:
            split_fn = load_plugin(get_abs_path('split_train_val.py', 'split_functions')).split_function

    # raw predictions on unfiltered data
    raw_sql, raw_app = _predict(
        xgb_model,
        dataset_name,
        profile_df,
        split_fn=split_fn,
        qual_tool_filter=qual_filter,
        calib_params=calib_params
    )

    # app level ground truth
    app_durations = (
        profile_df.loc[profile_df.runType == 'GPU'][
            ['appName', 'appDuration', 'description', 'scaleFactor']
        ]
        .groupby(['appName', 'appDuration', 'scaleFactor'])
        .first()
        .reset_index()
    )
    app_durations = app_durations.rename(columns={'appDuration': 'gpu_appDuration'})

    # join raw app data with app level gpu ground truth
    raw_app = raw_app.merge(
        app_durations[['appName', 'description', 'scaleFactor', 'gpu_appDuration']],
        on=['appName', 'description', 'scaleFactor'],
        how='left',
    )

    if not raw_app.loc[raw_app.gpu_appDuration.isna()].empty:
        logger.error(
            'missing gpu apps: %s', raw_app.loc[raw_app.gpu_appDuration.isna()].to_markdown()
        )
        raw_app = raw_app.loc[~raw_app.gpu_appDuration.isna()]

    raw_app = raw_app.rename({'gpu_appDuration': 'appDuration_actual'}, axis=1)
    raw_app['speedup_actual'] = raw_app['appDuration'] / raw_app['appDuration_actual']

    # adjusted prediction on filtered data
    filtered_sql, filtered_app = _predict(
        xgb_model,
        dataset_name,
        filtered_profile_df,
        split_fn=split_fn,
        qual_tool_filter=qual_filter,
        calib_params=calib_params
    )

    # merge results and join w/ qual_preds
    label = get_label()
    raw_sql_cols = {
        'appId': 'appId',
        'sqlID': 'sqlID',
        'scaleFactor': 'scaleFactor',
        'appDuration': 'appDuration',
        label: label,
        f'gpu_{label}': f'Actual GPU {label}',
        'y': 'Actual speedup',
        f'{label}_supported': f'QX {label}_supported',
        f'{label}_pred': f'QX {label}_pred',
        'y_pred': 'QX speedup',
        'split': 'split',
    }
    raw_sql = raw_sql.rename(raw_sql_cols, axis=1)
    raw_cols = [col for col in raw_sql_cols.values() if col in raw_sql]

    filtered_sql_cols = {
        'appId': 'appId',
        'appDuration': 'appDuration',
        'sqlID': 'sqlID',
        'scaleFactor': 'scaleFactor',
        label: label,
        f'{label}_supported': f'QXS {label}_supported',
        f'{label}_pred': f'QXS {label}_pred',
        # 'y_pred': 'QX speedup',
        'speedup_pred': 'QXS speedup',
    }
    filtered_sql = filtered_sql.rename(filtered_sql_cols, axis=1)

    results_sql = raw_sql[raw_cols].merge(
        filtered_sql[filtered_sql_cols.values()],
        on=['appId', 'sqlID', 'scaleFactor', 'appDuration', label],
        how='left',
        suffixes=[None, '_filtered']
    )

    raw_app_cols = {
        'appId': 'appId',
        'appDuration': 'appDuration',
        'speedup_actual': 'Actual speedup',
        f'{label}_pred': f'QX {label}_pred',
        f'{label}_supported': f'QX {label}_supported',
        'fraction_supported': 'QX fraction_supported',
        'appDuration_pred': 'QX appDuration_pred',
        'speedup': 'QX speedup',
    }
    raw_app = raw_app.rename(raw_app_cols, axis=1)

    filtered_app_cols = {
        'appId': 'appId',
        'appDuration': 'appDuration',
        f'{label}_pred': f'QXS {label}_pred',
        f'{label}_supported': f'QXS {label}_supported',
        'fraction_supported': 'QXS fraction_supported',
        'appDuration_pred': 'QXS appDuration_pred',
        'speedup': 'QXS speedup',
    }
    filtered_app = filtered_app.rename(filtered_app_cols, axis=1)

    results_app = raw_app[raw_app_cols.values()].merge(
        filtered_app[filtered_app_cols.values()],
        on=['appId', 'appDuration'],
        how='left',
    )

    print(
        '\nComparison of qualx raw (QX) and qualx w/ stage filtering (QXS)'
    )
    print(tabulate(results_app, headers='keys', tablefmt='psql', floatfmt='.2f'))
    print()

    # compute mean abs percentage error (MAPE) for each tool (QX, QXS)

    score_dfs = []
    for granularity, split in [('sql', 'test'), ('sql', 'all'), ('app', 'all')]:
        res = results_app if granularity == 'app' else results_sql
        res = res[res.split == 'test'] if split == 'test' else res
        if res.empty:
            logger.warning(
                'No per-%s %s evaluation results found for dataset: %s',
                granularity,
                split,
                dataset,
            )
            continue

        if 'Actual speedup' not in res:
            logger.error('No GPU rows found for dataset: %s', dataset)

        scores = compute_accuracy(
            res,
            'Actual speedup',
            {'QX': 'QX speedup', 'QXS': 'QXS speedup'},
            'appDuration' if granularity == 'app' else label,
        )

        score_df = pd.DataFrame(scores)
        print(f'Scores per {granularity} ({split}):')
        print(tabulate(score_df.transpose(), headers='keys', floatfmt='.4f'))
        print()

        # add identifying columns for these scores
        score_df = score_df.reset_index().rename(columns={'index': 'score'})
        score_df.insert(loc=0, column='model', value=model)
        score_df.insert(loc=1, column='platform', value=platform)
        score_df.insert(loc=2, column='dataset', value=dataset_name)
        score_df.insert(loc=3, column='granularity', value=granularity)
        score_df.insert(loc=4, column='split', value=split)
        score_dfs.append(score_df)

    # write mape scores as CSV
    scores_path = os.path.join(output_dir, f'{dataset_name}_mape.csv')
    if score_dfs:
        ds_scores_df = pd.concat(score_dfs)
    else:
        ds_scores_df = pd.DataFrame(
            columns=['model', 'platform', 'dataset', 'granularity', 'split', 'score', 'QX', 'QXS']
        )
    ds_scores_df.to_csv(scores_path, index=False)

    # write results as CSV
    sql_predictions_path = os.path.join(output_dir, f'{dataset_name}_sql.csv')
    logger.info('Writing per-SQL predictions to: %s', sql_predictions_path)
    results_sql.to_csv(sql_predictions_path, index=False, na_rep='nan')

    app_predictions_path = os.path.join(output_dir, f'{dataset_name}_app.csv')
    logger.info('Writing per-application predictions to: %s', app_predictions_path)
    results_app.to_csv(app_predictions_path, index=False, na_rep='nan')


def evaluate_summary(
    evaluate: str,  # pylint: disable=W0621
    *,
    score: str = 'dMAPE',
    split: str = 'test',
    config: Optional[str] = None,
) -> None:
    """
    Compute
    Parameters
    ----------
    evaluate: str
        Path to evaluation results directory.
    score: str
        Type of score to compare: 'MAPE', 'wMAPE', or 'dMAPE' (default).
    split: str
        Dataset split to compare: 'test' (default), 'train', or 'all'.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    get_config(config)

    summary_df, _ = _read_platform_scores(evaluate, score, split)
    print(tabulate(summary_df, headers='keys', tablefmt='psql', floatfmt='.4f'))
    summary_path = f'{evaluate}/summary.csv'
    logger.info('Writing per-platform %s scores for \'%s\' split to: %s', score, split, summary_path)
    summary_df.to_csv(summary_path)


def compare(
    previous: str,
    current: str,
    *,
    score: str = 'dMAPE',
    granularity: str = 'sql',
    split: str = 'all',
    config: Optional[str] = None,
) -> None:
    """Compare evaluation results between versions of code/models.

    This will print the delta of scores and also save them as a CSV file into the `current` directory.

    Parameters
    ----------
    previous: str
        Path to previous evaluation results directory.
    current: str
        Path to current evaluation results directory.
    score: str
        Type of score to compare: 'MAPE', 'wMAPE', or 'dMAPE' (default).
    granularity: str
        Granularity of score to compare: 'sql' (default) or 'app'.
    split: str
        Dataset split to compare: 'all' (default), 'train', or 'test'.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    get_config(config)

    # compare MAPE scores per dataset
    curr_df = _read_dataset_scores(current, score, granularity, split)
    prev_df = _read_dataset_scores(previous, score, granularity, split)

    compare_df = prev_df.merge(
        curr_df,
        how='right',
        on=['model', 'platform', 'dataset', 'granularity', 'split', 'score'],
        suffixes=('_prev', None),
    )
    for score_type in ['QX', 'QXS']:
        compare_df[f'{score_type}_delta'] = (
            compare_df[score_type] - compare_df[f'{score_type}_prev']
        )
    print(tabulate(compare_df, headers='keys', tablefmt='psql', floatfmt='.4f'))
    comparison_path = f'{current}/comparison_dataset.csv'
    logger.info('Writing dataset evaluation comparison to: %s', comparison_path)
    compare_df.to_csv(comparison_path)

    # compare app MAPE scores per platform
    # note: these aggregated scores can be impacted by the addition/removal of datasets
    # for removed datasets, just filter out from 'previous' aggregations
    # for added datasets, warn after printing table for more visibility
    curr_df, curr_datasets = _read_platform_scores(current, score, split)
    prev_df, prev_datasets = _read_platform_scores(
        previous, score, split, list(curr_datasets)
    )

    compare_df = prev_df.merge(
        curr_df,
        how='right',
        on=['model', 'platform'],
        suffixes=('_prev', None),
    )
    for score_type in ['QX', 'QXS']:
        compare_df[f'{score_type}_delta'] = (
            compare_df[score_type] - compare_df[f'{score_type}_prev']
        )
    print(tabulate(compare_df, headers='keys', tablefmt='psql', floatfmt='.4f'))
    comparison_path = f'{current}/comparison_platform.csv'
    logger.info('Writing platform evaluation comparison to: %s', comparison_path)
    compare_df.to_csv(comparison_path)

    # warn user of any new datasets
    added = curr_datasets - prev_datasets
    if added:
        logger.warning('New datasets added, comparisons may be skewed: added=%s', added)


def shap(
    platform: str,
    prediction_output: str,
    index: int,
    model: Optional[str] = None,
    config: Optional[str] = None,
) -> None:
    """Print a SHAP waterfall of features and their SHAP value contributions to the overall prediction for
    a specific index (sqlID) in the shap_values.csv file produced by prediction.

    Output description:
    - features are listed in order of importance (absolute value of `shap_value`), similar to a SHAP waterfall plot.
    - `model_rank` shows the feature importance rank on the training set.
    - `model_shap_value` shows the feature shap_value on the training set.
    - `train_[mean|std|min|max]` show the mean, standard deviation, min and max values of the feature in the
    training set.
    - `train_[25%|50%|75%]` show the feature value at the respective percentile in the training set.
    - `feature_value` shows the value of the feature used in prediction (for the indexed row/sqlID).
    - `out_of_range` indicates if the `feature_value` used in prediction was outside of the range of values seen in
    the training set.
    - `Shap base value` is the model's average prediction across the entire training set.
    - `Shap values sum` is the sum of the `shap_value` column for this indexed instance.
    - `Shap prediction` is the sum of `Shap base value` and `Shap values sum`, representing the model's predicted value.
    - `exp(prediction)` is the exponential of `Shap prediction`, which represents the predicted speedup
    (since the XGBoost model currently predicts `log(speedup)`).
    - the predicted speedup (which should match `y_pred` in `per_sql.csv`) is applied to the "supported" durations
    and combined with the unsupported" durations to produce a final per-sql speedup (`speedup_pred` in `per_sql.csv`).

    Parameters
    ----------
    platform: str
        Platform used during prediction.
    prediction_output: str
        Path to prediction output directory containing predictions.csv and xgboost_predictions folder.
    index: int
        Index of row/instance in shap_values.csv file to isolate for shap waterfall.
    model: Optional[str]
        Path to XGBoost model used in prediction.
    config:
        Path to a qualx-conf.yaml file to use for configuration.
    """
    # load config from command line argument, or use default
    get_config(config)

    # get model shap values w/ feature distribution metrics
    model_json_path = _get_model_path(platform, model)
    model_shap_path = Path(model_json_path).with_suffix('.metrics')
    if os.path.exists(model_shap_path):
        logger.info('Reading model metrics from: %s', model_shap_path)
        model_shap_df = pd.read_csv(model_shap_path, index_col=0)
        model_shap_df = model_shap_df.reset_index().rename(
            columns={'index': 'model_rank', 'shap_value': 'model_shap_value'}
        )
        model_col_names = {
            c: 'train_' + c
            for c in model_shap_df.columns
            if c not in ['feature', 'model_rank', 'model_shap_value']
        }
        model_shap_df = model_shap_df.rename(columns=model_col_names)
    else:
        logger.info('No model metrics found for: %s', model_json_path)
        model_shap_df = pd.DataFrame()

    # get prediction shap values and isolate specific instance
    prediction_paths = find_paths(prediction_output, lambda f: f == 'shap_values.csv')
    if len(prediction_paths) == 1:
        prediction_dir = Path(prediction_paths[0]).parent
    elif len(prediction_paths) > 1:
        raise ValueError(f'Found multiple shap_values.csv files in: {prediction_output}')
    else:
        raise FileNotFoundError(f'File: shap_values.csv not found in: {prediction_output}')

    df = pd.read_csv(os.path.join(prediction_dir, 'shap_values.csv'))
    instance_shap = df.iloc[index]

    # extract the SHAP expected/base value of the model
    expected_value = instance_shap['expected_value']

    # convert instance to dataframe where rows are features
    instance_shap_df = (
        instance_shap.drop('expected_value')
        .to_frame(name='shap_value')
        .reset_index()
        .rename(columns={'index': 'feature'})
    )

    # get features used in prediction and isolate specific instance
    df = pd.read_csv(os.path.join(prediction_dir, 'features.csv'))
    instance_features = df.iloc[index]

    # convert instance to dataframe where rows are features
    instance_features_df = (
        instance_features.to_frame(name='feature_value')
        .reset_index()
        .rename(columns={'index': 'feature'})
    )

    # merge instance shap dataframe w/ model metrics (if available)
    if not model_shap_df.empty:
        instance_shap_df = instance_shap_df.merge(model_shap_df, how='left', on='feature')

    # merge instance shap dataframe w/ prediction features
    instance_shap_df = instance_shap_df.merge(instance_features_df, how='left', on='feature')

    # add out-of-range indicator (if model metrics available)
    if not model_shap_df.empty:
        instance_shap_df['out_of_range'] = (instance_shap_df['feature_value'] < instance_shap_df['train_min']) | (
            instance_shap_df['feature_value'] > instance_shap_df['train_max']
        )

    # sort by absolute value of shap_value
    instance_shap_df = instance_shap_df.sort_values('shap_value', ascending=False, key=abs)
    instance_shap_df.reset_index(drop=True, inplace=True)

    formats = {
        'feature': '',
        'shap_value': '0.4f',
        'model_rank': '',
        'model_shap_value': '0.4f',
        'train_mean': '0.1e',
        'train_std': '0.1e',
        'train_min': '0.1e',
        'train_25%': '0.1e',
        'train_50%': '0.1e',
        'train_75%': '0.1e',
        'train_max': '0.1e',
        'feature_value': '0.1e',
        'out_of_range': '',
    }
    format_list = [''] + [formats[col] for col in instance_shap_df.columns]  # index + cols

    # print instance shap values w/ model metrics
    print(tabulate(instance_shap_df, headers='keys', tablefmt='psql', floatfmt=format_list))

    # print shap prediction and final prediction
    shap_sum = instance_shap_df.shap_value.sum()
    prediction = expected_value + shap_sum
    print(f'Shap base value: {expected_value:.4f}')
    print(f'Shap values sum: {shap_sum:.4f}')
    print(f'Shap prediction: {prediction:.4f}')
    print(f'exp(prediction): {np.exp(prediction):.4f}')


def entrypoint() -> None:
    """
    These are commands are intended for internal usage only.
    """
    cmds = {
        'models': models,
        'preprocess': preprocess,
        'train': train,
        'predict': _predict_cli,
        'evaluate': evaluate,
        'evaluate_summary': evaluate_summary,
        'compare': compare,
        'shap': shap,
    }
    fire.Fire(cmds)


if __name__ == '__main__':
    entrypoint()
