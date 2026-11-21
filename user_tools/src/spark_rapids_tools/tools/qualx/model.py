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

""" Model training and prediction functions for QualX """

from typing import Callable, Mapping, Optional, List, Tuple, Union, Dict
import json
import math
import shap
import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import Booster

from sklearn.model_selection import train_test_split
from scipy.optimize import least_squares

from spark_rapids_tools.tools.qualx.config import get_config, get_cost_function, get_label
from spark_rapids_tools.tools.qualx.preprocess import expected_raw_features
from spark_rapids_tools.tools.qualx.util import get_logger
# Import optional packages
try:
    import optuna
except ImportError:
    optuna = None

logger = get_logger(__name__)

FILTER_SPILLS = False  # remove queries with any disk/mem spill
LOG_LABEL = True  # use log(y) as target
LABEL_ATTR = 'qualx_label'  # model attribute holding the label that the model was trained on

# non-training features (and labels)
ignored_features = {
    'appDuration',
    'appTaskTime',
    'appWallTime',
    'appId',
    'appName',
    'description',
    'fraction_supported',
    'hash',
    'jobStartTime_min',
    'pluginEnabled',
    'runType',
    'scaleFactor',
    'sparkRuntime',
    'sparkVersion',
    'sqlID'
}


def compute_sample_weights(
    y_tune: pd.Series,
    threshold: float,
    positive_weight: Union[float, str],
    negative_weight: Union[float, str],
) -> Tuple[float, float]:
    """Automatically compute sample weights for positive/negative samples.

    Note: the default/minimum weight is 1.0.
    """
    if positive_weight == 'auto' or negative_weight == 'auto':
        num_positives = y_tune[y_tune > threshold].count()
        num_negatives = y_tune[y_tune <= threshold].count()
        if positive_weight == 'auto':
            positive_weight = np.max([num_negatives / num_positives, 1.0]) if num_positives > 0 else 1.0
        if negative_weight == 'auto':
            negative_weight = np.max([num_positives / num_negatives, 1.0]) if num_negatives > 0 else 1.0
    return positive_weight, negative_weight


def train(
    cpu_aug_tbl: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    n_trials: int = 200,
    base_model: Optional[Booster] = None,
    hyperparams: Optional[Dict[str, any]] = None,
) -> Booster:
    """Train model on preprocessed data."""
    if 'split' not in cpu_aug_tbl.columns:
        raise ValueError(
            'Training data must have a split column with values: train, val, test'
        )

    if LOG_LABEL:
        cpu_aug_tbl = cpu_aug_tbl.copy()
        cpu_aug_tbl[label_col] = np.log(cpu_aug_tbl[label_col])

    # remove nan label entries
    original_num_rows = cpu_aug_tbl.shape[0]
    cpu_aug_tbl = cpu_aug_tbl.loc[~cpu_aug_tbl[label_col].isna()].reset_index(
        drop=True
    )
    if cpu_aug_tbl.shape[0] < original_num_rows:
        logger.debug(
            'Removed %d rows with NaN label values', original_num_rows - cpu_aug_tbl.shape[0]
        )

    # split into train/val/test sets
    x_train = cpu_aug_tbl.loc[cpu_aug_tbl['split'] == 'train', feature_cols]
    y_train = cpu_aug_tbl.loc[cpu_aug_tbl['split'] == 'train', label_col]
    d_train = xgb.DMatrix(x_train, y_train)

    x_val = cpu_aug_tbl.loc[cpu_aug_tbl['split'] == 'val', feature_cols]
    y_val = cpu_aug_tbl.loc[cpu_aug_tbl['split'] == 'val', label_col]
    d_val = xgb.DMatrix(x_val, y_val)

    # dtest should be held out of all train/val steps
    # X_test = cpu_aug_tbl.loc[cpu_aug_tbl['split']=='test', feature_cols]
    # y_test = cpu_aug_tbl.loc[cpu_aug_tbl['split']=='test', label_col]
    # dtest = xgb.DMatrix(X_test, y_test)

    # tune hyperparameters on full dataset
    cpu_aug_tbl = cpu_aug_tbl.sort_values('description').reset_index(drop=True)
    x_tune = cpu_aug_tbl.loc[cpu_aug_tbl['split'] != 'test', feature_cols]
    y_tune = cpu_aug_tbl.loc[cpu_aug_tbl['split'] != 'test', label_col]

    # get positive/negative sample weights from config, or default to 1.0
    cfg = get_config()
    sample_weight = cfg.__dict__.get('sample_weight', {})
    threshold = sample_weight.get('threshold', 1.0)
    positive_weight = sample_weight.get('positive', 1.0)
    negative_weight = sample_weight.get('negative', 1.0)

    if LOG_LABEL:
        threshold = np.log(threshold)

    # automatically compute weights (if 'auto' is specified) to balance positive/negative samples
    positive_weight, negative_weight = compute_sample_weights(y_tune, threshold, positive_weight, negative_weight)

    sample_weights = np.where(
        y_tune > threshold,
        positive_weight,
        negative_weight,
    )
    d_tune = xgb.DMatrix(x_tune, y_tune, weight=sample_weights)

    if base_model:
        # use hyperparameters from base model (w/ modifications to learning rate and num trees)
        xgb_params = {}
        cfg = json.loads(base_model.save_config())
        train_params = cfg['learner']['gradient_booster']['tree_train_param']
        xgb_params['eta'] = float(train_params['eta']) / 10.0     # decrease learning rate
        xgb_params['gamma'] = float(train_params['gamma'])
        xgb_params['max_depth'] = int(train_params['max_depth'])
        xgb_params['min_child_weight'] = int(train_params['min_child_weight'])
        xgb_params['subsample'] = float(train_params['subsample'])
        n_estimators = cfg['learner']['gradient_booster']['gbtree_model_param']['num_trees']
        n_estimators = int(float(n_estimators) * 1.1)              # increase n_estimators
    elif hyperparams:
        base_params = {
            'random_state': 0,
            'objective': 'reg:squarederror',
            'eval_metric': ['mae', 'mape'],  # applied to eval_set/test_data if provided
            'booster': 'gbtree',
        }
        xgb_params = {**base_params, **hyperparams}
        n_estimators = xgb_params.pop('n_estimators', 100)
    else:
        # use optuna hyper-parameter tuning
        best_params = tune_hyperparameters(x_tune, y_tune, n_trials, sample_weights)
        logger.info(best_params)

        # train model w/ the best hyperparameters using data splits
        base_params = {
            'random_state': 0,
            'objective': 'reg:squarederror',
            'eval_metric': ['mae', 'mape'],  # applied to eval_set/test_data if provided
            'booster': 'gbtree',
        }
        xgb_params = {**base_params, **best_params}
        n_estimators = xgb_params.pop('n_estimators')

    # train model
    evals_result = {}
    xgb_model = xgb.train(
        xgb_params,
        dtrain=d_tune,
        num_boost_round=n_estimators,
        evals=[(d_train, 'train'), (d_val, 'val')],
        verbose_eval=50,
        evals_result=evals_result,
        xgb_model=base_model,
    )
    # record the label in the model, so that prediction can check it against the cost function
    xgb_model.set_attr(**{LABEL_ATTR: get_label()})

    return xgb_model


def calibrate(
    cpu_aug_tbl: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    xgb_model: Booster,
) -> Dict[str, float]:
    """
    Calibration involves first training a pre-calib xgb model on a smaller random split of full
    training data (train+val) that was used for training the main model that we want to calibrate.
    This pre-calibrated model serves as an independent near-identical version of the main model.
    This surrogate model is needed instead of main model since the calibration bias is only observed
    on out-of-sample data (not seen previously in training). We refer to this split of the full
    training data as pre-calib holdout data.

    The calibration parameters are fit using the pre-calib model predictions
    on the remaining training data, that is referred to as calib holdout data.
    """

    # Create (stratified) random split of train data into precalib and calib data
    cpu_aug_tbl = cpu_aug_tbl[cpu_aug_tbl['split'].isin(['train', 'val'])
                              & cpu_aug_tbl[label_col].notna()].copy()
    cpu_aug_tbl['label_quantile'] = pd.qcut(cpu_aug_tbl[label_col], q=10, duplicates='drop')
    cpu_aug_tbl_precalib, cpu_aug_tbl_calib = train_test_split(
        cpu_aug_tbl,
        test_size=0.2,
        random_state=0,
        stratify=cpu_aug_tbl[['label_quantile']]
    )

    # Extract and use same hyperparams as main model we want to calibrate
    hyperparams = {}
    cfg = json.loads(xgb_model.save_config())
    train_params = cfg['learner']['gradient_booster']['tree_train_param']
    hyperparams['eta'] = float(train_params['eta'])
    hyperparams['gamma'] = float(train_params['gamma'])
    hyperparams['lambda'] = float(train_params['lambda'])
    hyperparams['max_depth'] = int(train_params['max_depth'])
    hyperparams['min_child_weight'] = int(train_params['min_child_weight'])
    hyperparams['subsample'] = float(train_params['subsample'])
    hyperparams['n_estimators'] = int(cfg['learner']['gradient_booster']['gbtree_model_param']['num_trees'])

    # Train precalib model as an independent near-identical version of main model
    xgb_model_precalib = train(cpu_aug_tbl_precalib, feature_cols, label_col, hyperparams=hyperparams)

    # Get predictions from precalib model on calib holdout samples
    x = cpu_aug_tbl_calib[xgb_model_precalib.feature_names]
    y_pred = xgb_model_precalib.predict(xgb.DMatrix(x))
    if LOG_LABEL:
        y_pred = np.exp(y_pred)

    # Sample importance weighting to address label imbalance across the range.
    # Divide the entire range of label into bins of roughly equal length
    # such that each bin has a minimum number of samples.
    # Samples in each bin get importance weight as the ratio of
    # the largest count of samples in any bin to the count of samples in its bin.
    max_num_bins = 10
    min_samples_per_bin = 5
    y = cpu_aug_tbl_calib[label_col]
    y_min, y_max, y_avg = np.min(y), np.max(y), np.average(y)
    calib_df = pd.DataFrame({'y': y, 'y_pred': y_pred})
    for num_bins in range(max_num_bins, 1, -1):
        bins = np.append(
            np.linspace(y_min*0.99, y_avg, (num_bins+1)//2 + 1)[:-1],
            np.exp(np.linspace(math.log(y_avg), math.log(y_max*1.01), math.ceil((num_bins+1)/2)))
        )
        bin_labels = np.arange(0, num_bins)
        calib_df['y_qz'] = pd.cut(calib_df['y'], bins, labels=bin_labels)
        # In group_by statement, Enforce observed=False as the default 'observed=False' is
        # deprecated and will be changed to True in a future version of pandas.
        y_qz_hist_df = calib_df \
            .groupby(['y_qz'], as_index=False, observed=False) \
            .agg(count=pd.NamedAgg('y_qz', 'size')) \
            .sort_values('y_qz')
        smallest_count_per_bin = np.min(y_qz_hist_df['count'])
        print(y_qz_hist_df)
        if smallest_count_per_bin >= min_samples_per_bin:
            break

    largest_count_per_bin = np.max(y_qz_hist_df['count'])
    calib_df = calib_df.merge(y_qz_hist_df, on=['y_qz'])
    calib_df['weight'] = largest_count_per_bin / calib_df['count']

    # Fit calibration params y ~ c1*y_pred^c2 + c3
    y_pred = calib_df['y_pred'].to_numpy()
    y = calib_df['y'].to_numpy()
    w = calib_df['weight'].to_numpy()
    if LOG_LABEL:
        c_argmin = least_squares(
                        fun=lambda c: (np.log(y) - np.log(c[0]*np.pow(y_pred, c[1])+c[2])) * np.sqrt(w),
                        x0=np.array([1.0, 1.0, 0.0]),
                        bounds=([0.25, 1.0, 0.0], [4.0, 2.5, 0.001])
                    ).x
    else:
        c_argmin = least_squares(
                fun=lambda c: (y - (c[0]*np.pow(y_pred, c[1])+c[2])) * np.sqrt(w),
                x0=np.array([1.0, 1.0, 0.0]),
                bounds=([1.0, 1.0, -0.1], [3.0, 2.5, 0.1])
            ).x

    calib_params = {
        'c1': round(float(c_argmin[0]), 4),
        'c2': round(float(c_argmin[1]), 4),
        'c3': round(float(c_argmin[2]), 4)
    }
    return calib_params


def predict(
    xgb_model: xgb.Booster,
    cpu_aug_tbl: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    calib_params: Dict[str, float] = None,
) -> pd.DataFrame:
    """Use model to predict on feature data."""
    cost_fn = get_cost_function()
    label = cost_fn.label  # Duration, duration_sum, or the LABEL of a plugin
    model_label = xgb_model.attr(LABEL_ATTR)
    if model_label is not None and model_label != label:
        # Check the model label against the cost function label, only if it was set.
        # Older QualX models may not have the model_label attribute.
        raise ValueError(f'Model was trained on the label {model_label}, but the cost function has the label {label}')

    model_features = xgb_model.feature_names

    missing = set(model_features) - set(feature_cols)
    extra = set(feature_cols) - set(model_features)
    if missing:
        raise ValueError(f'Input is missing model features: {missing}')
    if extra:
        logger.debug('Input had extra features not present in model: %s', extra)

    x = cpu_aug_tbl[model_features]
    y = cpu_aug_tbl[label_col] if label_col else None

    d_mat = xgb.DMatrix(x)
    y_pred = xgb_model.predict(d_mat)

    if LOG_LABEL:
        y_pred = np.exp(y_pred)

    cfg = get_config()
    calib = cfg.calib
    if calib and calib_params:
        c1 = calib_params['c1']
        c2 = calib_params['c2']
        c3 = calib_params['c3']
        y_pred = np.maximum(c1 * np.pow(y_pred, c2) + c3, 0.01)

    preds = {'y_pred': y_pred}
    if y is not None:
        preds['y'] = y
    preds_df = pd.DataFrame(preds)

    select_columns = [
        'appName',
        'appId',
        'appDuration',
        'sqlID',
        'scaleFactor',
        label,
        'fraction_supported',
        'description',
    ]
    # carry the columns that the cost function reads to the rollup
    select_columns += [col for col in cost_fn.input_columns if col not in select_columns]
    if 'split' in cpu_aug_tbl:
        select_columns.append('split')

    # join predictions with select input features
    results_df = (
        cpu_aug_tbl[select_columns]
        .reset_index(drop=True)
        .merge(preds_df, how='outer', left_index=True, right_index=True)
    )

    if 'y' in results_df.columns:
        # reconstruct original gpu duration for validation purposes
        results_df[f'gpu_{label}'] = results_df[label] / results_df['y']
        results_df[f'gpu_{label}'] = np.floor(results_df[f'gpu_{label}'])

    # adjust raw predictions with stage/sqlID filtering of unsupported ops
    results_df[f'{label}_pred'] = results_df[label] * (
        1.0
        - results_df['fraction_supported']
        + (results_df['fraction_supported'] / results_df['y_pred'])
    )
    # compute fraction of duration in supported ops
    results_df[f'{label}_supported'] = (
        results_df[label] * results_df['fraction_supported']
    )
    # compute adjusted speedup (vs. raw speedup prediction: 'y_pred')
    # without qual data, this should be the same as the raw 'y_pred'
    results_df['speedup_pred'] = results_df[label] / results_df[f'{label}_pred']

    return results_df


def extract_model_features(
    df: pd.DataFrame,
    split_functions: Mapping[str, Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Extract model features from raw features."""
    cost_fn = get_cost_function()
    label = cost_fn.label
    expected_model_features = expected_raw_features() - ignored_features - {label}
    missing = expected_raw_features() - set(df.columns)
    if missing:
        logger.warning('Input dataframe is missing expected raw features: %s', missing)

    # Compute each SQL's cost, and the app-level cost, based on the cost function.
    # `label` is the SQL-level cost, and appDuration is the app-level cost.
    df = cost_fn.with_label(df)
    df['appDuration'] = cost_fn.app_cost(df)

    if FILTER_SPILLS:
        df = df[
            (df['diskBytesSpilledRatio'] == 0) & (df['memoryBytesSpilled_sum'] == 0)
        ]

    # use CPU runs as primary dataset
    cpu_aug_tbl = df[df['runType'] == 'CPU']

    # remove gpu sql operators from base cpu augmented table.
    # sql_ops_list = [cc for cc in cpu_aug_tbl.columns if cc.startswith('sqlOp_')]
    gpu_sql_ops_list = [
        cc
        for cc in cpu_aug_tbl.columns
        if cc.startswith('sqlOp_Gpu') or 'GpuInsertIntoHadoopFsRelationCommand' in cc
    ]
    cpu_aug_tbl = cpu_aug_tbl.drop(columns=gpu_sql_ops_list)

    gpu_aug_tbl = df[df['runType'] == 'GPU']
    if gpu_aug_tbl.shape[0] > 0:
        if gpu_aug_tbl.shape[0] != cpu_aug_tbl.shape[0]:
            logger.debug(
                'Number of GPU rows (%d) does not match number of CPU rows (%d)',
                gpu_aug_tbl.shape[0],
                cpu_aug_tbl.shape[0],
            )
        # train/validation dataset with CPU + GPU runs
        gpu_aug_tbl = gpu_aug_tbl[
            [
                'appName',
                'scaleFactor',
                'sqlID',
                label,
                'description',
            ]
        ]
        gpu_aug_tbl = gpu_aug_tbl.rename(columns={label: f'xgpu_{label}'})
        cpu_aug_tbl = cpu_aug_tbl.merge(
            gpu_aug_tbl,
            on=['appName', 'scaleFactor', 'sqlID', 'description'],
            how='left',
        )

        # warn for possible mismatched sqlIDs
        num_rows = len(cpu_aug_tbl)
        num_na = cpu_aug_tbl[f'xgpu_{label}'].isna().sum()
        if (
            num_na / num_rows > 0.05
        ):  # arbitrary threshold, misaligned sqlIDs still may 'match' most of the time
            logger.debug(
                'Percentage of NaN GPU durations is high: %d / %d. Per-sql actual speedups may be inaccurate.',
                num_na,
                num_rows,
            )

        # calculate speedup
        cpu_aug_tbl[f'{label}_speedup'] = (
            cpu_aug_tbl[label] / cpu_aug_tbl[f'xgpu_{label}']
        )
        cpu_aug_tbl = cpu_aug_tbl.drop(columns=[f'xgpu_{label}'])

        # use speedup as label
        label_col = f'{label}_speedup'
    else:
        # inference dataset with CPU runs only
        label_col = None

    # add aggregations for time features
    # time_cols = ['executorCPUTime', 'executorDeserializeTime', 'executorDeserializeCPUTime', 'executorRunTime',
    #     'gettingResultTime', 'jvmGCTime', 'resultSerializationTime', 'sr_fetchWaitTime', 'sw_writeTime']
    # time_agg_cols = [cc + '_sum' for cc in time_cols]
    # time_ratio_cols = [cc for cc in cpu_aug_tbl.columns if cc.endswith('TimeRatio')]

    # TODO: investigate leaky_cols
    # leaky_cols = ['duration_min', 'duration_max', 'duration_mean'] + time_agg_cols + time_ratio_cols
    # ignore_cols = ignore_cols  #+ leaky_cols

    # remove non-training columns
    feature_cols = [cc for cc in cpu_aug_tbl.columns if cc not in ignored_features]

    # sanity check features
    actual = set(feature_cols)
    missing = expected_model_features - actual
    extra = actual - expected_model_features
    if missing:
        raise ValueError(f'Input data is missing model features: {missing}')
    if extra:
        # remove extra columns
        logger.debug('Input data has extra features (removed): %s', extra)
        feature_cols = [c for c in feature_cols if c not in extra]

    # add train/val/test split column, if split function(s) provided
    if split_functions:
        # ensure 'split' column in cpu_aug_tbl
        if 'split' not in cpu_aug_tbl.columns:
            cpu_aug_tbl['split'] = pd.Series(dtype='str')

        # save schema, since df.update() defaults all dtypes to floats
        df_schema = cpu_aug_tbl.dtypes

        # extract default split function, if present
        default_split_fn = split_functions.pop('default') if 'default' in split_functions else None

        # handle all other dataset-specific split functions
        for ds_name, split_fn in split_functions.items():
            dataset_df = cpu_aug_tbl.loc[
                (cpu_aug_tbl.appName == ds_name) | (cpu_aug_tbl.appName.str.startswith(f'{ds_name}:'))
            ]
            modified_df = split_fn(dataset_df)
            if modified_df.index.equals(dataset_df.index):
                cpu_aug_tbl.update(modified_df)
                cpu_aug_tbl.astype(df_schema)
            else:
                raise ValueError(f'Plugin: split_function for {ds_name} unexpectedly modified row indices.')
            cpu_aug_tbl.update(dataset_df)

        # handle default split function
        if default_split_fn:
            default_df = cpu_aug_tbl.loc[~cpu_aug_tbl.appName.isin(split_functions.keys())]
            for ds_name in split_functions.keys():
                default_df = default_df.loc[~default_df.appName.str.startswith(f'{ds_name}:')]
            modified_default_df = default_split_fn(default_df)
            if modified_default_df.index.equals(default_df.index):
                cpu_aug_tbl.update(modified_default_df)
                cpu_aug_tbl.astype(df_schema)
            else:
                raise ValueError('Default split_function unexpectedly modified row indices.')
    return cpu_aug_tbl, feature_cols, label_col


def tune_hyperparameters(x, y, n_trials: int = 200, sample_weights: Optional[np.ndarray] = None) -> dict:
    # use full training set for hyperparameter search

    xgb_tmp = xgb.XGBRegressor(objective='reg:squarederror')

    optuna_search_params = {
        'eta': optuna.distributions.FloatDistribution(0.001, 0.05),
        'gamma': optuna.distributions.FloatDistribution(0, 50),
        'max_depth': optuna.distributions.IntDistribution(5, 10),
        'min_child_weight': optuna.distributions.IntDistribution(1, 5),
        'subsample': optuna.distributions.FloatDistribution(
            0.6, 1.0
        ),  # Helps with over fitting
        'n_estimators': optuna.distributions.IntDistribution(50, 100),
    }

    # Optuna with cross-validation:
    optuna_search = optuna.integration.OptunaSearchCV(
        xgb_tmp,
        optuna_search_params,
        cv=5,
        enable_pruning=False,
        scoring='neg_mean_squared_error',
        random_state=0,
        n_trials=n_trials,
        timeout=900,
    )

    # run the search
    if sample_weights is not None:
        optuna_search.fit(x, y, sample_weight=sample_weights)
    else:
        optuna_search.fit(x, y)

    return optuna_search.best_params_


def compute_shapley_values(xgb_model: xgb.Booster, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Shapley values for the model given an input dataframe.

    Returns two dataframes representing:
    - feature importance
    - raw shapley values (per row of input dataframe)
    """
    pd.set_option('display.max_rows', None)
    explainer = shap.TreeExplainer(xgb_model)
    feature_cols = xgb_model.feature_names
    shap_values = explainer.shap_values(features[feature_cols])

    # raw shap values per row (w/ expected value of model)
    shap_values_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_values_df['expected_value'] = explainer.expected_value

    # feature importance
    shap_vals = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame(
        list(zip(feature_cols, shap_vals)), columns=['feature', 'shap_value']
    )
    feature_importance_df = feature_importance_df.sort_values(
        by=['shap_value'], ascending=False
    ).reset_index(drop=True)

    return feature_importance_df, shap_values_df
