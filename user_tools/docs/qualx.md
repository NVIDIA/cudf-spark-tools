# QualX: NVIDIA cuDF plugin for Apache Spark Qualification Tool using XGBoost

Tool to qualify Spark applications for GPU acceleration, based on XGBoost.

## Usage

### Prediction

To predict the speedup of running a Spark application with the cuDF plugin on GPUs, use the cmd below. It uses an XGBoost model trained on matching CPU and GPU runs of various Spark applications.

```bash
spark_rapids prediction \
--platform <platform_name> \
--qual_output </path/to/qual_output> \
--output_folder </path/to/save/csv/files>
```

Arguments:
- platform: one of the following: "databricks-aws", "databricks-azure", "dataproc", "emr", "onprem" (default).
- qual_output: Path to a directory containing qualification tool output.
- output_folder: Path to store the output.
- custom_model_file: (OPTIONAL) Path to a custom-trained model.json file.
- qualx_config: (OPTIONAL) Path to a qualx-conf.yaml file to use for configuration. If not provided, the wrapper will use the default: https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-conf.yaml.

Output files:
- prediction.csv: per-application speedup predictions.
- xgboost_predictions/per_sql.csv: per-sql speedup predictions.
- xgboost_predictions/per_app.csv: raw (unformatted) per-application speedup predictions.
- xgboost_predictions/features.csv: feature values used as input to model prediction.
- xgboost_predictions/feature_importance.csv: shapley feature importance values for prediction dataset.
- xgboost_predictions/shap_values: per-sql shapley values for prediction dataset.

### Training

To train an XGBoost model on a specific dataset, follow these steps below. Refer to [Getting Started](https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/README.md#getting-started) section for installing the required dependencies for training.

#### Environment Setup

Training requires the following environment variables to be set:
```bash
export SPARK_HOME=/path/to/spark
export QUALX_DATA_DIR=/path/to/qualx/datasets
```

Since training internally uses the Profiling and Qualification tools, it requires the core tools jar as a dependency.
By default, QualX automatically locates and uses the bundled core tools jar from the package resources.

Notes:
- `QUALX_DATA_DIR` should be a valid local path containing the training data.

##### Advanced Environment Setup

For advanced users or specific deployment scenarios, user can override the embedded core tools jar that is pre-packaged
with user_tools.  To do this, user can set the following environment variable:
```bash
export SPARK_RAPIDS_TOOLS_JAR=/path/to/custom/tools.jar
```

- For users building from source or needing custom jar builds, refer to the [user-tools README](../README.md#building-from-source) for build instructions.

#### Data Collection

Training expects the training data (eventlogs) to be collected in the location specified by `QUALX_DATA_DIR`.  This directory MUST be structured as follows:
```
${QUALX_DATA_DIR}/<platform>/<dataset>/<app_name>/eventlogs/{cpu,gpu}/*
```

Where:
- platform: one of the supported platforms, e.g. `databricks-aws`, `databricks-azure`, `dataproc`, `emr`, or `onprem`.
- dataset: unique name for a group of related Spark applications, e.g. `nds`.
- app_name: unique name for a specific run of a Spark application, e.g. `powerrun`.  Note that this can vary from the actual Spark application name, as long as it uniquely identifies this dataset.

For example, given eventlogs for an on-prem NDS powerrun, we might see something like this:
```
$QUALX_DATA_DIR
└── onprem
    └── nds
       └── powerrun
           └── eventlogs
               ├── cpu
               │   └── app-20240412202337-0001
               └── gpu
                   └── app-20240412202939-0000
```

#### Dataset Metadata

Once the eventlogs have been collected to the `QUALX_DATA_DIR`, we must define some metadata about the dataset in JSON format.

For simplicitly, we will continue the example from above:
```bash
cat <<EOF > datasets/onprem/nds.json
{
    "nds_powerrun": {
        "eventlogs": [
            "${QUALX_DATA_DIR}/onprem/nds/powerrun/eventlogs/cpu",
            "${QUALX_DATA_DIR}/onprem/nds/powerrun/eventlogs/gpu"
        ],
        "app_meta": {
            "app-20240412202337-0001": {"runType": "CPU", "scaleFactor": 1},
            "app-20240412202939-0000": {"runType": "GPU", "scaleFactor": 1},
        }
    }
}
EOF
```

Notes:
- The top level key is a unique name for this set of eventlogs.  Generally, this is the combination of the dataset and app_name by convention.
- The "eventlogs" key is required and should point to the location of the eventlogs on disk.
- The "app_meta" key defines application-specific metadata, such as "runType" and "scaleFactor".
  - The "scaleFactor" key allows multiple eventlogs from the same application run at different scales.

These JSON metadata files should be collected into a single directory where the first level must be a supported platform, e.g.
```
datasets
└── onprem
    └── nds.json
```

#### Command

After the datasets have been collected and the metadata defined, run training as follows:
```bash
spark_rapids train \
--dataset </path/to/dataset/files(s)> \
--model </path/to/save/trained/model> \
--output_folder </path/to/save/csv/files>
```

Arguments:
- dataset: Path to a folder containing one or more dataset JSON files.
- model: Path to save the trained XGBoost model.
- output_folder: Path to store the output.
- n_trials: (OPTIONAL) Number of trials for hyperparameter search, default: 200.
- base_model: (OPTIONAL) Path to pre-trained model to serve as a base for fine-tuning/continued-training.
- features_csv_dir: (OPTIONAL) Path to a directory containing one or more features.csv files to augment the training dataset.
- qualx_config: (OPTIONAL) Path to a qualx-conf.yaml file to use for configuration. If not provided, the wrapper will use the default: https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/src/spark_rapids_pytools/resources/qualx-conf.yaml.

Continuing the example from above, we would use:
```bash
spark_rapids train \
--dataset datasets \
--model custom_onprem.json \
--output_folder train_output
```

Once satisfied with the model, just supply the path to this model in the `--custom_model_file` argument for prediction.

### Train and Evaluate Pipeline

To continually train and evaluate a custom XGBoost model on eventlogs collected over time, run the following periodically:
```bash
spark_rapids train_and_evaluate \
--qualx_pipeline_config qualx-pipeline-conf.yaml
--output_folder train_and_evaluate_output
```

Arguments:
- qualx_pipeline_config: Path to YAML config file containing the required training parameters.

#### Configuration
The training and evaluation pipeline is configured via the `qualx-pipeline-conf.yaml` file, which should have the following format:
```yaml
# dataset/model
dataset_name: mydataset
platform: onprem
alignment_dir: /path/to/alignments
eventlogs:
  cpu:
    - /path/to/CPU/eventlogs
  gpu:
    - /path/to/GPU/eventlogs
output_dir: pipeline
# qualx
cache_dir: qualx_cache
datasets: datasets
featurizers:
  - default.py
  - hash_plan.py
modifiers:
  - align_sql_id.py
label: duration_sum
split_functions:
  train:
    path: split_stratified.py
    args:
      label_col: duration_sum_speedup
      threshold: 1.0
  test: split_all_test.py
model_type: xgboost
xgboost:
  model_name: xgb_model.json
  n_trials: 200
  qual_tool_filter: stage
```

The `dataset/model` section must be updated for each new pipeline that produces a model.
- `dataset_name` - unique name for this dataset/model.
- `platform` - platform supported by the Profiling/Qualification tools, pick the closest one.
- `alignment_dir` - path to a directory containing alignment CSV files.
- `eventlogs` - paths to CPU and GPU eventlogs.
- `output_dir` - path to save pipeline artifacts like the trained model and evaluation results.

The `qualx` section can just use the defaults shown (except for more advanced use cases).

#### Alignment

The `alignment_dir` should point to a directory that contains one or more alignment CSV files.  These CSV files are used to match CPU appIds to GPU appIds (and CPU sqlIDs to GPU sqlIDs, if provided).  This allows computation of the actual speedups from a CPU/GPU eventlog pair.  The speedups are subsequently used as labels for model training and evaluation.

The CSV file(s):
- must be named `{dataset_name}.csv`, e.g. `mydataset.csv`.
- must have a header line with the following required columns: `appId_cpu,appId_gpu`
- may have the following additional columns: `sqlID_cpu,sqlID_gpu`
- can have rows representing only incremental/new rows.
- can have rows representing all rows seen to date.

#### Pipeline workflow

The files in the `alignment_dir` directory drive the pipeline execution, per the following pseudo-code:

```
if `{dataset_name}_*.inprogress` file exists in alignment directory
    use that file as the "current" file to process
else
    look for a `{dataset_name}.csv` in alignment directory and use this file as the "current" file to process

compute the delta of the "current" file vs. all previously processed files (in directory) to identify new rows
if there are no new rows
    exit with "no new rows"
else
    copy `{dataset_name}.csv` to `{dataset_name}_YYYYMMDD.inprogress`

create a new dataset from the new rows
if a previously-trained model exists
    evaluate the new dataset using the previous model

split the new dataset into train/val/test splits
train a new model on all datasets, including the new one
evaluate the newly-trained model on the all datasets, including the new one
move `{dataset_name}_YYYYMMDD.inprogress` to `{dataset_name}_YYYYMMDD.csv`
```

So, over time, this alignment directory may look like this:
```bash
alignment
├── mydataset_20250405161629.csv
├── mydataset_20250407174118.csv
├── mydataset_20250408165829.csv
├── mydataset_20250408172843.csv
├── mydataset_20250417221233.csv
├── mydataset_20250419020312.inprogress
└── mydataset.csv
```

Note that the presence of an `*.inprogress` file may indicate that a pipeline is currently running or it failed to complete the last run.  If the timestamp is stale, review the logs for failures.

#### Pipeline output

Successful execution of the pipeline will save the latest trained model and evaluation results to the `output_dir`.

For example:
```bash
pipeline
├── evaluate
│   ├── mydataset_20250405161629_app.csv
│   ├── mydataset_20250405161629_mape.csv
│   ├── mydataset_20250405161629_sql.csv
│   ├── mydataset_20250407174118_app.csv
│   ├── mydataset_20250407174118_mape.csv
│   ├── mydataset_20250407174118_sql.csv
│   ├── mydataset_20250408165829_app.csv
│   ├── mydataset_20250408165829_mape.csv
│   ├── mydataset_20250408165829_sql.csv
│   ├── mydataset_20250408172843_app.csv
│   ├── mydataset_20250408172843_mape.csv
│   ├── mydataset_20250408172843_sql.csv
│   ├── mydataset_20250417221233_app.csv
│   ├── mydataset_20250417221233_mape.csv
│   ├── mydataset_20250417221233_sql.csv
│   ├── mydataset_20250419020312_app.csv
│   ├── mydataset_20250419020312_mape.csv
│   └── mydataset_20250419020312_sql.csv
└── xgboost
    ├── xgb_model.cfg
    ├── xgb_model.json
    └── xgb_model.metrics
```

The model will be saved to the `xgboost` subdirectory with the following files:
- `*.cfg` - hyperparameters used to produce the model.
- `*.json` - xgboost model saved in JSON format.
- `*.metrics` - feature shap values and statistics.

The evaluation results will be saved to the `evaluate` subdirectory with the following files (per dataset):
- `*_app.csv` - per-app predicted and actual speedups.
- `*_mape.csv` - per-app and per-sql MAPE (mean absolute percentage error) scores.
- `*_sql.csv` - per-sql predicted and actual speedups.

Previous output directies will be archived with same datetime stamp as the most recent pipeline execution, e.g.:
```bash
pipeline_20250407174118
pipeline_20250408165829
pipeline_20250408172843
pipeline_20250417221233
pipeline_20250419020312
pipeline
```

The `pipeline` directory will contain the most recently trained model and evaluations, while the previous output will be renamed/archived to `pipeline_20250419020312`.  Note that the previous output will contain the evaluation results of the most recent dataset against the previous model.  This allows for a rolling evaluation of each (prior) model on entirely unseen datasets.

### Training (Advanced)
#### Fine-tuning / Incremental Training

To continue training an existing pre-trained model on new data, just set up the new dataset per above and then
reference the existing base model when training:
```bash
spark_rapids train \
--dataset datasets \
--model custom_onprem.json \
--base_model user_tools/src/spark_rapids_pytools/resources/qualx/models/xgboost/onprem.json \
--output_folder train_output
```

Once satisfied with the model, just supply the path to this model in the `--custom_model_file` argument for prediction.

#### Training with features.csv

During prediction, a `features.csv` file is written to the prediction output to help understand and debug the
predictions.  This file shows the raw feature values used as input to the XGBoost model.  The values are derived
from the output of the Profiling and Qualification tools, which in turn, is derived from the Spark eventlogs.

In situations where the `features.csv` file is available, but the none of the other upstream data sources are
available, these files can be used to supplement training.  The disadvantage is that these files are fixed snapshots
of the features generated by a specific version of the code, so they will become outdated as the code evolves over
time.

Note that these files do not contain the label required for training the model, since they were produced from
prediction.  So, in order to use these files for training, the label column, `Duration_speedup`, must be added,
along with the observed speedup values.

For example, if the model predicted a speedup of 1.5, but the actual speedup was observed to be 1.0, modify the
`features.csv` file as follows.
```python
import os
import pandas as pd
df = pd.read_csv('features.csv')
df['Duration_speedup'] = 1.0
os.mkdir('features')
df.to_csv('features/features_with_label.csv', index=False)
```

Then, train a custom model with the `--features_csv_dir features` argument.

Once satisfied with the model, just supply the path to this model in the `--custom_model_file` argument for prediction.

#### Dataset-specific Plugins

In certain situations, a dataset may require custom handling.  For these cases, we provide a plugin mechanism
for custom code that can be attached to that dataset.  The plugin implementation is just a python file that defines
any of the following functions:
```python
import pandas as pd

def load_profiles_hook(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Custom post processing on the load_profiles dataframe.

    Notes:
    - profile_df contains "raw" features for the target dataset, extracted from the Profiler tool's output CSV files.
    - profile_df is a slice of the original dataframe, so the original indices must be preserved,
      and any new columns will be ignored/dropped.
    """
    # Insert custom code to modify the profile_df as needed.
    return profile_df


def split_function(cpu_aug_tbl: pd.DataFrame) -> pd.DataFrame:
    """Custom train/test/val split function.

    Notes:
    - cpu_aug_tbl contains the "model" features for the target dataset, which will be used in training.
    - cpu_aug_tbl is a slice of the original dataframe, so the original indices must be preserved,
      and any new columns will be ignored/dropped.
    - if not supplied, the default split function will randomly split the data by ratios of 60/20/20.
    """
    # Insert custom code to set cpu_aug_tbl['split'] to 'train', 'test', or 'val'.
    return cpu_aug_tbl
```

In order to use a custom plugin, just reference it in the associated dataset JSON file:
```
# datasets/onprem/my_custom_dataset.json
{
    "my_custom_dataset": {
        "eventlogs": [
            "/path/to/eventlogs"
        ],
        "app_meta": {
            ...
        },
        "load_profiles_hook": "/path/to/custom_plugin.py",
        "split_function": "/path/to/custom_plugin.py"
    }
}
```
