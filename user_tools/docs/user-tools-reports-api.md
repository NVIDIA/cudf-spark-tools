# User Tools Reports API Documentation

This documentation provides a comprehensive guide for using the Spark Rapids Tools API v1 to create result handlers and read various report formats.

## Table of Contents

1. [Overview](#overview)
2. [Building Result Handlers](#building-result-handlers)
3. [Reading Report Data](#reading-report-data)
4. [Working with Applications](#working-with-applications)
5. [Return Types and Data Handling](#return-types-and-data-handling)
6. [Error Handling](#error-handling)
7. [Complete Examples](#complete-examples)
8. [Best Practices](#best-practices)

## Overview

The Spark Rapids Tools API v1 provides a fluent builder interface for:

- **Creating result handlers** for different tool outputs (qualification, profiling)
- **Loading data** from reports in multiple formats (CSV, JSON, JSON Properties, TXT)
- **Handling both global and per-application data** with type-safe operations
- **Managing cloud and local storage paths** seamlessly

### Supported Report Types

| Report Type | Description | Return Type |
|-------------|-------------|-------------|
| CSV | Tabular data with pandas DataFrames | `LoadDFResult` |
| JSON | JSON data (objects and arrays) | `JSONResult` |
| JSON Properties | Configuration and metadata | `JPropsResult` |
| TXT | Plain text logs and outputs | `TXTResult` |

## Building Result Handlers

### Basic Result Handler Creation

Result handlers are created using specialized classes for each tool output type:

```python
from spark_rapids_tools.api_v1 import QualCore, ProfCore, QualWrapper, ProfWrapper

# Qualification Core Output
qual_core_handler = QualCore("/path/to/qualification/core/output")

# Profiling Core Output
prof_core_handler = ProfCore("/path/to/profiling/core/output")

# Qualification Wrapper Output
qual_wrapper_handler = QualWrapper("/path/to/qualification/wrapper/output")

# Profiling Wrapper Output
prof_wrapper_handler = ProfWrapper("/path/to/profiling/wrapper/output")
```

### Path Configuration

The API supports both local and cloud storage paths:

```python
from spark_rapids_tools.api_v1 import QualCore
from spark_rapids_tools.storagelib.cspfs import BoundedCspPath

# Local filesystem path
local_handler = QualCore("/local/path/to/qualification/output")

# Cloud storage path (S3, GCS, Azure)
cloud_path = BoundedCspPath("s3://my-bucket/spark-rapids/qual_output")
cloud_handler = QualCore(cloud_path)

# HDFS path
hdfs_handler = QualCore("hdfs://namenode:port/path/to/qual_output")
```

## Reading Report Data

### CSV Reports

CSV reports provide tabular data wrapped in LoadDFResult objects containing pandas DataFrames.

#### Global Table Data

```python
from spark_rapids_tools.api_v1.builder import CSVReport

# Load global summary data
summary_result = (CSVReport(handler)
    .table("qualCoreCSVSummary")
    .load())

# Access the DataFrame
df = summary_result.data
print(f"Loaded {len(df)} rows of qualification data")
```

#### Per-Application Table Data

```python
# Load data for specific applications
app_results = (CSVReport(handler)
    .table("coreRawJobInformationCSV")
    .app("application_001")
    .app("application_002")
    .load())

# Load data for multiple applications at once
multi_app_results = (CSVReport(handler)
    .table("coreRawApplicationInformationCSV")
    .apps(["app_001", "app_002", "app_003"])
    .load())

# Load data for a single application (returns direct result)
single_app_result = (CSVReport(handler)
    .table("coreRawExecutorInformationCSV")
    .app("application_001")
    .load())
```

#### Advanced CSV Options

```python
# Custom pandas read arguments
custom_result = (CSVReport(handler)
    .table("coreRawApplicationInformationCSV")
    .pd_args({
        "sep": ",",
        "encoding": "utf-8",
        "parse_dates": ["App Start Time", "App End Time"]
    })
    .load())

# Column mapping for renaming
mapped_result = (CSVReport(handler)
    .table("coreRawJobInformationCSV")
    .map_cols({
        "Job ID": "JobIdentifier",
        "Job Duration": "DurationMs"
    })
    .load())

# Fallback data when table is missing
def create_fallback_data():
    import pandas as pd
    return pd.DataFrame({
        "App ID": ["fallback_app"],
        "Status": ["no_data"]
    })

result_with_fallback = (CSVReport(handler)
    .table("coreRawDataSourceInformationCSV")
    .fall_cb(create_fallback_data)
    .load())
```

### Java Properties Reports

Java Properties reports handle configuration data in `.properties` file format. Note that most configuration data is stored as CSV files instead.

```python
from spark_rapids_tools.api_v1 import JPropsReport, CSVReport

# Load global runtime properties (PROPERTIES format)
runtime_props = (JPropsReport(handler)
    .table("runtimeProperties")
    .load())

# Access property data
print(f"Tool version: {runtime_props.props.get('rapids.tools.version')}")

# Load Spark properties for applications (CSV format, not PROPERTIES)
spark_props_csv = (CSVReport(handler)
    .table("coreRawSparkPropertiesCSV")
    .app("application_001")
    .load())

# Access CSV data
if spark_props_csv.success:
    props_df = spark_props_csv.data
    print(f"Found {len(props_df)} Spark properties")
```

### TXT Reports

Text reports handle plain text data like logs, summaries, and JSONL files:

```python
from spark_rapids_tools.api_v1.builder import TXTReport

# Note: TXT format is primarily used for JSONL files (see JSON section)
# For example, loading SQL plan information in JSONL format
sql_plans = (TXTReport(handler)
    .table("coreRawSqlPlanPreAQEJson")
    .app("application_001")
    .load())

# Access text content (each line is a separate JSON object)
if sql_plans.success:
    print(f"Number of lines: {len(sql_plans.lines)}")
    for line in sql_plans.lines[:3]:  # First 3 lines
        print(line)
```

### JSON Reports

JSON reports handle structured data in JSON format, including both standard JSON (objects/arrays) and JSONL (JSON Lines) format:

```python
from spark_rapids_tools.api_v1 import QualCore, QualWrapper, JSONReport, TXTReport
import json

# Create handlers
qual_wrapper_handler = QualWrapper("/path/to/qualification/wrapper/output")
qual_core_handler = QualCore("/path/to/qualification/core/output")

global_json = (JSONReport(qual_wrapper_handler)
    .table("qualWrapperAppMetadataJson")
    .load())

if global_json.success:
    data_dict = global_json.to_dict()  # For JSON objects
    print(f"Application metadata: {data_dict}")

# Load JSON reports for multiple applications from core output
app_jsons = (JSONReport(qual_core_handler)
    .table("coreRawClusterInformationJson")
    .apps(["app_001", "app_002"])
    .load())

for app_id, json_result in app_jsons.items():
    if json_result.success:
        cluster_info = json_result.to_dict()
        print(f"App {app_id} cluster: {cluster_info['sourceClusterInfo']}")

# Load JSON report for a single application (returns array)
build_info = (JSONReport(qual_core_handler)
    .table("coreRawSparkRapidsBuildInfoJson")
    .app("application_001")
    .load())

if build_info.success:
    info_list = build_info.to_list()  # For JSON arrays
    print(f"cuDF plugin version: {info_list[0]['sparkRapidsBuildInfo']['version']}")
```

#### JSONL (JSON Lines) Format

For JSONL format files (one JSON object per line), use TXT reports and parse each line:

```python
# Load JSONL file (e.g., SQL plan information)
# Use qual_core_handler or prof_core_handler depending on your output
sql_plans_txt = (TXTReport(qual_core_handler)
    .table("coreRawSqlPlanPreAQEJson")
    .app("application_001")
    .load())

if sql_plans_txt.success:
    # Parse each line as a separate JSON object
    sql_plans = []
    for line in sql_plans_txt.lines:
        if line.strip():  # Skip empty lines
            plan = json.loads(line)
            sql_plans.append(plan)

    print(f"Loaded {len(sql_plans)} SQL plans")

    # Process each SQL plan
    for plan in sql_plans:
        sql_id = plan['sqlID']
        root_node = plan['sparkPlanInfo']['nodeName']
        print(f"SQL {sql_id}: Root operator = {root_node}")
```

#### JSON Result Helper Methods

```python
# Use to_dict() for JSON objects
cluster_json = (JSONReport(qual_core_handler)
    .table("coreRawClusterInformationJson")
    .app("app_001")
    .load())
if cluster_json.success:
    cluster_dict = cluster_json.to_dict()
    if cluster_dict:
        print(f"App name: {cluster_dict.get('appName')}")

# Use to_list() for JSON arrays
build_json = (JSONReport(qual_core_handler)
    .table("coreRawSparkRapidsBuildInfoJson")
    .app("app_001")
    .load())
if build_json.success:
    build_list = build_json.to_list()
    if build_list:
        print(f"Build info entries: {len(build_list)}")

# Access raw data (can be dict or list)
raw_json = (JSONReport(qual_wrapper_handler)
    .table("qualWrapperAppMetadataJson")
    .load())
if raw_json.success and raw_json.data:
    if isinstance(raw_json.data, dict):
        print("Data is a dictionary")
    elif isinstance(raw_json.data, list):
        print("Data is a list")
```

## Working with Applications

### Application Identifiers

The API supports both string identifiers and AppHandler objects:

```python
from spark_rapids_tools.api_v1 import AppHandler

# Using string identifiers
result1 = (CSVReport(handler)
    .table("per_app_table")
    .app("application_1234567890_0001")
    .load())

# Using AppHandler objects
app_handler = AppHandler("application_1234567890_0001")
result2 = (CSVReport(handler)
    .table("per_app_table")
    .app(app_handler)
    .load())

# Multiple applications with mixed types
results = (CSVReport(handler)
    .table("per_app_table")
    .app("app_001")
    .app(AppHandler("app_002"))
    .apps(["app_003", "app_004"])
    .load())
```

### Application Selection Patterns

```python
# Select all applications from a specific run
all_apps = ["application_001", "application_002", "application_003"]
all_results = (CSVReport(handler)
    .table("coreRawApplicationInformationCSV")
    .apps(all_apps)
    .load())

# Filter applications by pattern (example with filtering)
filtered_apps = [app for app in all_apps if app.endswith("_001")]
filtered_results = (CSVReport(handler)
    .table("coreRawJobInformationCSV")
    .apps(filtered_apps)
    .load())
```

## Return Types and Data Handling

### Global Tables

Global tables return single result objects:

```python
# CSV: Returns LoadDFResult
csv_result = CSVReport(handler).table("qualCoreCSVSummary").load()
dataframe = csv_result.data

# JSON: Returns JSONResult
json_result = (JSONReport(qual_wrapper_handler)
    .table("qualWrapperAppMetadataJson")
    .load())
json_data = json_result.to_dict()  # or to_list() for arrays

# Java Properties: Returns JPropsResult
props_result = JPropsReport(handler).table("runtimeProperties").load()
properties = props_result.props

# TXT: Returns TXTResult
txt_result = TXTReport(handler).table("coreRawSqlPlanPreAQEJson").app("app_001").load()
text_content = txt_result.text
```

### Per-Application Tables

Per-application tables have different return types based on the number of applications:

```python
# Multiple applications: Returns Dict[str, ResultType]
multi_results = (CSVReport(handler)
    .table("coreRawApplicationInformationCSV")
    .apps(["app_001", "app_002"])
    .load())

# Access individual results
app1_data = multi_results["app_001"].data
app2_data = multi_results["app_002"].data

# Single application: Returns ResultType directly (not a dictionary)
single_result = (CSVReport(handler)
    .table("coreRawJobInformationCSV")
    .app("app_001")
    .load())

# Direct access to data
single_app_data = single_result.data
```

## Error Handling

### Input Validation

The API provides comprehensive input validation:

```python
try:
    # Invalid output path
    handler = QualCore("")
except ValueError as e:
    print(f"Validation Error: {e}")

try:
    # Non-existent output path
    handler = QualCore("/non/existent/path")
    if handler.handler.is_empty():
        print("Warning: Handler is empty - no valid output found")
except Exception as e:
    print(f"Path Error: {e}")

try:
    # Invalid application configuration
    result = (CSVReport(handler)
        .table("qualCoreCSVSummary")  # This is a global table
        .app("app_001")  # Cannot specify apps for global tables
        .load())
except ValueError as e:
    print(f"Configuration Error: {e}")
```

### Data Loading Errors


If the data fails to be loaded or consumed, the error will be captured by the return object
and the success flag is set to false.

```python
try:
    result = (CSVReport(handler)
        .table("nonExistentTableLabel")
        .load())
    if not result.success:
        print(result.get_fail_cause())
except ValueError as e:
    print(f"Configuration Error: {e}")
```


## Complete Examples

### Qualification Analysis Workflow

```python
from spark_rapids_tools.api_v1 import QualCore, CSVReport, JPropsReport

# 1. Create qualification result handler
qual_handler = QualCore("/data/qualification/output")

# 2. Load qualification summary
summary = (CSVReport(qual_handler)
    .table("qualCoreCSVSummary")
    .load())

print(f"Analyzed {len(summary.data)} applications")

# 3. Load detailed SQL to stage information for specific apps
target_apps = ["application_001", "application_002"]
sql_analysis = (CSVReport(qual_handler)
    .table("coreRawSqlToStageInformationCSV")
    .apps(target_apps)
    .load())

# 4. Process results for each application
for app_id, result in sql_analysis.items():
    df = result.data
    print(f"App {app_id}: {len(df)} SQL operations analyzed")

# 5. Load runtime properties (tool version information)
runtime_props = (JPropsReport(qual_handler)
    .table("runtimeProperties")
    .load())

print(f"Tool version: {runtime_props.props.get('rapids.tools.version')}")
```

### Profiling Analysis Workflow

```python
from spark_rapids_tools.api_v1 import ProfCore, CSVReport, TXTReport

# 1. Create profiling result handler
prof_handler = ProfCore("s3://my-bucket/profiling/output")

# 2. Load application information
app_info = (CSVReport(prof_handler)
    .table("coreRawApplicationInformationCSV")
    .load())

print(f"Profiled applications: {len(app_info.data)}")

# 3. Load stage-level aggregated metrics for all applications
app_ids = app_info.data['App ID'].tolist()
stage_metrics = (CSVReport(prof_handler)
    .table("coreRawStageLevelAggregatedTaskMetricsCSV")
    .apps(app_ids)
    .load())

# 4. Analyze performance bottlenecks
for app_id, result in stage_metrics.items():
    stages_df = result.data

    # Find stages with most task time
    print(f"App {app_id} - Stage metrics summary:")
    print(f"  Total stages: {len(stages_df)}")
    print(f"  Avg executor CPU time: {stages_df['Executor CPU Time'].mean():.2f}ms")
```
### Tuning Reports

Tuning reports provide AutoTuner recommendations for GPU migration (requires `--auto-tuner` flag).
All tuning files are in Spark command-line format (`--conf key=value`):

```python
from spark_rapids_tools.api_v1 import QualCore

# Create qualification handler
handler = QualCore("/path/to/qual_core_output")

# Load bootstrap configuration (required GPU configs)
bootstrap_conf = handler.txt("tuningBootstrapConf").app("application_001").load()

if bootstrap_conf.success:
    print("Bootstrap configurations (Spark CLI format):")
    print(bootstrap_conf.data)
    # Output format:
    # --conf spark.rapids.sql.enabled=true
    # --conf spark.executor.cores=16
    # --conf spark.executor.memory=32g
    # ...

# Load detailed recommendations with comments
recommendations = handler.txt("tuningRecommendationsLog").app("application_001").load()

if recommendations.success:
    print("\nDetailed Recommendations:")
    print(recommendations.data)
    # Output includes:
    # Spark Properties:
    #   --conf ...
    # Comments:
    #   - Explanation of each setting...

# Load combined configuration (optional - only if generated)
combined_conf = handler.txt("tuningCombinedConf").app("application_001").load()

if combined_conf.success:
    print("\nCombined configurations:")
    print(combined_conf.data)
else:
    print("Combined config not available (optional file)")

# Load tuning for multiple applications
app_ids = ["application_001", "application_002", "application_003"]
bootstrap_configs = handler.txt("tuningBootstrapConf").apps(app_ids).load()

for app_id, config in bootstrap_configs.items():
    if config.success:
        # Count lines starting with --conf
        num_configs = len([l for l in config.data.split('\n') if l.startswith('--conf')])
        print(f"{app_id}: {num_configs} configurations")

# Parse configurations into dictionary
def parse_spark_conf(text):
    """Parse Spark CLI format into dict."""
    configs = {}
    for line in text.strip().split('\n'):
        if line.startswith('--conf '):
            conf = line[7:]  # Remove '--conf '
            if '=' in conf:
                key, value = conf.split('=', 1)
                configs[key] = value
    return configs

bootstrap = handler.txt("tuningBootstrapConf").app("application_001").load()
if bootstrap.success:
    configs_dict = parse_spark_conf(bootstrap.data)
    print(f"Parsed {len(configs_dict)} configurations:")
    for key, value in configs_dict.items():
        print(f"  {key} = {value}")
```

### Multi-Format Data Integration

```python
def analyze_application_performance(output_path, app_id):
    """Complete analysis combining multiple report formats."""

    # Create handlers for both qualification and profiling
    qual_handler = QualCore(f"{output_path}/qualification")
    prof_handler = ProfCore(f"{output_path}/profiling")

    results = {}

    # 1. Get qualification score
    try:
        qual_summary = (CSVReport(qual_handler)
            .table("qualCoreCSVSummary")
            .load())

        app_qual = qual_summary.data[
            qual_summary.data['App ID'] == app_id
        ]

        if not app_qual.empty:
            results['qualification_score'] = app_qual.iloc[0]['Estimated GPU Speedup']
    except Exception as e:
        print(f"Could not load qualification data: {e}")

    # 2. Get detailed SQL to stage information
    try:
        sql_ops = (CSVReport(qual_handler)
            .table("coreRawSqlToStageInformationCSV")
            .app(app_id)
            .load())

        results['total_sql_ops'] = len(sql_ops.data)
    except Exception as e:
        print(f"Could not load SQL operations: {e}")

    # 3. Get profiling metrics
    try:
        app_metrics = (CSVReport(prof_handler)
            .table("coreRawApplicationInformationCSV")
            .load())

        app_data = app_metrics.data[app_metrics.data['App ID'] == app_id]
        if not app_data.empty:
            results['duration_ms'] = app_data.iloc[0]['App Duration']
    except Exception as e:
        print(f"Could not load application metrics: {e}")

    # 4. Get Spark configuration
    try:
        config_csv = (CSVReport(prof_handler)
            .table("coreRawSparkPropertiesCSV")
            .app(app_id)
            .load())

        if config_csv.success:
            # Convert CSV rows to dict: {propertyName: propertyValue}
            results['spark_config'] = dict(zip(
                config_csv.data['propertyName'],
                config_csv.data['propertyValue']
            ))
    except Exception as e:
        print(f"Could not load configuration: {e}")

    return results

# Usage
app_analysis = analyze_application_performance(
    "/data/spark-rapids-analysis",
    "application_1234567890_0001"
)

print(f"Analysis Results: {app_analysis}")
```

## Best Practices

### 1-Input Validation and Error Handling

```python
def create_robust_handler(report_type, output_path):
    """Create a result handler with proper validation."""
    if not output_path:
        raise ValueError("Output path cannot be empty")

    try:
        if report_type == "qualification_core":
            return QualCore(output_path)
        elif report_type == "profiling_core":
            return ProfCore(output_path)
        elif report_type == "qualification_wrapper":
            return QualWrapper(output_path)
        elif report_type == "profiling_wrapper":
            return ProfWrapper(output_path)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    except Exception as e:
        print(f"Failed to create handler: {e}")
        raise
```

### 2-Efficient Data Loading

```python
# Use fallback callbacks for optional tables
def create_empty_dataframe():
    import pandas as pd
    return pd.DataFrame(columns=['App ID', 'Status'])

optional_data = (CSVReport(handler)
    .table("coreRawDataSourceInformationCSV")
    .fall_cb(create_empty_dataframe)
    .load())

# Optimize pandas reading for large files
large_data = (CSVReport(handler)
    .table("coreRawApplicationInformationCSV")
    .pd_args({
        "chunksize": 10000,      # Read in chunks
        "low_memory": False,     # Better type inference
        "dtype": {"App ID": str} # Explicit types
    })
    .load())
```

### 3-Application Management

```python
def get_application_list(handler):
    """Get list of available applications from summary table."""
    try:
        summary = (CSVReport(handler)
            .table("qualCoreCSVSummary")
            .load())
        return summary.data['App ID'].unique().tolist()
    except Exception:
        return []

def batch_process_applications(handler, table_name, app_list, batch_size=5):
    """Process applications in batches to manage memory."""
    results = {}

    for i in range(0, len(app_list), batch_size):
        batch = app_list[i:i + batch_size]

        try:
            batch_results = (CSVReport(handler)
                .table(table_name)
                .apps(batch)
                .load())

            results.update(batch_results)
        except Exception as e:
            print(f"Failed to process batch {i//batch_size + 1}: {e}")

    return results
```

### 4-Type Safety and Documentation

```python
from typing import Dict, List, Union, Optional
from spark_rapids_tools.api_v1.builder import LoadDFResult

def load_application_data(
    handler,
    table_name: str,
    apps: Union[str, List[str]],
    required: bool = True
) -> Optional[Union[LoadDFResult, Dict[str, LoadDFResult]]]:
    """
    Load CSV data for specified applications.

    Args:
        handler: Result handler instance
        table_name: Name of the table to load
        apps: Single app ID or list of app IDs
        required: Whether the table is required (affects error handling)

    Returns:
        LoadDFResult for single app, Dict[str, LoadDFResult] for multiple apps,
        or None if table not found and not required
    """
    try:
        report = CSVReport(handler).table(table_name)

        if isinstance(apps, str):
            return report.app(apps).load()
        else:
            return report.apps(apps).load()

    except FileNotFoundError:
        if required:
            raise
        return None
    except Exception as e:
        print(f"Error loading {table_name}: {e}")
        if required:
            raise
        return None
```


### 5-Resource Management

```python
# Use context managers for large datasets
class ReportAnalysis:
    def __init__(self, handler):
        self.handler = handler
        self._cached_data = {}

    def get_table_data(self, table_name, apps=None, use_cache=True):
        """Get table data with optional caching."""
        cache_key = f"{table_name}_{apps}"

        if use_cache and cache_key in self._cached_data:
            return self._cached_data[cache_key]

        report = CSVReport(self.handler).table(table_name)

        if apps:
            if isinstance(apps, str):
                result = report.app(apps).load()
            else:
                result = report.apps(apps).load()
        else:
            result = report.load()

        if use_cache:
            self._cached_data[cache_key] = result

        return result

    def clear_cache(self):
        """Clear cached data to free memory."""
        self._cached_data.clear()
```
