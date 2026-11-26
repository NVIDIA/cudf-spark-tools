# Target cluster file reference

The target cluster file is a YAML file passed to the Profiling and Qualification tools to describe the cluster that the AutoTuner should tune for. This page lists every key the file accepts, derived from the classes that load it.

## Passing the file

| Entry point | Option |
| --- | --- |
| Core JAR (Profiling and Qualification) | `--target-cluster-info <path>` |
| Python CLI (`spark_rapids`) | `--target_cluster_info <path>` |

The option is optional. In the core JAR, if the file cannot be opened (an `IOException`), the tool logs a warning and continues without target cluster information. The Python CLI checks the path before launching the JAR and fails with a validation error if it is not an existing `.yaml` or `.yml` file. The core JAR parses the file with SnakeYAML into `TargetClusterProps` and validates it right after loading; a validation failure throws `IllegalArgumentException`.

When the Python CLI finds a non-empty `workerInfo` in the file, it records that worker information was provided (`targetWorkerInfoProvided`). For the Qualification Tool, this adds a warning to the console output that speedup estimates may be inaccurate; it does not skip the speedup calculation.

Example files are in [`core/docs/sampleFiles/targetClusterInfo/`](../sampleFiles/targetClusterInfo/).

## Schema

All top-level sections are optional. Omitted sections behave as empty.

```yaml
driverInfo:
  instanceType: <string>
workerInfo:
  instanceType: <string>
  cpuCores: <int>
  memoryGB: <long>
  gpu:
    name: <string>
    count: <int>
    memory: <string>
sparkProperties:
  enforced:
    <spark property>: <value>
  preserve:
    - <spark property>
  exclude:
    - <spark property>
  tuningDefinitions:
    - label: <spark property>
      ...
```

### `driverInfo`

| Key | Type | Description |
| --- | --- | --- |
| `instanceType` | string | Driver instance type of the target cluster (for example `n1-standard-8` on Dataproc). |

`driverInfo` cannot be combined with an on-prem `workerInfo` (see Validation rules).

### `workerInfo`

A worker is described in one of two ways: by a cloud instance type, or by on-prem resources.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `instanceType` | string | empty | Cloud instance type (for example `g2-standard-24`). Setting it makes the worker a cloud-instance description. |
| `cpuCores` | int | `0` | CPU cores per worker. On-prem only. |
| `memoryGB` | long | `0` | Memory per worker in GB. On-prem only. |
| `gpu` | map | all unset | GPU settings, see below. |

A worker counts as on-prem when `cpuCores` or `memoryGB` is greater than 0. When `instanceType` is set and a GPU count is also set, the pair (instance type, GPU count) is used to look up the instance in the platform's instance table; with no GPU count only the instance type is used. If the instance type (or the type and GPU count pair) is not in the platform's table, the tool throws `MatchingInstanceTypeNotFoundException`.

#### `workerInfo.gpu`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | empty | GPU model. Stored in lower case, so `L4` and `l4` are equivalent. Supported values: `a100`, `t4`, `v100`, `k80`, `p100`, `p4`, `l4`, `l20`, `a10`, `a10g`. For an on-prem worker the name is what selects the GPU device; without it no target worker instance is built from `workerInfo`. For a cloud worker the GPU comes from the instance type. |
| `count` | int | `0` | GPUs per worker. `0` means not set: a cloud instance is then looked up by instance type alone, and an on-prem worker is built with 0 GPUs, so set it for on-prem workers. |
| `memory` | string | `0m` | Total memory of one GPU, for example `24g`. `0m`, an empty string, or omitting the key means "not set", and the memory of the resolved GPU device is used instead. Any other value is passed on unchanged and is checked later by the code that uses it. |

### `sparkProperties`

| Key | Type | Description |
| --- | --- | --- |
| `enforced` | map of property to value | Values that override the AutoTuner's recommendation. Each enforced property is added to the tuning table and to the recommendations, and the output comment says it was user-enforced. |
| `preserve` | list of properties | Properties whose value is kept from the source application. A property that is missing from the source gets a comment saying so. |
| `exclude` | list of properties | Properties that are removed from the tuning table and so are not recommended. |
| `tuningDefinitions` | list of definitions | Extra tuning table entries that are merged with the default table (`core/src/main/resources/bootstrap/tuningTable.yaml`); an entry with the same `label` as a default entry replaces it. Only entries with `enabled: true` are used. |

#### `sparkProperties.tuningDefinitions[]`

Each item has the following keys. The names match `TuningEntryDefinition`.

| Key | Type | Description |
| --- | --- | --- |
| `label` | string | The Spark property name. This is the key the entry is merged under. |
| `description` | string | Why the property matters and how it is used. |
| `enabled` | boolean | Defaults to `true`. Entries with `false` are ignored. |
| `level` | string | `job` (application-level settings) or `cluster` (executor or resource-level settings). Case-insensitive. |
| `category` | string | `functionality` (needed for the plugin to work), `tuning` (needed to tune the runtime), or `multiThreadReadCoreMultiplier` (the property used as the core multiplier when tuning `spark.rapids.sql.multiThreadedRead.numThreads`). Case-insensitive. |
| `confType.name` | string | Value type: `int`, `long`, `double`, `boolean`, `string`, `time` or `byte`. Case-insensitive. |
| `confType.defaultUnit` | string | Optional default unit, for example `MiB` for a `byte` property. |
| `bootstrapEntry` | boolean | Defaults to `true`. Properties whose label starts with `spark.rapids.` are always treated as bootstrap entries. |
| `defaultSpark` | string | Spark's default value, used as the original value when the event log does not set the property. |
| `modifiedBy` | string | Comma-separated list of plugin rules that set the property. |
| `specialValues` | list of strings | Values that skip type-specific normalization and formatting. |
| `comments` | map | Optional comment texts: `missing`, `missingPolicy` (`default` or `omit`), `persistent`, `updated`. |

The no-argument constructor leaves `level`, `category` and `confType` empty, and the parsers reject empty values with `Unknown level`, `Unknown category` and `Unable to create ConfType without name`. These checks are lazy, so give all three in every definition rather than relying on a default. This was read from the source and not run.

## Validation rules

These are the checks in `TargetClusterProps.validate()` and the classes it calls:

- `workerInfo.instanceType` and the on-prem values (`cpuCores`, `memoryGB`) cannot both be set.
- An on-prem worker must also have GPU information (`gpu` must not be empty).
- `gpu.name`, when set, must be one of the supported names above.
- When `cpuCores` is set, `gpu.count` cannot be larger than `cpuCores`.
- An on-prem `workerInfo` cannot be combined with a non-empty `driverInfo`.
- A property cannot appear in more than one of `preserve`, `exclude` and the keys of `enforced`.

## Examples

| File | Shows |
| --- | --- |
| `targetCluster01.yaml` | Cloud `workerInfo.instanceType` only. |
| `targetCluster02.yaml` | Cloud worker plus `sparkProperties.enforced`. |
| `targetCluster03.yaml` | On-prem worker with `cpuCores`, `memoryGB` and `gpu`. |
| `targetCluster04.yaml` | On-prem worker plus `enforced`. |
| `targetCluster05.yaml` | `driverInfo` and `workerInfo` instance types. |
| `targetCluster06.yaml` | `tuningDefinitions`. |
| `targetCluster07.yaml` | `preserve` and `exclude`. |
