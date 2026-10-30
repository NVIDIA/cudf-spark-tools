# Dynamic allocation heuristic

[AutoTuner heuristics](README.md)

The dynamic allocation heuristic recommends the executor counts of a GPU application that runs with `spark.dynamicAllocation.enabled=true`. It scales the source executor counts to the target executor shape and sets the number of executors the application starts with. Spark starts a dynamic allocation application with `max(minExecutors, initialExecutors, executor.instances)` executors, so the heuristic sets `initialExecutors` and `executor.instances` together.

## Contract

| Category | Behavior |
| --- | --- |
| Recommends | `spark.dynamicAllocation.minExecutors`, `spark.dynamicAllocation.initialExecutors`, `spark.dynamicAllocation.maxExecutors`, and `spark.executor.instances`. |
| Applies to | Cluster-level AutoTuner output when the effective `spark.dynamicAllocation.enabled` (enforced or source value) is `true`. When it is not `true`, the `minExecutors`, `initialExecutors`, and `maxExecutors` recommendations are removed and `spark.executor.instances` keeps the executor count of the recommended cluster. |
| Reads | The source executor counts, the source executor cores (`spark.executor.cores`, or the executor cores recorded in the event log when it is unset), the recommended target executor cores, the cluster sizing strategy, and target controls. |
| Honors | Enforced, preserved, and excluded target values. An enforced or preserved value is never lowered. |
| Does not recommend or change | `spark.executor.cores` or the executor count of the recommended cluster shape. |

## Scaling the source counts

The scale factor comes from the cluster sizing strategy, so the dynamic allocation counts follow the same rule as the recommended executor count.

| Strategy | Used by | Scale |
| --- | --- | --- |
| Constant total cores | Qualification | `source executor cores / target executor cores`. For example, 8 source cores and 16 target cores halve the counts. |
| Constant GPU count | Profiling on a CSP or with target worker info | 1.0, the executor counts are kept. |
| Source cores preserving | Profiling on-prem without target worker info | 1.0, the executor counts are kept. |

| Property | Rounding | Reason |
| --- | --- | --- |
| `initialExecutors`, `executor.instances`, `maxExecutors` | `max(1, ceil(value × scale))` | The GPU application can start with and grow to at least the source cores. |
| `minExecutors` | `max(1, floor(value × scale))` | The idle footprint stays within the source. |

- A source value of 0 is kept as 0, for example `minExecutors=0` to release all idle executors.
- `initialExecutors` and `executor.instances` are scaled only when the source sets them. The executor count of the recommended cluster shape is never used as a starting count.
- When the source or target executor cores are unknown, the source values are kept unscaled.

## Enforced `spark.executor.instances` is the target cluster size

Under dynamic allocation, a positive enforced `spark.executor.instances` is read as the size of the target cluster, for example workers × GPUs per worker. It is not read as the number of executors to start with. Keeping it would start the application at the full cluster size.

- It caps `spark.dynamicAllocation.maxExecutors`. If the source has no `maxExecutors`, it sets `maxExecutors`. A lower scaled source `maxExecutors` is kept, for example when the application shares the cluster. An enforced, preserved, or excluded `maxExecutors` is not changed.
- It is replaced in the output by the starting executor count (see below), and a comment records the enforced value and its replacement.

To enforce the number of executors the application starts with instead, enforce `spark.dynamicAllocation.initialExecutors` (or `minExecutors`). Those enforced values are kept and are part of the starting count.

Example: the source runs 8-core executors with `minExecutors=2` and `maxExecutors=40`, and the target enforces 16-core executors and `spark.executor.instances=8`.

| Property | Result | Why |
| --- | --- | --- |
| `maxExecutors` | 8 | Scaled to 20, then capped to the cluster size of 8. |
| `minExecutors` | 1 | `floor(2 × 0.5)`. |
| `initialExecutors` | 1 | The starting count. |
| `executor.instances` | 1 | The enforced 8 is replaced by the starting count. |

## Ordering invariants

Spark rejects `minExecutors > maxExecutors` and, on YARN and Kubernetes, a starting count above `maxExecutors`. After scaling, the heuristic applies these rules in order:

1. If an enforced or preserved `minExecutors`, `initialExecutors`, or `executor.instances` (other than an enforced cluster size) is above `maxExecutors`, `maxExecutors` is raised to it, since the protected value cannot be lowered.
2. `minExecutors` is capped to `maxExecutors`.
3. `initialExecutors` and `executor.instances` are kept within `[minExecutors, maxExecutors]`. A missing `minExecutors` counts as 0.

An excluded `minExecutors`, `initialExecutors`, `executor.instances`, or `maxExecutors` is not emitted, so its source value still applies and is not protected. The heuristic does not change it, but reports a conflict with the tuned values (an excluded value above the tuned `maxExecutors`, or an excluded `maxExecutors` below a tuned value) with the same comment as an unenforceable invariant.

Rule 3 does not lower an enforced or preserved value, and it does not raise one to `minExecutors`, since Spark starts with the largest of them rather than rejecting a lower value.

## Starting executor count

The starting count is the largest of:

- the recommended `minExecutors` and `initialExecutors`,
- the recommended `executor.instances`, unless it is an enforced cluster size,
- the source `minExecutors` scaled and rounded up, capped to `maxExecutors`, so the application starts with at least the source minimum cores,
- 0, the Spark default `minExecutors`.

`initialExecutors` and `executor.instances` are both set to the starting count unless they are enforced, preserved, or excluded. Setting them explicitly keeps a caller that merges the recommendations over the source configuration, for example a `spark-submit` with `--num-executors`, from falling back to the source CPU executor count.

| Control on `executor.instances` | Counted in the starting count as |
| --- | --- |
| None | The scaled source value, then replaced by the starting count. |
| Preserved | The unscaled source value. It can raise `maxExecutors` (rule 1). |
| Enforced, positive | Not counted; it is the target cluster size. |
| Excluded | Not counted, since it is not emitted. |

The recommended cluster is capped to `maxExecutors` and raised to the starting count if the workload sizing left it smaller, unless an enforced `executor.instances` gives the cluster size. The recommended cluster summary reports the tuned `minExecutors` and `maxExecutors` and the starting count as its initial executors. Properties without a recommendation keep their source value.

## Boundaries and limitations

- Every positive enforced `spark.executor.instances` is read as the cluster size under dynamic allocation, even when it was meant as a starting count. Enforce `initialExecutors` for a starting count.
- An enforced or preserved `minExecutors` or `initialExecutors` above the enforced cluster size raises `maxExecutors` above the cluster size. The application then asks for more executors than the cluster has.
- The downward shuffle partition pass sizes its slot count from `min(recommended cluster executors, maxExecutors)`. It does not use the starting count.

## Implementation map

| Component | Responsibility |
| --- | --- |
| [`AutoTuner`](../../src/main/scala/com/nvidia/spark/rapids/tool/tuning/AutoTuner.scala) | `recommendDynamicAllocationConfigs` scales the counts and applies the target cluster cap, `enforceDynamicAllocationInvariants` applies the ordering rules, `recommendStartingExecutorCount` sets the starting count, and `syncDynamicAllocationToRecommendedCluster` updates the cluster summary. |
| [`ClusterConfigurationStrategy`](../../src/main/scala/com/nvidia/spark/rapids/tool/ClusterConfigurationStrategy.scala) | `ClusterSizingStrategy.executorCountScale` owns the scale factor of each sizing strategy. |
| [`tuningTable.yaml`](../../src/main/resources/bootstrap/tuningTable.yaml) | Owns the public property descriptions and their cluster level. |
