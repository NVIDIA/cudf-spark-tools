# Dynamic allocation heuristic

[AutoTuner heuristics](README.md)

The dynamic allocation heuristic recommends the executor counts of a GPU application that runs with `spark.dynamicAllocation.enabled=true`. It scales the source executor counts to the target executor shape and sets the number of executors the application starts with. Spark starts a dynamic allocation application with `max(minExecutors, initialExecutors, executor.instances)` executors, so the heuristic sets `initialExecutors` and `executor.instances` together.

## Contract

| Category | Behavior |
| --- | --- |
| Recommends | `spark.dynamicAllocation.minExecutors`, `spark.dynamicAllocation.initialExecutors`, `spark.dynamicAllocation.maxExecutors`, and `spark.executor.instances`. |
| Applies to | Cluster-level AutoTuner output when the effective `spark.dynamicAllocation.enabled` (enforced or source value) is `true`. When it is not `true`, the `minExecutors`, `initialExecutors`, and `maxExecutors` recommendations are removed, including enforced ones, and `spark.executor.instances` keeps the executor count of the recommended cluster. |
| Reads | The source executor counts, the source executor cores (`spark.executor.cores`, or the executor cores recorded in the event log when it is unset), the recommended target executor cores, the cluster sizing strategy, and target controls. |
| Honors | Enforced, preserved, and excluded target values. An enforced or preserved value is never lowered, except an enforced `spark.executor.instances`, which is the size of the target cluster and is replaced by the starting count (see [Enforced `spark.executor.instances`](#enforced-sparkexecutorinstances)). |
| Changes | The executor count and worker node count of the recommended cluster, which is capped to `maxExecutors` and raised to the starting count, unless `spark.executor.instances` is enforced. |
| Does not recommend or change | `spark.executor.cores`. |

## Scaling the source counts

The scale factor comes from the cluster sizing strategy (`CLUSTER_SIZING_STRATEGY` in the tuning configs), so the dynamic allocation counts follow the same rule as the recommended executor count. The defaults with `auto` are listed below; an explicit strategy applies to both tools.

| Strategy | Default for | Scale |
| --- | --- | --- |
| Constant total cores | Qualification on a CSP or with target worker info | `source executor cores / target executor cores`. For example, 8 source cores and 16 target cores halve the counts. |
| Constant GPU count | Profiling on a CSP or with target worker info | 1.0, the executor counts are kept. |
| Source cores preserving | Both tools on-prem without target worker info | 1.0, the executor counts are kept. |

| Property | Rounding | Reason |
| --- | --- | --- |
| `initialExecutors`, `executor.instances`, `maxExecutors` | `max(1, ceil(value × scale))` | The GPU application can start with and grow to at least the source cores. |
| `minExecutors` | `max(1, floor(value × scale))` | The idle footprint stays within the source. |

- A source value of 0 is kept as 0, for example `minExecutors=0` to release all idle executors.
- `initialExecutors` and `executor.instances` are scaled only when the source sets them. The executor count of the recommended cluster shape is never used as a starting count.
- When the source or target executor cores are unknown, the source values are kept unscaled.

## Enforced `spark.executor.instances`

Under dynamic allocation, a positive enforced `spark.executor.instances` describes the size of the target cluster rather than an executor count to start with. Keeping it would start the application at the full cluster size, since Spark starts with `max(minExecutors, initialExecutors, executor.instances)`. So the heuristic:

- caps `maxExecutors` to the size of the target cluster, keeping a lower scaled source value (for example to share the cluster), or sets it when the source has no `maxExecutors`. If an enforced or preserved `minExecutors` or `initialExecutors` is above the size of the target cluster, the cap is that value instead, since Spark starts with the largest of them and rejects a starting count above `maxExecutors`. The comment then reports that final ceiling. An enforced, preserved, or excluded `maxExecutors` is not changed;
- leaves it out of the starting count and replaces it with the starting count, with a comment in place of the enforced-property comment. It is not replaced when it already equals the starting count. When the source sets no `minExecutors` or `initialExecutors`, the starting count is 0, so it is replaced with 0 and the application starts with no executors and scales up;
- keeps the recommended cluster at the enforced size, without capping it to `maxExecutors` or raising it to the starting count.

Example: the source runs 8-core executors with `minExecutors=2` and `maxExecutors=40`, and the target enforces 16-core executors and `spark.executor.instances=8`.

| Property | Result | Why |
| --- | --- | --- |
| `maxExecutors` | 8 | Scaled to 20, then capped to the target cluster size of 8. |
| `minExecutors` | 1 | `floor(2 × 0.5)`. |
| `initialExecutors` | 1 | The starting count, `max(1, ceil(2 × 0.5))`. |
| `executor.instances` | 1 | The enforced 8 replaced by the starting count. |

## Ordering invariants

Spark rejects `minExecutors > maxExecutors` and, on YARN and Kubernetes, a starting count above `maxExecutors`. After scaling, the heuristic applies these rules in order:

1. If an enforced or preserved `minExecutors`, `initialExecutors`, or `executor.instances` is above `maxExecutors`, `maxExecutors` is raised to it, since the protected value cannot be lowered. An enforced `executor.instances` that is the size of the target cluster is not protected, since it is replaced by the starting count.
2. `minExecutors` is capped to `maxExecutors`.
3. `initialExecutors` and `executor.instances` (unless it is the enforced size of the target cluster) are kept within `[minExecutors, maxExecutors]`. A missing `minExecutors` counts as 0.

An excluded `minExecutors`, `initialExecutors`, `executor.instances`, or `maxExecutors` is not emitted, so its source value still applies and is not protected. The heuristic does not change it, but reports a conflict with the tuned values (an excluded value above the tuned `maxExecutors`, or an excluded `maxExecutors` below the tuned starting count) with the same comment as an unenforceable invariant. The conflict is checked after the starting executor count is set, so it covers the final `initialExecutors` and `executor.instances`.

Rule 3 does not lower an enforced or preserved value, and it does not raise one to `minExecutors`, since Spark starts with the largest of them rather than rejecting a lower value. A violation that only involves values that cannot be changed, for example an enforced `maxExecutors` below an enforced `minExecutors`, is reported in a comment instead.

## Starting executor count

The starting count is the largest of:

- the recommended `minExecutors` and `initialExecutors`,
- the recommended `executor.instances`, unless it is the enforced size of the target cluster,
- the source `minExecutors` scaled and rounded up, capped to `maxExecutors` (the recommended value, or the source value of an excluded `maxExecutors`, which still applies), so the application starts with at least the source minimum cores,
- 0, the Spark default `minExecutors`.

`initialExecutors` and `executor.instances` are both set to the starting count unless they are enforced, preserved, or excluded. An enforced `executor.instances` is replaced as described in [Enforced `spark.executor.instances`](#enforced-sparkexecutorinstances). Setting them explicitly keeps a caller that merges the recommendations over the source configuration, for example a `spark-submit` with `--num-executors`, from falling back to the source CPU executor count.

| Control on `executor.instances` | Counted in the starting count as |
| --- | --- |
| None | The scaled source value, then replaced by the starting count. |
| Preserved | The unscaled source value. It can raise `maxExecutors` (rule 1). |
| Enforced, positive | Not counted. It caps `maxExecutors` and is replaced by the starting count. |
| Excluded | Not counted, since it is not emitted. |

Unless `executor.instances` is enforced, the recommended cluster is capped to `maxExecutors` and raised to the starting count if the workload sizing left it smaller. The recommended cluster summary reports the tuned `minExecutors` and `maxExecutors` and the starting count as its initial executors. Properties without a recommendation keep their source value.

## Boundaries and limitations

- An enforced `spark.executor.instances` only caps `maxExecutors`. A scaled source `maxExecutors` below it is kept. Enforce `maxExecutors` as well to use the whole target cluster.
- An enforced or preserved `minExecutors` or `initialExecutors` above the enforced `executor.instances` raises `maxExecutors` above the size of the target cluster, and the cluster summary still reports the enforced size.
- The downward shuffle partition pass sizes its slot count from `min(recommended cluster executors, maxExecutors)`. It does not use the starting count.
- When `spark.dynamicAllocation.enabled` is not `true`, enforced `minExecutors`, `initialExecutors`, and `maxExecutors` are removed from the output, although their enforced-property comments remain. Enforce `spark.dynamicAllocation.enabled=true` as well to keep them.

## Implementation map

| Component | Responsibility |
| --- | --- |
| [`AutoTuner`](../../src/main/scala/com/nvidia/spark/rapids/tool/tuning/AutoTuner.scala) | `recommendDynamicAllocationConfigs` scales the counts and caps `maxExecutors` to an enforced `executor.instances` (`targetClusterExecutors`), `enforceDynamicAllocationInvariants` applies the ordering rules, `recommendStartingExecutorCount` sets the starting count and replaces an enforced `executor.instances` with it, and `syncDynamicAllocationToRecommendedCluster` updates the cluster summary. |
| [`ClusterConfigurationStrategy`](../../src/main/scala/com/nvidia/spark/rapids/tool/ClusterConfigurationStrategy.scala) | `ClusterSizingStrategy.executorCountScale` owns the scale factor of each sizing strategy. |
| [`tuningTable.yaml`](../../src/main/resources/bootstrap/tuningTable.yaml) | Owns the public property descriptions and their cluster level. |
