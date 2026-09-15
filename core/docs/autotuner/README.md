# AutoTuner heuristics

The AutoTuner combines recorded application data, source Spark and RAPIDS properties, prospective target-cluster configuration, platform information, and tuning policy to recommend target Spark properties. A missing recommendation can be intentional when a heuristic does not apply or cannot establish a required input safely.

These pages are code-adjacent developer documentation for understanding and maintaining individual heuristics. The [Qualification](https://docs.nvidia.com/spark-rapids/user-guide/latest/qualification/overview.html) and [Profiling](https://docs.nvidia.com/spark-rapids/user-guide/latest/profiling/overview.html) user guides remain the source for running the tools and interpreting their general output.

## Common concepts

- **Source value:** A property or metric recorded for the analyzed application.
- **Prospective target value:** The effective value after target enforcement, preservation, exclusion, caller controls, and ordinary recommendation logic are considered.
- **Evidence:** An observed event-log metric used by a heuristic. Evidence is distinct from a configured value or hardware assumption.
- **Compatibility model:** A version-aware model of runtime behavior that the tool can reproduce from known target inputs.
- **Fail closed:** Omit a dependent recommendation when a required input or compatibility model cannot be trusted instead of substituting an unrelated calculation.

## Heuristic page contract

Each page identifies the properties the heuristic recommends, the inputs it reads, the controls and caps it honors, the conditions that can suppress output, and the properties or conclusions it does not produce. Reading a property does not imply that the heuristic changes or recommends that property.

## Documented heuristics

| Heuristic | Recommends | Summary |
| --- | --- | --- |
| [GPU task concurrency](gpu-concurrency.md) | `spark.rapids.sql.concurrentGpuTasks` | Selects the legacy device-capacity calculation or a profiling evidence path that seeds GPU memory admission from the largest historical task footprint. |

This catalog is intentionally incremental and is not an exhaustive inventory of AutoTuner behavior.
