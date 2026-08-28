# Event Log Runtime Detector

This package provides a lightweight Python detector for deciding which full
tools flow should handle a single Spark application event log.

The detector is an early routing check. It scans a bounded prefix of an event
log, stops as soon as it has enough information, and returns one of:

- `PROFILING`: a cuDF plugin runtime signal was found.
- `QUALIFICATION`: startup properties indicate standard OSS Spark with no
  cuDF plugin markers.
- `UNKNOWN`: the scan did not reach enough information within the event budget.

## Detection Flow

1. `resolver.py` resolves the input into ordered event-log files.
   Supported inputs are a single event-log file or an Apache Spark rolling
   event-log directory using the `eventlog_v2_*` / `events_*` layout.
2. `stream.py` opens each file through `CspPath.open_input_stream()` and yields
   one decoded event-log line at a time. The full log is not loaded into memory.
3. `scanner.py` parses events until a decision is available or the
   `max_events_scanned` budget is reached.
4. `classifier.py` classifies the accumulated Spark properties as `SPARK` or
   `SPARK_RAPIDS`.
5. `detector.py` maps the scan result to `ToolExecution`.

## cuDF plugin detection

cuDF plugin logs are detected from either of these signals:

- `SparkRapidsBuildInfoEvent`, emitted by cuDF plugin event logs.
- Spark properties showing `spark.plugins` contains
  `com.nvidia.spark.SQLPlugin` and `spark.rapids.sql.enabled` is not `false`.

The `spark.rapids.sql.enabled` parse matches the Scala tools behavior:
missing or unparseable values default to `true`.

## CPU Fast Path

When `SparkListenerEnvironmentUpdate` is reached and startup Spark properties
contain no cuDF plugin-related configuration, the detector can return
`QUALIFICATION` immediately. This applies to both single-file and OSS rolling
event logs.

If cuDF plugin-related configuration is present but not decisive, the scanner keeps
reading within the configured event budget. This avoids treating a log as CPU
when later `modifiedConfigs` may make the cuDF plugin configuration active.

## Streaming And Memory

The detector streams one line at a time. Memory is bounded to:

- a small set of runtime metadata fields,
- the accumulated Spark properties map,
- the current decoded event record.

It does not retain raw events or read entire event-log files into memory.
