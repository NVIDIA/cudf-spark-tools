# Supported Ops Sync PR Checklist

<!--
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
-->

- Mention the plugin ref used for `tools/generated_files`.
- Confirm the PR branch starts from `dev` unless the user requested another base.
- Include the generated report summary for `supportedDataSource.csv`, `supportedExecs.csv`, and `supportedExprs.csv`.
- List new execs and expressions from `new_operators.txt`.
- Explain which new rows remain `TNEW` and why.
- Explain every `TNEW -> S`, `NS -> S`, or `CO -> S` promotion.
- For exec or datasource promotions, cite plugin code-path review and event-log/parser evidence.
- Call out config-gated, format-specific, datatype-specific, or Spark-version-specific support.
- Confirm operator score CSVs were updated when new operators were added.
- Confirm targeted tests run, or state why a test is not applicable.
- Call out Spark 4-only rows separately.
- Commit with `git commit -s`.
- Open a draft PR against `dev` with a title like `Sync supported ops with RAPIDS plugin as of YYYY-MM-DD`.
