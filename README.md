# NVIDIA cuDF plugin for Apache Spark Tools

This repo provides tools for the [NVIDIA cuDF plugin for Apache Spark](https://github.com/NVIDIA/cudf-spark).

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/cudf-spark-tools)

## Catalog

- [cuDF plugin core tools](./core): Tools that help developers getting the most out of their Apache
  Spark applications
  without any code change:
  - Report acceleration potential of the cuDF plugin on a set of Spark applications.
  - Generate comprehensive profiling analysis for Apache Sparks executing on accelerated GPU instances. This information
    can be used to further tune and optimize the application.
- [spark-rapids-user-tools](./user_tools): A simple wrapper process around cloud service
  providers to run
  [cuDF plugin core tools](./core) across multiple cloud platforms. In addition, the output educates
  the users on
  the cost savings and acceleration potential of the cuDF plugin and makes recommendations to tune
  the application performance based on the cluster shape.
