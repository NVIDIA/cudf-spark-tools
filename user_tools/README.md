# spark-rapids-user-tools

User tools to help with the adoption, installation, execution, and tuning of the NVIDIA cuDF plugin for Apache Spark.

The wrapper improves end-user experience within the following dimensions:
1. **Qualification**: Educate the CPU customer on the cost savings and acceleration potential of the cuDF plugin. The output shows a list of apps recommended for the cuDF plugin with estimated savings
   and speed-up.
2. **Tuning**: Tune the cuDF plugin configs based on initial job run leveraging Spark event logs. The output
   shows recommended per-app cuDF plugin configuration settings.
3. **Diagnostics**: Run diagnostic functions to validate the Dataproc environment with the cuDF plugin to
   make sure the cluster is healthy and ready for Spark jobs.
4. **Prediction**: Predict the speedup of running a Spark application with the cuDF plugin on GPUs.
5. **Train**: Train a model to predict the performance of a Spark job with the cuDF plugin. The output shows
   the model file that can be used to predict the performance of a Spark job.


## Getting started

Set up a Python environment with a version between 3.10 and 3.12

1. Run the project in a virtual environment. Note, .venv is the directory created to put
   the virtual env in, so modify if you want a different location.
    ```sh
    $ python -m venv .venv
    $ source .venv/bin/activate
    ```
2. Install spark-rapids-user-tools
    - Using released package.

      ```sh
      $ pip install spark-rapids-user-tools
      ```
    - Install from source.

      ```sh
      $ pip install -e .
      ```

      Note:
      - To install dependencies required for running unit tests, use the optional `test` parameter: `pip install -e '.[test]'`
      - To install dependencies required for QualX training, use the optional `qualx` parameter `pip install -e '.[qualx]'`
      - To install all the required dependencies, use the optional `dev-env` parameter: `pip install -e '.[dev-env]'`

    - Using wheel package built from the repo (see the build steps below).

      ```sh
      $ pip install <wheel-file>
      ```

3. Make sure to install CSP SDK if you plan to run the tool wrapper.

## Building from source

Set up a Python environment similar to the steps above.

1. Create a virtual environment. Note, .venv is the directory created to put
   the virtual env in, so modify if you want a different location.
    ```sh
    $ python -m venv .venv
    $ source .venv/bin/activate
    ```

2. Run the provided build script to compile the project.

   ```sh
   $> ./build.sh
   ```

3. **Fat Mode:** Similar to `fat jar` in Java, this mode solves the problem when web access is not
   available to download resources having Url-paths (http/https).
   The command builds the tools jar file and downloads the necessary dependencies and packages them
   with the source code into a single 'wheel' file.

   ```sh
   $> ./build.sh fat
   ```

## Logging Configuration

The core tools project uses Log4j for logging. Default log level is set to INFO.
You can configure logging settings in the `log4j.properties` file located in the
`src/spark_rapids_pytools/resources/dev/` directory. This is applicable when
you clone the project and build it from source.
To change the logging level, modify the `log4j.rootLogger` property.
Possible levels include `DEBUG`, `INFO`, `WARN`, `ERROR`.

## Usage and supported platforms

Please refer to [spark-rapids-user-tools guide](https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/docs/index.md) for details on how to use the tools
and the platform.

Please refer to [qualx guide](https://github.com/NVIDIA/cudf-spark-tools/blob/main/user_tools/docs/qualx.md) for details on how to use the QualX tool for prediction and training.

## What's new

Please refer to [CHANGELOG.md](https://github.com/NVIDIA/cudf-spark-tools/blob/main/CHANGELOG.md) for our latest changes.
