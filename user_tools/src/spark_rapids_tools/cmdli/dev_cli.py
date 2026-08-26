# Copyright (c) 2024-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI to run development related tools."""

from typing import Optional

import fire

from spark_rapids_pytools.common.utilities import ToolLogging
from spark_rapids_pytools.rapids.dev.instance_description import InstanceDescription
from spark_rapids_pytools.rapids.profiling_core import ProfilingCoreAsLocal
from spark_rapids_pytools.rapids.qualification_core import QualificationCoreAsLocal
from spark_rapids_pytools.rapids.qualification_stats import SparkQualStats
from spark_rapids_tools.cmdli.argprocessor import AbsToolUserArgModel
from spark_rapids_tools.enums import CspEnv
from spark_rapids_tools.utils.util import gen_app_banner, init_environment


class DevCLI(object):  # pylint: disable=too-few-public-methods
    """CLI to run development related tools and core tool variants."""

    def qualification_core(self,
                           eventlogs: str = None,
                           platform: str = None,
                           output_folder: str = None,
                           tools_jar: str = None,
                           jvm_heap_size: int = None,
                           jvm_threads: int = None,
                           tools_config_file: str = None,
                           verbose: bool = None,
                           **rapids_options) -> Optional[str]:
        """The Core Qualification cmd.

        :param eventlogs: Event log filenames, CSP storage directories containing event logs
                (comma separated), or path to a TXT file containing a list of event log paths.
        :param platform: Platform type: "onprem", "emr", "dataproc", "databricks-aws", "databricks-azure".
        :param output_folder: Local path to store the output.
        :param tools_jar: Path to a bundled jar including Rapids tool. If missing, downloads the latest
                rapids-4-spark-tools_*.jar from maven repository.
        :param jvm_heap_size: The maximum heap size of the JVM in gigabytes.
                Default is calculated based on a function of the total memory of the host.
        :param jvm_threads: Number of threads to use for parallel processing on the eventlogs batch.
                Default is calculated as a function of the total number of cores and the heap size on the host.
        :param tools_config_file: Path to a configuration file that contains the tools' options.
               For sample configuration files, please visit
               https://github.com/NVIDIA/cudf-spark-tools/tree/main/user_tools/tests/spark_rapids_tools_ut/resources/tools_config/valid
        :param verbose: True or False to enable verbosity of the script.
        :param rapids_options: A list of valid Qualification tool options.
        :return: The output folder where the qualification results are stored.
        """
        if verbose:
            ToolLogging.enable_debug_mode()
        session_uuid = init_environment('qual')

        qual_args = AbsToolUserArgModel.create_tool_args('qualification_core',
                                                         eventlogs=eventlogs,
                                                         platform=platform,
                                                         output_folder=output_folder,
                                                         tools_jar=tools_jar,
                                                         jvm_heap_size=jvm_heap_size,
                                                         jvm_threads=jvm_threads,
                                                         tools_config_path=tools_config_file,
                                                         session_uuid=session_uuid)
        if qual_args:
            rapids_options.update(qual_args.get('rapidOptions', {}))
            tool_obj = QualificationCoreAsLocal(platform_type=qual_args['runtimePlatform'],
                                                output_folder=qual_args['outputFolder'],
                                                wrapper_options=qual_args,
                                                rapids_options=rapids_options)
            tool_obj.launch()
            return tool_obj.csp_output_path
        return None

    def profiling_core(self,
                       eventlogs: str = None,
                       platform: str = None,
                       output_folder: str = None,
                       tools_jar: str = None,
                       jvm_heap_size: int = None,
                       jvm_threads: int = None,
                       tools_config_file: str = None,
                       verbose: bool = None,
                       **rapids_options) -> Optional[str]:
        """The Core Profiling cmd runs the profiling tool JAR directly with minimal processing.

        This is a simplified version for development and testing purposes that directly executes
        the profiling tool JAR without the extra processing layers.

        :param eventlogs: Event log filenames, cloud storage directories containing event logs
                (comma separated), or path to a TXT file containing a list of event log paths.
        :param platform: Platform type: "onprem", "emr", "dataproc", "databricks-aws", "databricks-azure".
        :param output_folder: Local path to store the output.
        :param tools_jar: Path to a bundled jar including Rapids tool. If missing, downloads the latest
                rapids-4-spark-tools_*.jar from maven repository.
        :param jvm_heap_size: The maximum heap size of the JVM in gigabytes.
                Default is calculated based on a function of the total memory of the host.
        :param jvm_threads: Number of threads to use for parallel processing on the eventlogs batch.
                Default is calculated as a function of the total number of cores and the heap size on the host.
        :param tools_config_file: Path to a configuration file that contains the tools' options.
               For sample configuration files, please visit
               https://github.com/NVIDIA/cudf-spark-tools/tree/main/user_tools/tests/spark_rapids_tools_ut/resources/tools_config/valid
        :param verbose: True or False to enable verbosity of the script.
        :param rapids_options: A list of valid Profiling tool options.
        :return: The output folder where the profiling results are stored.
        """
        if verbose:
            ToolLogging.enable_debug_mode()
        session_uuid = init_environment('prof')

        prof_args = AbsToolUserArgModel.create_tool_args('profiling_core',
                                                         eventlogs=eventlogs,
                                                         platform=platform,
                                                         output_folder=output_folder,
                                                         tools_jar=tools_jar,
                                                         jvm_heap_size=jvm_heap_size,
                                                         jvm_threads=jvm_threads,
                                                         tools_config_path=tools_config_file,
                                                         session_uuid=session_uuid)
        if prof_args:
            rapids_options.update(prof_args.get('rapidOptions', {}))
            tool_obj = ProfilingCoreAsLocal(platform_type=prof_args['runtimePlatform'],
                                            output_folder=prof_args['outputFolder'],
                                            wrapper_options=prof_args,
                                            rapids_options=rapids_options)
            tool_obj.launch()
            return tool_obj.csp_output_path
        return None

    def generate_instance_description(self,
                                      platform: str = None,
                                      output_folder: str = None):
        """The generate_instance_description cmd takes a platform and generates a json file with all the
        instance type descriptions for that CSP platform.

        :param platform: defines one of the following "dataproc", "emr", and "databricks-azure".
        :param output_folder: local path to store the output.
        """
        # Since this is an internal tool, we enable debug mode by default
        ToolLogging.enable_debug_mode()

        init_environment('generate_instance_description')

        instance_description_args = AbsToolUserArgModel.create_tool_args('generate_instance_description',
                                                                         cli_class='DevCLI',
                                                                         cli_name='spark_rapids_dev',
                                                                         target_platform=platform,
                                                                         output_folder=output_folder)
        if instance_description_args:
            tool_obj = InstanceDescription(platform_type=instance_description_args['targetPlatform'],
                                           output_folder=instance_description_args['output_folder'],
                                           wrapper_options=instance_description_args)
            tool_obj.launch()

    def stats(self,
              config_path: str = None,
              output_folder: str = None,
              qual_output: str = None):
        """The stats cmd generates statistics from the qualification tool output.

        Statistics is generated per AppId, per SQLID, and per Operator. For each operator, the
        statistics include the number of times the operator was executed, total task time of
        the stages that contain the operator, the total task time of the SQL that the operator
        is part of. The count of operators is also differentiated whether the operator is
        supported or unsupported.

        :param config_path: Path to the configuration file.
        :param output_folder: Path to store the output.
        :param qual_output: path to the directory, which contains the qualification tool output.
                            E.g. user should specify the parent directory $WORK_DIR where
                            $WORK_DIR/qual_core_output exists.
        """
        ToolLogging.enable_debug_mode()
        init_environment('stats')

        stats_args = AbsToolUserArgModel.create_tool_args('stats',
                                                          platform=CspEnv.get_default(),
                                                          config_path=config_path,
                                                          output_folder=output_folder,
                                                          qual_output=qual_output)
        tool_obj = SparkQualStats(platform_type=stats_args['runtimePlatform'],
                                  config_path=config_path,
                                  output_folder=output_folder,
                                  qual_output=qual_output,
                                  wrapper_options=stats_args)

        tool_obj.launch()


def main():
    # Make Python Fire not use a pager when it prints a help text
    fire.core.Display = lambda lines, out: out.write('\n'.join(lines) + '\n')
    print(gen_app_banner('Development'))
    fire.Fire(DevCLI())


if __name__ == '__main__':
    main()
