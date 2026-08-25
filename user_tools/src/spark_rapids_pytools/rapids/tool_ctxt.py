# Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

"""Implementation of class holding the execution context of a rapids tool"""

import os
import re
import tarfile
from dataclasses import dataclass, field
from logging import Logger
from typing import Type, Any, ClassVar, List, Union

from spark_rapids_pytools.cloud_api.sp_types import PlatformBase
from spark_rapids_pytools.common.prop_manager import YAMLPropertiesContainer
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import ToolLogging, Utils
from spark_rapids_tools import CspEnv, CspPath
from spark_rapids_tools.enums import SubmissionMode
from spark_rapids_tools.utils import Utilities


@dataclass
class ToolContext(YAMLPropertiesContainer):
    """
    A container that holds properties and characteristics of a given execution.
    """
    name: str = None
    platform_cls: Type[PlatformBase] = None
    platform_opts: dict = field(default_factory=dict)
    logger: Logger = field(default=None, init=False)
    platform: PlatformBase = field(default=None, init=False)
    uuid: str = field(default=None, init=False)
    prepackage_paths: ClassVar[List[str]] = [
        Utils.resource_path('csp-resources.tgz'),
        Utils.resource_path('csp-resources')
    ]
    tools_resource_path: ClassVar[List[str]] = [
        Utils.resource_path('generated_files/core/jars')
    ]

    @classmethod
    def are_resources_prepackaged(cls) -> bool:
        return any(os.path.exists(f) for f in cls.prepackage_paths)

    def __connect_to_platform(self):
        self.logger.info('Start connecting to the platform')
        self.platform = self.platform_cls(ctxt_args=self.platform_opts)

    def __create_and_set_uuid(self):
        # The common sessionUuid is currently only set in ProfilingCore
        # and QualCore. Need to have the RUN_ID alignment for
        # all tools that use ToolContext
        if self.platform_opts.get('sessionUuid'):
            self.uuid = self.platform_opts['sessionUuid']
            return
        # If RUN_ID is provided (in init_environment), align uuid with it
        # RUN_ID is expected to be in the format <name>_<time>_<unique_id>
        # Safe access is needed in case of non-cli based context access that
        # do not trigger init_environment
        run_id = Utils.get_or_set_rapids_tools_env('RUN_ID')
        if isinstance(run_id, str) and run_id:
            parts = run_id.split('_')
            if len(parts) >= 3:
                self.uuid = '_'.join(parts[-2:])
                return
        # Default behavior
        self.uuid = Utils.gen_uuid_with_ts(suffix_len=8)

    def __create_and_set_cache_folder(self):
        # get the cache folder from environment variables or set it to default
        cache_folder = Utils.get_or_set_rapids_tools_env('CACHE_FOLDER', '/var/tmp/spark_rapids_user_tools_cache')
        # make sure the environment is set
        Utils.set_rapids_tools_env('CACHE_FOLDER', cache_folder)
        FSUtil.make_dirs(cache_folder)
        self.set_local('cacheFolder', cache_folder)

    def get_cache_folder(self) -> str:
        return self.get_local('cacheFolder')

    def _init_fields(self):
        self.logger = ToolLogging.get_and_setup_logger(f'rapids.tools.{self.name}.ctxt')
        self.__connect_to_platform()
        self.__create_and_set_uuid()
        self.props['localCtx'] = {}
        self.props['remoteCtx'] = {}
        self.props['wrapperCtx'] = {}
        # add a dictionary that holds all the rapids plugin args
        self.props['wrapperCtx']['rapidsArgs'] = {}
        # add a dictionary that holds arguments to be passed to the plugin args
        self.props['wrapperCtx']['jobArgs'] = {}
        # create cache_folder that will be used to hold large downloaded files
        self.__create_and_set_cache_folder()

    def get_deploy_mode(self) -> Any:
        return self.platform_opts.get('deployMode')

    def use_local_tools_jar(self) -> bool:
        return self.get_ctxt('useLocalToolsJar')

    def is_fat_wheel_mode(self) -> bool:
        return self.get_ctxt('fatWheelModeEnabled')

    def is_distributed_mode(self) -> bool:
        return self.get_ctxt('submissionMode') == SubmissionMode.DISTRIBUTED

    def is_local_mode(self) -> bool:
        return self.get_ctxt('submissionMode') == SubmissionMode.LOCAL

    def set_ctxt(self, key: str, val: Any):
        self.props['wrapperCtx'][key] = val

    def add_rapids_args(self, key: str, val: Any):
        self.props['wrapperCtx']['rapidsArgs'][key] = val

    def add_job_args(self, key: str, val: Any):
        self.props['wrapperCtx']['jobArgs'][key] = val

    def update_job_args(self, extra_args: dict):
        self.props['wrapperCtx']['jobArgs'].update(extra_args)

    def get_ctxt(self, key: str):
        return self.props['wrapperCtx'].get(key)

    def set_remote(self, key: str, val: Any):
        self.props['remoteCtx'][key] = val

    def set_local(self, key: str, val: Any):
        self.props['localCtx'][key] = val

    def get_local(self, key: str):
        return self.props['localCtx'].get(key)

    def get_remote(self, key: str):
        return self.props['remoteCtx'].get(key)

    def _set_local_dep_dir(self) -> None:
        """
        Create the dependency folder. It is a subdirectory of temp folders because we cannot
        store those on remote storage. Especially if this used for the classPath.
        The directory is going to be in 'tmp/run_name/work_dir'
        """
        cache_folder = self.get_cache_folder()
        exec_full_name = self.get_ctxt('execFullName')

        dep_folder_name = 'work_dir'
        self.set_ctxt('depFolderName', dep_folder_name)
        temp_folder = FSUtil.build_path(cache_folder, exec_full_name)
        self.set_local('tmpFolder', temp_folder)
        dep_folder = FSUtil.build_path(temp_folder, dep_folder_name)
        FSUtil.make_dirs(dep_folder, exist_ok=False)
        self.logger.info('Dependencies are generated locally in local disk as: %s', dep_folder)
        self.set_local('depFolder', dep_folder)

    def set_local_directories(self, output_parent_folder: str) -> None:
        """
        Creates and initializes local directories used for dependencies and output folder
        :param output_parent_folder: the directory where the local output is going to be created.
        """
        short_name = self.get_value('platform', 'shortName')
        # If RUN_ID is provided, use it verbatim to ensure exact match with logging RUN_ID
        run_id = Utils.get_or_set_rapids_tools_env('RUN_ID')
        exec_dir_name = run_id if run_id else f'{short_name}_{self.uuid}'
        # Ensure RUN_ID is set when absent (non-CLI usage); unify logs and folder names
        if not run_id:
            Utils.set_rapids_tools_env('RUN_ID', exec_dir_name)
        self.set_ctxt('execFullName', exec_dir_name)
        # create the local dependency folder
        self._set_local_dep_dir()
        # create the local output folder
        self.set_local_output_folder(output_parent_folder)

    def set_local_output_folder(self, parent: str) -> None:
        """
        create and initialized output folder on local disk. it will be as follows:
        parent/exec_full_name
        :param parent: the parent directory that will contain the subdirectory
        """
        exec_full_name = self.get_ctxt('execFullName')
        exec_root_dir = FSUtil.build_path(parent, exec_full_name)
        self.set_local('outputFolder', exec_root_dir)
        # For now set the cspOutputPath here
        self.set_csp_output_path(exec_root_dir)
        FSUtil.make_dirs(exec_root_dir, exist_ok=False)
        self.logger.info('Local output folder is set as: %s', exec_root_dir)

    def _identify_tools_wheel_jar(self, resource_files: List[str]) -> None:
        """
        Identifies the tools JAR file from resource files and sets its path in the context.
        :param resource_files: List of resource files to search for the tools JAR file.
        :raises AssertionError: If the number of matching files is not exactly one.
        """
        tools_jar_regex_str = self.get_value('sparkRapids', 'toolsJarRegex')
        tools_jar_regex = re.compile(tools_jar_regex_str)
        matched_files = [f for f in resource_files if tools_jar_regex.search(f)]
        assert len(matched_files) == 1, \
            (f'Expected exactly one tools JAR file, found {len(matched_files)}. '
             'Rebuild the wheel package with the correct tools JAR file.')
        # set the tools JAR file path in the context
        self.set_ctxt('useLocalToolsJar', True)
        self.set_ctxt('toolsJarFilePath', matched_files[0])

    def load_tools_jar_resources(self):
        """
        Checks for the tools jar and identifies it from the tools-resources directory.
        This method handles only the tools jar identification part without loading other prepackaged resources.
        """
        for tools_related_files in self.tools_resource_path:
            # This function uses a regex based comparison to identify the tools jar file
            # from the tools-resources directory. The jar is pre-packed in the wheel file
            # and will be copied to the work directory when the tool runs.
            self.logger.info('Checking for tools related files in %s', tools_related_files)
            if os.path.exists(tools_related_files):
                tools_files = FSUtil.get_all_files(tools_related_files)
                self._identify_tools_wheel_jar(tools_files)

    def load_prepackaged_resources(self):
        """
        Checks for the tools jar and adds it to context
        Checks if the packaging includes the CSP dependencies. If so, it moves the dependencies
        into the tmp folder. This allows the tool to pick the resources from cache folder.
        """
        self.load_tools_jar_resources()

        if not self.are_resources_prepackaged():
            self.logger.info('No prepackaged resources found.')
            return

        self.set_ctxt('fatWheelModeEnabled', True)
        self.logger.info(Utils.gen_str_header('Fat Wheel Mode Is Enabled',
                                              ruler='_', line_width=50))

        for res_path in self.prepackage_paths:
            if os.path.exists(res_path):
                if os.path.isdir(res_path):
                    # this is a directory, copy all the contents to the tmp
                    FSUtil.copy_resource(res_path, self.get_cache_folder())
                else:
                    # this is an archived file
                    with tarfile.open(res_path, mode='r:*') as tar_file:
                        tar_file.extractall(self.get_cache_folder())
                        tar_file.close()

    def get_output_folder(self) -> str:
        return self.get_local('outputFolder')

    def get_wrapper_summary_file_path(self) -> str:
        summary_file_name = self.get_value('local', 'output', 'fileName')
        summary_path = FSUtil.build_path(self.get_csp_output_path(), summary_file_name)
        return summary_path

    def get_local_work_dir(self) -> str:
        return self.get_local('depFolder')

    def get_rapids_jar_url(self) -> str:
        self.logger.info('Fetching the Rapids Jar URL from local context')
        if self.use_local_tools_jar():
            return self._get_tools_jar_from_local()
        self.logger.info('Tools JAR not found in local context. Downloading from Maven.')
        mvn_base_url = Utilities.resolve_maven_url(self.get_value('sparkRapids', 'mvnUrl'))
        jar_version = Utilities.get_latest_mvn_jar_from_metadata(mvn_base_url)
        rapids_url = self.get_value('sparkRapids', 'repoUrl').format(mvn_base_url, jar_version, jar_version)
        return rapids_url

    def get_tool_main_class(self) -> str:
        return self.get_value('sparkRapids', 'mainClass')

    def get_rapids_auto_tuner_enabled(self) -> bool:
        return self.get_value('sparkRapids', 'enableAutoTuner')

    def requires_eventlogs(self) -> bool:
        flag = self.get_value_silent('sparkRapids', 'requireEventLogs')
        if flag is None:
            return True
        return flag

    def get_rapids_output_folder(self) -> str:
        # TODO: Remove the subfolder entry from here as it is not relevant
        #       in the new output folder structure for Qualification
        root_dir = self.get_csp_output_path()
        rapids_subfolder = self.get_value_silent('toolOutput', 'subFolder')
        if rapids_subfolder is None:
            return root_dir
        return FSUtil.build_path(root_dir, rapids_subfolder)

    def get_metrics_output_folder(self) -> CspPath:
        root_dir = CspPath(self.get_rapids_output_folder())
        metrics_subfolder = self.get_value('toolOutput', 'metricsSubFolder')
        return root_dir.create_sub_path(metrics_subfolder)

    def get_log4j_properties_file(self) -> str:
        return self.get_value_silent('toolOutput', 'textFormat', 'log4jFileName')

    def get_platform_name(self) -> str:
        """
        This used to get the lower case of the platform of the runtime.
        :return: the name of the platform of the runtime in lower_case.
        """
        return CspEnv.pretty_print(self.platform.type_id)

    def _get_tools_jar_from_local(self) -> str:
        """
        Extracts the tools JAR file from the context and returns its path from the resources directory.
        """
        jar_filepath = self.get_ctxt('toolsJarFilePath')
        if jar_filepath is None:
            raise ValueError(
                'Tools JAR file path not found in context. '
                'Make sure the tools JAR is included in the package.'
            )
        if not FSUtil.resource_exists(jar_filepath):
            raise FileNotFoundError(
                f'Tools JAR not found at path: {jar_filepath}. '
                'Rebuild the wheel package'
            )
        self.logger.info('Using jar from wheel file %s', jar_filepath)
        return jar_filepath

    def do_cleanup_tmp_directory(self) -> bool:
        """
        checks whether the temp folder created for the run should be deleted at the end or not.
        :return: True/False if the temp folders hsould be cleaned up
        """
        config_val = self.get_value_silent('platform', 'cleanUp')
        return Utilities.string_to_bool(config_val)

    def get_local_tmp_folder(self) -> str:
        return self.get_local('tmpFolder')

    def set_csp_output_path(self, path: Union[str, CspPath]) -> None:
        csp_path = CspPath(path)
        self.set_ctxt('cspOutputPath', csp_path)

    def get_csp_output_path(self) -> str:
        return self.get_output_folder()
