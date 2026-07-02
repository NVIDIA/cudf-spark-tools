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

"""
This module defines the logic to fetch all dependencies and resources required to run the user-tools
without need to access the web during runtime.
"""

import os
import tarfile
from typing import Optional

import fire

from spark_rapids_pytools.common.prop_manager import JSONPropertiesContainer
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.rapids.rapids_tool import RapidsTool
from spark_rapids_tools import CspEnv, CspPath
from spark_rapids_tools.configuration.common import RuntimeDependency
from spark_rapids_tools.utils import Utilities
from spark_rapids_tools.utils.net_utils import DownloadManager, DownloadTask

# Defines the constants and static configurations
prepackage_conf = {
    '_supported_platforms': [csp.value for csp in CspEnv if csp != CspEnv.NONE],
    '_configs_suffix': '-configs.json',
    '_mvn_base_url': 'https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark-tools_2.12',
    '_folder_name': 'csp-resources',
    # define relative path of the core jars
    '_tools_folder_name': 'generated_files/core/jars'
}


class PrepackageMgr:   # pylint: disable=too-few-public-methods
    """ Class that handles downloading dependencies to be pre-packaged ahead of runtime.

    The script can be triggered by passing the arguments and the properties of the class
    For example:

    $> python prepackage_mgr.py --resource_dir=RESOURCE_DIR --tools_jar=TOOLS_JAR run
    $> python prepackage_mgr.py run --resource_dir=RESOURCE_DIR --tools_jar=TOOLS_JAR

    For more information:

    $> python prepackage_mgr.py --help

    :param resource_dir: Root folder where the configuration files are located.
    :param dest_dir: Directory in which the resources are downloaded and stored.
           If missing, the tool creates a subfolder '$resource_dir/csp-resources'.
           Warning: the 'dest_dir' may be deleted when 'archive_enabled' is set to True.
    :param tools_jar: Path of the spark-rapids-user-tools jar file. Typically, this is the snapshot
           jar file of the current build.
    :param archive_enabled: A flag to enable/disable compressing the resources.
           Note that a wheel-package with compressed prepackaged resources is 30% less size
           compared to the non-compressed one (~ 600 Vs. 900 MB).
           When enabled, the prepackaged-resources are stored in '$resource_dir/csp-resources.tgz'.
           If the 'dest_dir' is provided, then the output is stored as '$dest_dir/../csp-resources.tgz'
    """

    def __init__(self,
                 resource_dir: str,
                 dest_dir: str = None,
                 tools_jar: str = None,
                 archive_enabled: bool = True,
                 fetch_all_csp: bool = False):
        for field_name in prepackage_conf:
            setattr(self, field_name, prepackage_conf.get(field_name))
        self.resource_dir = resource_dir
        self.dest_dir = dest_dir
        self.tools_jar = tools_jar
        self.archive_enabled = archive_enabled
        self.fetch_all_csp = fetch_all_csp
        print(f'Resource directory is: {self.resource_dir}')
        print(f'tools_jar = {tools_jar}')
        self.resource_dir = FSUtil.get_abs_path(self.resource_dir)
        self.tools_resources_dir = FSUtil.build_full_path(self.resource_dir, self._tools_folder_name)  # pylint: disable=no-member
        if self.dest_dir is None:
            self.dest_dir = FSUtil.build_full_path(self.resource_dir, self._folder_name)  # pylint: disable=no-member
        else:
            self.dest_dir = FSUtil.get_abs_path(self.dest_dir)

    def _get_spark_rapids_jar_url(self) -> str:
        mvn_base_url = Utilities.resolve_maven_url(self._mvn_base_url)  # pylint: disable=no-member
        jar_version = Utilities.get_latest_mvn_jar_from_metadata(mvn_base_url)
        return (f'{mvn_base_url}/'
                f'{jar_version}/rapids-4-spark-tools_2.12-{jar_version}.jar')

    def _fetch_resources(self) -> dict:
        """
        Fetches the resource information from configuration files for each supported platform.
        Tools jar if passed explicitly is set as a dependency. Else it is build from source
        and added as a dependency.
        Returns a dictionary of resource details.
        """
        resource_uris = {}

        # Add RAPIDS JAR as dependency
        if self.tools_jar:
            # copy from existing file.
            tools_jar_cspath = CspPath(self.tools_jar)
            tools_jar_url = str(tools_jar_cspath)
            jar_file_name = tools_jar_cspath.base_name()
            print(f'Using the provided tools_jar {tools_jar_url}')
        else:
            tools_jar_url = self._get_spark_rapids_jar_url()
            jar_file_name = FSUtil.get_resource_name(tools_jar_url)
        resource_uris[tools_jar_url] = {
            'depItem': RuntimeDependency(name=jar_file_name, uri=tools_jar_url),
            'prettyName': jar_file_name,
            'isToolsResource': True
        }

        if self.fetch_all_csp:
            for platform in self._supported_platforms:  # pylint: disable=no-member
                config_file = FSUtil.build_full_path(self.resource_dir,
                                                     f'{platform}{self._configs_suffix}')  # pylint: disable=no-member
                platform_conf = JSONPropertiesContainer(config_file)
                dependency_list = RapidsTool.get_rapids_tools_dependencies('LOCAL', platform_conf)
                for dependency in dependency_list:
                    if dependency.uri:
                        uri_str = str(dependency.uri)
                        pretty_name = FSUtil.get_resource_name(uri_str)
                        resource_uris[uri_str] = {
                            'depItem': dependency,
                            'prettyName': pretty_name,
                            'isToolsResource': False
                        }
        else:
            print('Skipping fetching all CSP resources')
        return resource_uris

    def _download_resources(self, resource_uris: dict):
        download_tasks = []
        for res_uri, res_info in resource_uris.items():
            resource_name = res_info.get('prettyName')
            is_tools_resource = res_info.get('isToolsResource')
            dest_folder = self.tools_resources_dir if is_tools_resource else self.dest_dir
            print(f'Creating download task: {resource_name}')
            # All the downloadTasks enforces download
            download_tasks.append(DownloadTask(src_url=Utilities.resolve_maven_url(res_uri),  # pylint: disable=no-value-for-parameter
                                               dest_folder=dest_folder,
                                               configs={'forceDownload': True}))
        # Begin downloading the resources
        download_results = DownloadManager(download_tasks, max_workers=12).submit()
        print('----Download summary---')
        for res in download_results:
            print(res.pretty_print())
        print('-----------------------')

    def _compress_resources(self) -> Optional[str]:
        if not self.archive_enabled:
            return self.dest_dir
        root_dir = os.path.dirname(self.dest_dir)
        tar_file = FSUtil.build_full_path(root_dir, f'{self._folder_name}.tgz')  # pylint: disable=no-member
        print('Creating archive.....')
        with tarfile.open(tar_file, 'w:gz') as tarhandle:
            tarhandle.add(self.dest_dir, arcname='.')
            tarhandle.close()
        print('Created archived resources successfully')
        # delete the csp-resources folder
        FSUtil.remove_path(self.dest_dir)
        return tar_file

    def run(self):
        """
        Main method to fetch and download dependencies.
        This function goes through the following steps:
        1. Fetches the resources from the configuration files.
        2. Downloads the resources to the specified directory.
        3. Optionally compresses the resources into a tar file.
        4. Cleans up the resources if compression is enabled.
        """
        resources_to_download = self._fetch_resources()
        self._download_resources(resources_to_download)
        if self.fetch_all_csp:
            output_res = self._compress_resources()
            print(f'CSP-prepackaged resources stored as {output_res}')
        else:
            print('Packaged with tools resources only.')


if __name__ == '__main__':
    fire.Fire(PrepackageMgr)
