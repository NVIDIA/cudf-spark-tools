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

"""Implementation class representing wrapper around the cuDF plugin Prediction tool."""

import os
from dataclasses import dataclass

from spark_rapids_pytools.rapids.rapids_tool import RapidsTool
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_pytools.cloud_api.sp_types import get_platform
from spark_rapids_pytools.rapids.tool_ctxt import ToolContext


@dataclass
class InstanceDescription(RapidsTool):
    """Wrapper layer around Generate_Instance_Description Tool."""

    name = 'instance_description'
    instance_file = ''  # local absolute path of the instance description file

    def _connect_to_execution_cluster(self) -> None:
        pass

    def _collect_result(self) -> None:
        pass

    def _archive_phase(self) -> None:
        pass

    def _init_ctxt(self) -> None:
        """
        Initialize the tool context, reusing qualification configurations.
        """
        self.config_path = Utils.resource_path('qualification-conf.yaml')
        self.ctxt = ToolContext(platform_cls=get_platform(self.platform_type),
                                platform_opts=self.wrapper_options.get('platformOpts'),
                                prop_arg=self.config_path,
                                name=self.name)

    def _process_output_args(self) -> None:
        self.logger.debug('Processing Output Arguments')
        if self.output_folder is None:
            self.output_folder = Utils.get_or_set_rapids_tools_env('OUTPUT_DIRECTORY', os.getcwd())
        # make sure that output_folder is being absolute
        self.output_folder = FSUtil.get_abs_path(self.output_folder)
        FSUtil.make_dirs(self.output_folder)
        self.instance_file = f'{self.output_folder}/{self.platform_type}-instance-catalog.json'
        self.logger.debug('Instance description output will be saved in: %s', self.instance_file)

    def _run_rapids_tool(self) -> None:
        self.ctxt.platform.cli.generate_instance_description(self.instance_file)
