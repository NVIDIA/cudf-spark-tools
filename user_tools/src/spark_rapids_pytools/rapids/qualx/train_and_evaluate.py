# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

"""Implementation class representing wrapper around the cuDF plugin QualX training tool."""

from dataclasses import dataclass

from spark_rapids_pytools.rapids.qualx.qualx_tool import QualXTool
from spark_rapids_tools.tools.qualx.qualx_pipeline import train_and_evaluate


@dataclass
class TrainAndEvaluate(QualXTool):
    """
    Wrapper layer around QualX pipeline.

    Attributes
    ----------
    qualx_pipeline_config:
        Path to YAML config file containing training parameters.
    """

    qualx_pipeline_config: str = None

    name = 'train_and_evaluate'

    def _run_rapids_tool(self):
        """
        Runs the QualX pipeline.
        """
        try:
            train_and_evaluate(config=self.qualx_pipeline_config)
            self.logger.info('QualX pipeline completed successfully.')
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('QualX pipeline failed with error: %s', e)
            raise e
