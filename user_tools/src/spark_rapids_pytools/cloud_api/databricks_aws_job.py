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

"""Implementation of Job submissions on Databricks AWS"""

from dataclasses import dataclass

from spark_rapids_pytools.rapids.rapids_job import RapidsLocalJob


@dataclass
class DBAWSLocalRapidsJob(RapidsLocalJob):
    """
    Implementation of a cuDF plugin job that runs local on a local machine.
    """
    job_label = 'DBAWSLocal'

    def _build_submission_cmd(self) -> list:
        # env vars are added later as a separate dictionary
        cmd_arg = super()._build_submission_cmd()
        return cmd_arg
