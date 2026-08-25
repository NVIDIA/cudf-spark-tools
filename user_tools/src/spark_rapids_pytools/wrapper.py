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

"""Wrapper class to run tools associated with the NVIDIA cuDF plugin for Apache Spark."""

import fire

from spark_rapids_pytools.wrappers.databricks_aws_wrapper import DBAWSWrapper
from spark_rapids_pytools.wrappers.databricks_azure_wrapper import DBAzureWrapper
from spark_rapids_pytools.wrappers.dataproc_wrapper import DataprocWrapper
from spark_rapids_pytools.wrappers.emr_wrapper import EMRWrapper


def main():
    fire.Fire({
        'emr': EMRWrapper,
        'dataproc': DataprocWrapper,
        'databricks-aws': DBAWSWrapper,
        'databricks-azure': DBAzureWrapper,
    })


if __name__ == '__main__':
    main()
