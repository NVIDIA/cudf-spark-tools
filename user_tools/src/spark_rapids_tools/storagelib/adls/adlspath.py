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

"""Wrapper implementation for ADLS remote path"""

import os

from ..csppath import CspPath, register_path_class


@register_path_class("adls")
class AdlsPath(CspPath):
    """Implementation for ADLS paths"""

    protocol_prefix: str = "abfss://"

    @classmethod
    def get_abfs_account_name(cls, path: str) -> str:
        # ABFS path format: abfss://<file_system>@<account_name>.dfs.core.windows.net/<path_to_file>
        return path.split("@")[1].split(".")[0]

    @classmethod
    def is_protocol_prefix(cls, value: str) -> bool:
        valid_prefix = super().is_protocol_prefix(value)
        if valid_prefix:
            # Check if AZURE_STORAGE_ACCOUNT_NAME env_variable is defined. If not,
            # set it to avoid failures when user is not specifying the platform.
            # https://github.com/NVIDIA/cudf-spark-tools/issues/981
            if "AZURE_STORAGE_ACCOUNT_NAME" not in os.environ:
                account_name = cls.get_abfs_account_name(value)
                os.environ["AZURE_STORAGE_ACCOUNT_NAME"] = account_name
        return valid_prefix
