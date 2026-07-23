# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

"""Wrapper for the S3 File system"""

from typing import Any

from spark_rapids_tools.storagelib.cspfs import register_fs_class, CspFs
from spark_rapids_tools.storagelib.s3.aws_config import resolve_s3_endpoint_override


@register_fs_class('s3', 'S3FileSystem')
class S3Fs(CspFs):
    """
    Implementation of FileSystem for S3-backed filesystem on top of pyArrow
    (Docstring copied from pyArrow.S3FileSystem)

    The S3FileSystem is initialized with the following list of arguments:

    >>> S3FileSystem(access_key=None, *, secret_key=None, session_token=None, bool anonymous=False,
    ...     region=None, request_timeout=None, connect_timeout=None, scheme=None,
    ...     endpoint_override=None, bool background_writes=True, default_metadata=None,
    ...     role_arn=None, session_name=None, external_id=None, load_frequency=900,
    ...     proxy_options=None, allow_bucket_creation=False, allow_bucket_deletion=False,
    ...     retry_strategy: S3RetryStrategy = AwsStandardS3RetryStrategy(max_attempts=3))

    If neither access_key nor secret_key are provided, and role_arn is also not
    provided, then attempts to initialize from AWS environment variables,
    otherwise both access_key and secret_key must be provided.
    """

    @classmethod
    def create_fs_handler(cls, *args: Any, **kwargs: Any):
        """
        Create S3FileSystem handler with automatic endpoint detection.
        """
        endpoint_url = resolve_s3_endpoint_override()
        if endpoint_url and 'endpoint_override' not in kwargs:
            # Only set endpoint_override if it's not already explicitly provided
            kwargs['endpoint_override'] = endpoint_url
        return super().create_fs_handler(*args, **kwargs)
