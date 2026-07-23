# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""AWS S3 endpoint resolution helpers."""

import configparser
import os
from pathlib import Path
from typing import Optional

_AWS_CONFIG_FILE = 'AWS_CONFIG_FILE'
_AWS_DEFAULT_PROFILE = 'AWS_DEFAULT_PROFILE'
_AWS_ENDPOINT_URL = 'AWS_ENDPOINT_URL'
_AWS_ENDPOINT_URL_S3 = 'AWS_ENDPOINT_URL_S3'
_AWS_PROFILE = 'AWS_PROFILE'


def resolve_s3_endpoint_override() -> Optional[str]:
    """
    Resolve an S3 endpoint override using explicit environment variables first,
    then the active AWS profile from the AWS config file.
    """
    env_endpoint = os.environ.get(_AWS_ENDPOINT_URL_S3) or os.environ.get(_AWS_ENDPOINT_URL)
    if env_endpoint:
        return env_endpoint
    return _resolve_profile_s3_endpoint()


def _resolve_profile_s3_endpoint() -> Optional[str]:
    config = configparser.ConfigParser()
    config_path = Path(os.environ.get(_AWS_CONFIG_FILE, Path.home() / '.aws' / 'config')).expanduser()
    try:
        if not config_path.exists():
            return None
        config.read(config_path)
    except (configparser.Error, OSError):
        return None

    for profile_name in _candidate_profile_names():
        endpoint = _resolve_profile_section_endpoint(config, profile_name)
        if endpoint:
            return endpoint
    return None


def _candidate_profile_names() -> list[str]:
    candidates = [
        os.environ.get(_AWS_PROFILE),
        os.environ.get(_AWS_DEFAULT_PROFILE),
        'default',
    ]
    result: list[str] = []
    for candidate in candidates:
        if candidate:
            profile_name = candidate.strip()
            if profile_name and profile_name not in result:
                result.append(profile_name)
    return result


def _profile_section(profile_name: str) -> str:
    return 'default' if profile_name == 'default' else f'profile {profile_name}'


def _resolve_profile_section_endpoint(
        config: configparser.ConfigParser,
        profile_name: str) -> Optional[str]:
    section_name = _profile_section(profile_name)
    if not config.has_section(section_name):
        return None
    section = config[section_name]

    services_endpoint = _resolve_services_section_endpoint(config, section.get('services'))
    if services_endpoint:
        return services_endpoint

    inline_s3_endpoint = _parse_nested_s3_endpoint(section.get('s3'))
    if inline_s3_endpoint:
        return inline_s3_endpoint

    endpoint = section.get('endpoint_url')
    return endpoint.strip() if endpoint and endpoint.strip() else None


def _resolve_services_section_endpoint(
        config: configparser.ConfigParser,
        services_name: Optional[str]) -> Optional[str]:
    if not services_name:
        return None
    section_name = f'services {services_name.strip()}'
    if not config.has_section(section_name):
        return None
    return _parse_nested_s3_endpoint(config[section_name].get('s3'))


def _parse_nested_s3_endpoint(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    for line in raw_value.splitlines():
        key, sep, value = line.strip().partition('=')
        if sep and key.strip() == 'endpoint_url' and value.strip():
            return value.strip()
    return None
