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

"""Unit tests for CspPath normalization and file-info caching."""

from pathlib import Path

import pyarrow.fs as arrow_fs
import pytest
from pydantic_core import PydanticCustomError

from spark_rapids_tools.storagelib import CspFs, CspPath
from spark_rapids_tools.storagelib.local.localpath import LocalPath
from spark_rapids_tools.storagelib.s3.aws_config import resolve_s3_endpoint_override
from spark_rapids_tools.storagelib.s3.s3fs import S3Fs
from spark_rapids_tools.storagelib.s3.s3path import S3Path


class TestCspPathNormalization:
    """Verify normalization logic at the CspPath entry point."""

    @pytest.mark.parametrize(
        "input_path",
        [
            "s3a://bucket/path/to/file",
            "s3n://bucket/path/to/file",
            " 's3a://bucket/path/to/file' ",
        ],
    )
    def test_normalizes_s3_family_schemes(self, input_path):
        csp_path = CspPath(input_path)

        assert isinstance(csp_path, S3Path)
        assert str(csp_path) == "s3://bucket/path/to/file"
        assert csp_path.no_scheme == "bucket/path/to/file"

    @pytest.mark.parametrize("input_path", ["file:/tmp/demo.txt", "file://tmp/demo.txt", "file:////tmp/demo.txt"])
    def test_normalizes_file_scheme_variants(self, input_path):
        csp_path = CspPath(input_path)

        assert isinstance(csp_path, LocalPath)
        assert str(csp_path) == Path("/tmp/demo.txt").as_uri()
        assert csp_path.no_scheme == "/tmp/demo.txt"


class TestCspPathFileInfoCaching:
    """Verify metadata reuse and invalidation behaviour."""

    def test_is_file_path_returns_false_when_local_file_lookup_raises(self, monkeypatch, tmp_path):
        local_file = tmp_path / "guarded.txt"
        local_file.write_text("data", encoding="utf-8")

        def raising_is_file(path_obj):
            del path_obj
            raise OSError("permission denied")

        monkeypatch.setattr(CspPath, "is_file", raising_is_file)

        assert CspPath.is_file_path(local_file.as_uri(), raise_on_error=False) is False
        with pytest.raises(PydanticCustomError, match="could not be validated as a local file"):
            CspPath.is_file_path(local_file.as_uri())

    def test_list_all_reuses_file_info_from_directory_listing(self, monkeypatch, tmp_path):
        child = tmp_path / "child.txt"
        child.write_text("data", encoding="utf-8")

        root = CspPath(str(tmp_path))
        original_get_file_info = root.fs_obj.get_file_info
        direct_lookup_calls = []

        def tracking_get_file_info(path_or_selector):
            if not isinstance(path_or_selector, arrow_fs.FileSelector):
                direct_lookup_calls.append(path_or_selector)
            return original_get_file_info(path_or_selector)

        monkeypatch.setattr(root.fs_obj, "get_file_info", tracking_get_file_info)

        items = CspFs.list_all(root)
        child_path = next(item for item in items if str(item).endswith("child.txt"))

        direct_lookup_calls.clear()
        assert child_path.exists() is True
        assert child_path.base_name() == "child.txt"
        assert not direct_lookup_calls

    def test_open_output_stream_invalidates_stale_file_info(self, tmp_path):
        created_path = CspPath(str(tmp_path / "created.txt"))

        assert created_path.exists() is False

        with created_path.open_output_stream() as output_stream:
            output_stream.write(b"payload")

        assert created_path.exists() is True
        assert created_path.is_file() is True


class TestS3EndpointResolution:
    """Verify S3-compatible endpoint override precedence."""

    def test_s3_endpoint_env_takes_precedence_over_profile_config(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text(
            "[default]\n"
            "endpoint_url = https://profile.example\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://env.example")
        monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://s3-env.example")

        assert resolve_s3_endpoint_override() == "https://s3-env.example"

    def test_s3_endpoint_resolves_active_profile_endpoint_url(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text(
            "[default]\n"
            "endpoint_url = https://default.example\n"
            "[profile swiftstack]\n"
            "endpoint_url = https://swiftstack.example\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("AWS_PROFILE", "swiftstack")

        assert resolve_s3_endpoint_override() == "https://swiftstack.example"

    def test_s3_endpoint_resolves_default_profile_endpoint_url(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text(
            "[default]\n"
            "endpoint_url = https://default-profile.example\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("AWS_DEFAULT_PROFILE", "default")

        assert resolve_s3_endpoint_override() == "https://default-profile.example"

    def test_s3_endpoint_resolves_profile_services_s3_endpoint(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text(
            "[profile swiftstack]\n"
            "services = local-services\n"
            "[services local-services]\n"
            "s3 =\n"
            "  endpoint_url = https://services-s3.example\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("AWS_PROFILE", "swiftstack")

        assert resolve_s3_endpoint_override() == "https://services-s3.example"

    def test_s3_endpoint_resolves_profile_nested_s3_endpoint(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text(
            "[profile swiftstack]\n"
            "endpoint_url = https://profile.example\n"
            "s3 =\n"
            "  endpoint_url = https://nested-s3.example\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("AWS_PROFILE", "swiftstack")

        assert resolve_s3_endpoint_override() == "https://nested-s3.example"

    def test_s3_endpoint_ignores_invalid_profile_config(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text("endpoint_url = https://invalid.example\n", encoding="utf-8")
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))

        assert resolve_s3_endpoint_override() is None

    def test_s3_fs_applies_profile_endpoint_override(self, monkeypatch, tmp_path):
        config_file = tmp_path / "aws_config"
        config_file.write_text(
            "[profile swiftstack]\n"
            "endpoint_url = https://swiftstack.example\n",
            encoding="utf-8",
        )
        captured_kwargs = {}

        def capture_handler(cls, *args, **kwargs):
            del cls, args
            captured_kwargs.update(kwargs)
            return object()

        monkeypatch.setattr(CspFs, "create_fs_handler", classmethod(capture_handler))
        monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
        monkeypatch.setenv("AWS_PROFILE", "swiftstack")

        S3Fs.create_fs_handler()

        assert captured_kwargs["endpoint_override"] == "https://swiftstack.example"

    def test_s3_fs_preserves_explicit_endpoint_override(self, monkeypatch):
        captured_kwargs = {}

        def capture_handler(cls, *args, **kwargs):
            del cls, args
            captured_kwargs.update(kwargs)
            return object()

        monkeypatch.setattr(CspFs, "create_fs_handler", classmethod(capture_handler))
        monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://env.example")

        S3Fs.create_fs_handler(endpoint_override="https://explicit.example")

        assert captured_kwargs["endpoint_override"] == "https://explicit.example"
