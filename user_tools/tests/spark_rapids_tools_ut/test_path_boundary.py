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

"""Tests for path qualification at the tools-to-Hadoop boundary."""

from pathlib import Path
from types import SimpleNamespace

from spark_rapids_pytools.rapids.rapids_tool import RapidsJarTool


class RapidsJarToolTestShim(RapidsJarTool):
    """Expose wrapper boundary helpers through a public test shim."""

    def process_jar_arg(self):
        return self._process_jar_arg()

    @classmethod
    def prepare_hadoop_arg_path(cls, path_value: str) -> str:
        return getattr(RapidsJarTool, '_prepare_hadoop_arg_path')(path_value)

    @classmethod
    def prepare_hadoop_arg_value(cls, option_key: str, option_value):
        return getattr(RapidsJarTool, '_prepare_hadoop_arg_value')(option_key, option_value)

    @classmethod
    def prepare_eventlog_args_for_hadoop(cls, eventlogs, work_dir: str):
        return getattr(RapidsJarTool, '_prepare_eventlog_args_for_hadoop')(eventlogs, work_dir)

    @classmethod
    def prepare_eventlog_arg_for_hadoop(cls, eventlog: str, work_dir: str) -> str:
        return getattr(RapidsJarTool, '_prepare_eventlog_arg_for_hadoop')(eventlog, work_dir)


def _silent_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )


def _build_jar_tool_shim(wrapper_options, ctxt):
    tool = object.__new__(RapidsJarToolTestShim)
    tool.wrapper_options = wrapper_options
    tool.ctxt = ctxt
    tool.logger = _silent_logger()
    return tool


class TestPathBoundaryQualification:
    """Verify Hadoop-bound path qualification remains explicit and idempotent."""

    def test_process_jar_arg_uses_local_resource_when_no_explicit_jar(self):
        rapids_args = {}
        local_resource_jar = '/opt/resources/rapids-4-spark-tools_2.12-25.08.1.jar'

        def fail_download(*args, **kwargs):
            raise AssertionError('local resource jar should not be copied')

        ctxt = SimpleNamespace(
            platform=SimpleNamespace(storage=SimpleNamespace(download_resource=fail_download)),
            use_local_tools_jar=lambda: True,
            get_rapids_jar_url=lambda: local_resource_jar,
            get_local_work_dir=lambda: '/tmp/work',
            add_rapids_args=rapids_args.__setitem__,
        )
        tool = _build_jar_tool_shim({}, ctxt)

        tool.process_jar_arg()

        assert rapids_args['jarFilePath'] == local_resource_jar
        assert rapids_args['jarFileName'] == 'rapids-4-spark-tools_2.12-25.08.1.jar'

    def test_process_jar_arg_downloads_explicit_remote_jar(self):
        rapids_args = {}
        download_calls = []
        remote_tools_jar = 'gs://bucket/path/rapids-4-spark-tools_2.12-25.08.1.jar'
        downloaded_jar = '/tmp/work/rapids-4-spark-tools_2.12-25.08.1.jar'

        def record_download(src, dest, fail_ok=False, create_dir=True):
            download_calls.append((src, dest, fail_ok, create_dir))
            return downloaded_jar

        ctxt = SimpleNamespace(
            platform=SimpleNamespace(storage=SimpleNamespace(download_resource=record_download)),
            use_local_tools_jar=lambda: True,
            get_rapids_jar_url=lambda: '/opt/resources/rapids-4-spark-tools_2.12-25.08.1.jar',
            get_local_work_dir=lambda: '/tmp/work',
            add_rapids_args=rapids_args.__setitem__,
        )
        tool = _build_jar_tool_shim({'toolsJar': remote_tools_jar}, ctxt)

        tool.process_jar_arg()

        assert download_calls == [(remote_tools_jar, '/tmp/work', False, True)]
        assert rapids_args['jarFilePath'] == downloaded_jar
        assert rapids_args['jarFileName'] == 'rapids-4-spark-tools_2.12-25.08.1.jar'

    def test_prepare_hadoop_arg_path_qualifies_local_paths(self, tmp_path):
        local_file = tmp_path / 'worker_info.yaml'

        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(str(local_file)) == f'file://{local_file}'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(f" '{local_file}' ") == f'file://{local_file}'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('file:/tmp/demo.yaml') == 'file:///tmp/demo.yaml'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('file://tmp/demo.yaml') == 'file:///tmp/demo.yaml'

    def test_prepare_hadoop_arg_path_normalizes_s3_family_for_hadoop(self):
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('s3://bucket/path') == 's3a://bucket/path'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('s3n://bucket/path') == 's3a://bucket/path'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('s3a://bucket/path') == 's3a://bucket/path'

    def test_prepare_hadoop_arg_path_preserves_non_s3_remote_schemes(self):
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('gs://bucket/path') == 'gs://bucket/path'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path('hdfs://namenode:8020/path') == 'hdfs://namenode:8020/path'

    def test_prepare_hadoop_arg_value_targets_only_jvm_bound_keys(self, tmp_path):
        local_file = tmp_path / 'driver.log'

        assert RapidsJarToolTestShim.prepare_hadoop_arg_value('driverlog', str(local_file)) == f'file://{local_file}'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_value('target_cluster_info', str(local_file)) == \
            f'file://{local_file}'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_value('toolsJar', str(local_file)) == str(local_file)

    def test_prepare_eventlog_args_for_hadoop_qualifies_each_token(self, tmp_path):
        local_dir = tmp_path / 'eventlogs'
        local_txt = tmp_path / 'eventlogs.txt'
        tokens = [
            str(local_dir),
            'file:/tmp/eventlog.zstd',
            's3://bucket/eventlog-legacy.zstd',
            's3a://bucket/eventlog.zstd',
            str(local_txt),
        ]

        prepared_tokens = RapidsJarToolTestShim.prepare_eventlog_args_for_hadoop(tokens, str(tmp_path))

        assert prepared_tokens == [
            f'file://{local_dir}',
            'file:///tmp/eventlog.zstd',
            's3a://bucket/eventlog-legacy.zstd',
            's3a://bucket/eventlog.zstd',
            f'file://{local_txt}',
        ]

    def test_prepare_eventlog_arg_for_hadoop_rewrites_local_txt_lists(self, tmp_path):
        eventlogs_txt = tmp_path / 'eventlogs.txt'
        eventlogs_txt.write_text(
            '/tmp/logs/app-1,\n'
            'file:/tmp/logs/app-2;\n'
            's3://bucket/app-legacy\t'
            's3a://bucket/app-3\n',
            encoding='utf-8'
        )

        rewritten_arg = RapidsJarToolTestShim.prepare_eventlog_arg_for_hadoop(str(eventlogs_txt), str(tmp_path))

        assert rewritten_arg.startswith('file://')
        assert rewritten_arg != f'file://{eventlogs_txt}'

        rewritten_path = Path(rewritten_arg.removeprefix('file://'))
        assert rewritten_path.exists()
        assert rewritten_path.read_text(encoding='utf-8') == (
            'file:///tmp/logs/app-1\n'
            'file:///tmp/logs/app-2\n'
            's3a://bucket/app-legacy\n'
            's3a://bucket/app-3\n'
        )

    def test_prepare_eventlog_arg_for_hadoop_preserves_empty_local_txt_lists(self, tmp_path):
        eventlogs_txt = tmp_path / 'empty-eventlogs.txt'
        eventlogs_txt.write_text('  \n,\n;\n', encoding='utf-8')

        rewritten_arg = RapidsJarToolTestShim.prepare_eventlog_arg_for_hadoop(str(eventlogs_txt), str(tmp_path))

        assert rewritten_arg == eventlogs_txt.as_uri()

    def test_prepare_eventlog_arg_for_hadoop_preserves_remote_txt_lists(self):
        remote_txt = 's3a://bucket/eventlogs.txt'

        assert RapidsJarToolTestShim.prepare_eventlog_arg_for_hadoop(remote_txt, '/tmp') == remote_txt

    def test_prepare_hadoop_arg_path_preserves_glob_metacharacters(self):
        # Hadoop expands globs on the JAR side, so `*`, `?`, and `[...]` ranges must survive
        # URI construction verbatim. Percent-encoding them silently disables glob expansion.
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(
            '/data/eventlogs/cpu/*'
        ) == 'file:///data/eventlogs/cpu/*'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(
            'file:///data/eventlogs/gpu/*'
        ) == 'file:///data/eventlogs/gpu/*'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(
            '/data/eventlogs/app-?'
        ) == 'file:///data/eventlogs/app-?'
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(
            '/data/eventlogs/app-[0-9]'
        ) == 'file:///data/eventlogs/app-[0-9]'
        # Remote globs must not be touched either.
        assert RapidsJarToolTestShim.prepare_hadoop_arg_path(
            's3://bucket/eventlogs/*'
        ) == 's3a://bucket/eventlogs/*'

    def test_rewrite_local_eventlog_list_preserves_glob_metacharacters(self, tmp_path):
        eventlogs_txt = tmp_path / 'eventlogs.txt'
        eventlogs_txt.write_text(
            '/data/eventlogs/cpu/*\n'
            '/data/eventlogs/gpu/*\n',
            encoding='utf-8'
        )

        rewritten_arg = RapidsJarToolTestShim.prepare_eventlog_arg_for_hadoop(
            str(eventlogs_txt), str(tmp_path)
        )

        rewritten_path = Path(rewritten_arg.removeprefix('file://'))
        assert rewritten_path.read_text(encoding='utf-8') == (
            'file:///data/eventlogs/cpu/*\n'
            'file:///data/eventlogs/gpu/*\n'
        )
