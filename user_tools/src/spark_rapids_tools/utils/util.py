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

"""Utility and helper methods"""

import base64
import os
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import ssl
import sys
import textwrap
import urllib
import urllib.parse
import tempfile
import xml.etree.ElementTree as elem_tree
from functools import reduce
from operator import getitem
from typing import Any, Optional, ClassVar, Iterator

import certifi
import fire
import pandas as pd
import psutil
from packaging.version import Version
from pydantic import ValidationError, AnyHttpUrl, TypeAdapter

import spark_rapids_pytools
from spark_rapids_pytools import get_version
from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.common.utilities import Utils
from spark_rapids_tools.exceptions import CspPathAttributeError


@contextmanager
def temp_file_with_contents(contents: str,
                            suffix: str = '.txt',
                            dir_path: str = None) -> Iterator[str]:
    """Create a temporary local file with provided contents and ensure cleanup.
    :param contents: The text to write into the temporary file.
    :param suffix: File name suffix to use (e.g., '.txt', '.json').
    :param dir_path: Directory where the temp file is created. Defaults to the system temp directory.
    """
    tmp_file = tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', delete=False,
        dir=dir_path or tempfile.gettempdir(), suffix=suffix
    )
    try:
        tmp_file.write(contents)
        tmp_file.flush()
        temp_path = tmp_file.name
    finally:
        tmp_file.close()

    try:
        yield Path(temp_path).absolute().as_uri()
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:  # pylint: disable=broad-except
            # best-effort cleanup; ignore failures
            pass


def get_elem_from_dict(data, keys):
    try:
        return reduce(getitem, keys, data)
    except LookupError:
        print(f'ERROR: Could not find elements [{keys}]')
        return None


def get_elem_non_safe(data, keys):
    try:
        return reduce(getitem, keys, data)
    except LookupError:
        return None


def stringify_path(fpath) -> str:
    if isinstance(fpath, str):
        actual_val = fpath
    elif hasattr(fpath, '__fspath__'):
        actual_val = os.fspath(fpath)
    else:
        raise CspPathAttributeError('Not a valid path')
    expanded_path = os.path.expanduser(actual_val)
    # make sure we return absolute path
    return os.path.abspath(expanded_path)


def _strip_path_quotes_and_whitespace(fpath: str) -> str:
    """Remove surrounding whitespace and quote characters from a path-like string."""
    return fpath.strip().strip("'\"")


# Glob metacharacters must survive URI construction so that Hadoop's Path/GlobPattern
# can expand wildcards on the JAR side. PurePath.as_uri() percent-encodes these, which
# silently defeats glob expansion for local paths like `/data/cpu/*`.
_LOCAL_URI_SAFE_CHARS = '/*?[]'


def _local_path_to_file_uri(local_path: str) -> str:
    """Build a `file://` URI from an absolute local path, preserving glob metacharacters."""
    return 'file://' + urllib.parse.quote(local_path, safe=_LOCAL_URI_SAFE_CHARS)


def _normalize_file_uri(fpath: str) -> str:
    """Canonicalize `file:` URIs into absolute `file://` form when the scheme is present."""
    if not fpath.lower().startswith('file:'):
        return fpath

    _, _, path_part = fpath.partition(':')
    normalized_path = '/' + path_part.lstrip('/') if path_part else '/'
    absolute_path = os.path.abspath(normalized_path)
    return _local_path_to_file_uri(absolute_path)


def _normalize_s3_uri(fpath: str, target_scheme: str = 's3') -> str:
    """Rewrite S3-family URI schemes to the requested target scheme without changing the path."""
    scheme, separator, path_part = fpath.partition('://')
    if separator and scheme.lower() in {'s3', 's3a', 's3n'}:
        return f'{target_scheme}://{path_part}'
    return fpath


def resolve_and_prepare_log_file(tools_home_dir: str):
    run_id = Utils.get_or_set_rapids_tools_env('RUN_ID')
    log_dir = f'{tools_home_dir}/logs'
    log_file = f'{log_dir}/{run_id}.log'
    Utils.set_rapids_tools_env('LOG_FILE', log_file)
    FSUtil.make_dirs(log_dir)
    return log_file


def is_http_file(value: Any) -> bool:
    try:
        TypeAdapter(AnyHttpUrl).validate_python(value)
        return True
    except ValidationError:
        # ignore
        return False


def get_path_as_uri(fpath: str) -> str:
    normalized_path = _normalize_s3_uri(_normalize_file_uri(_strip_path_quotes_and_whitespace(fpath)))
    if re.match(r'\w+://', normalized_path):
        # that's already a valid url
        return normalized_path
    # stringify the path to apply the common methods which is expanding the file.
    local_path = stringify_path(normalized_path)
    return _local_path_to_file_uri(local_path)


def get_path_as_uri_for_hadoop(fpath: str) -> str:
    """
    Normalize a path for Hadoop/JVM consumers.

    This differs from ``get_path_as_uri`` in one important way:
    S3-family URIs are normalized to ``s3a://`` because that is the
    Hadoop-facing scheme expected by the tools JAR.
    """
    normalized_path = _normalize_s3_uri(
        _normalize_file_uri(_strip_path_quotes_and_whitespace(fpath)),
        target_scheme='s3a'
    )
    if re.match(r'\w+://', normalized_path):
        return normalized_path
    local_path = stringify_path(normalized_path)
    return _local_path_to_file_uri(local_path)


def to_camel_case(word: str) -> str:
    return word.split('_')[0] + ''.join(x.capitalize() or '_' for x in word.split('_')[1:])


def to_camel_capital_case(word: str) -> str:
    return ''.join(x.capitalize() for x in word.split('_'))


def to_snake_case(word: str) -> str:
    return ''.join(['_' + i.lower() if i.isupper() else i for i in word]).lstrip('_')


def dump_tool_usage(cli_class: Optional[str], cli_name: Optional[str], tool_name: Optional[str],
                    raise_sys_exit: Optional[bool] = True) -> None:
    imported_module = __import__('spark_rapids_tools.cmdli.tools_cli', globals(), locals(), [cli_class])
    wrapper_clzz = getattr(imported_module, cli_class)
    usage_cmd = f'{tool_name} -- --help'
    try:
        fire.Fire(wrapper_clzz(), name=cli_name, command=usage_cmd)
    except fire.core.FireExit:
        # ignore the sys.exit(0) thrown by the help usage.
        # ideally we want to exit with error
        pass
    if raise_sys_exit:
        sys.exit(1)


def gen_app_banner(mode: str = '') -> str:
    """
    ASCII Art is generated by an online Test-to-ASCII Art generator tool https://patorjk.com/software/taag
    :return: a string representing the banner of the user tools including the version
    """

    tool_mode_note = '' if mode == '' else f'{mode} CMD'
    c_ver = spark_rapids_pytools.__version__
    return rf"""

********************************************************************
*                                                                  *
*    _____                  __      ____              _     __     *
*   / ___/____  ____ ______/ /__   / __ \____ _____  (_)___/ /____ *
*   \__ \/ __ \/ __ `/ ___/ //_/  / /_/ / __ `/ __ \/ / __  / ___/ *
*  ___/ / /_/ / /_/ / /  / ,<    / _, _/ /_/ / /_/ / / /_/ (__  )  *
* /____/ .___/\__,_/_/  /_/|_|  /_/ |_|\__,_/ .___/_/\__,_/____/   *
*     /_/__  __                  ______    /_/     __              *
*       / / / /_______  _____   /_  __/___  ____  / /____          *
*      / / / / ___/ _ \/ ___/    / / / __ \/ __ \/ / ___/          *
*     / /_/ (__  )  __/ /       / / / /_/ / /_/ / (__  )           *
*     \____/____/\___/_/       /_/  \____/\____/_/____/            *
*                                                                  *
*                                      Version. {c_ver}            *
*{'':>38}{tool_mode_note:<28}*
*                                                                  *
* NVIDIA Corporation                                               *
* spark-rapids-support@nvidia.com                                  *
********************************************************************

"""


def init_environment(short_name: str) -> str:
    """
    Initialize the Python Rapids tool environment.
    Note:
    - This function is not implemented as the `__init__()` method to avoid execution
      when `--help` argument is passed.
    Returns:
        str: The generated UUID for this tool execution session.
    """
    uuid = Utils.gen_uuid_with_ts(suffix_len=8)

    # Set the 'tools_home_dir' to store logs and other configuration files.
    home_dir = Utils.get_sys_env_var('HOME', '/tmp')
    tools_home_dir = FSUtil.build_path(home_dir, '.spark_rapids_tools')
    Utils.set_rapids_tools_env('HOME', tools_home_dir)

    # 'RUN_ID' is used to create a common ID across a tools execution.
    # This ID unifies -
    #    * Appended to loggers for adding meta information
    #    * For creating the output_directory with the same ID
    #    * For creating local dependency work folders
    #    * For creating the log file with the same ID
    Utils.set_rapids_tools_env('RUN_ID', f'{short_name}_{uuid}')

    log_file = resolve_and_prepare_log_file(tools_home_dir)
    print(Utils.gen_report_sec_header('Application Logs'))
    print(f"Run ID  : {Utils.get_or_set_rapids_tools_env('RUN_ID')}")
    print(f'Location: {log_file}')
    print('In case of any errors, please share the log file with the Spark RAPIDS team.\n')

    return uuid


class Utilities:
    """Utility class used to enclose common helpers and utilities."""
    # Assume that the minimum xmx jvm heap allowed to the java cmd is 6 GB.
    min_jvm_xmx: ClassVar[int] = 6
    # Assume that the maximum xmx jvm heap allowed to the java cmd is 32 GB.
    max_jvm_xmx: ClassVar[int] = 32
    # Assume that any tools thread would need at least 6 GB of heap memory.
    min_jvm_heap_per_thread: ClassVar[int] = 6
    # Assume that maximum allowed number of threads to be passed to the tools java cmd is 8.
    max_tools_threads: ClassVar[int] = 8
    # Flag used to disable running tools in parallel. This is a temporary hack to reduce possibility
    # of OOME. Later we can re-enable it.
    conc_mode_enabled: ClassVar[bool] = False
    maven_central_base_url: ClassVar[str] = 'https://repo1.maven.org/maven2'
    maven_base_url_env: ClassVar[str] = 'RAPIDS_TOOLS_MAVEN_BASE_URL'
    maven_username_env: ClassVar[str] = 'RAPIDS_TOOLS_MAVEN_USERNAME'
    maven_password_env: ClassVar[str] = 'RAPIDS_TOOLS_MAVEN_PASSWORD'

    @classmethod
    def get_maven_base_url(cls) -> str:
        env_value = os.environ.get(cls.maven_base_url_env)
        if env_value is None or env_value.strip() == '':
            return cls.maven_central_base_url
        return env_value.strip().rstrip('/')

    @classmethod
    def resolve_maven_url(cls, url: str) -> str:
        if not isinstance(url, str):
            return url
        default_base = cls.maven_central_base_url
        configured_base = cls.get_maven_base_url()
        if configured_base == default_base:
            return url
        if url == default_base:
            return configured_base
        if url.startswith(f'{default_base}/'):
            return f'{configured_base}/{url[len(default_base) + 1:]}'
        return url

    @classmethod
    def is_maven_url(cls, url: str) -> bool:
        if not isinstance(url, str):
            return False
        maven_base_url = cls.get_maven_base_url()
        return url == maven_base_url or url.startswith(f'{maven_base_url}/')

    @classmethod
    def get_maven_basic_auth_header(cls) -> Optional[str]:
        user = os.environ.get(cls.maven_username_env)
        password = os.environ.get(cls.maven_password_env)
        if user is None or user.strip() == '' or password is None or password == '':
            return None
        token = base64.b64encode(f'{user.strip()}:{password}'.encode()).decode()
        return f'Basic {token}'

    @classmethod
    def get_maven_http_headers(cls, url: str) -> dict:
        auth_header = cls.get_maven_basic_auth_header()
        if auth_header is None or not cls.is_maven_url(url):
            return {}
        return {'Authorization': auth_header}

    @classmethod
    def build_maven_url_request(cls, url: str):
        request = urllib.request.Request(url)
        for header, value in cls.get_maven_http_headers(url).items():
            request.add_header(header, value)
        return request

    @classmethod
    def get_latest_mvn_jar_from_metadata(cls, url_base: str,
                                         loaded_version: str = None) -> str:
        """
        Given the defined version in the python tools build, we want to be able to get the highest
        version number of the jar available for download from the mvn repo.
        The returned version is guaranteed to be LEQ to the defined version. For example, it is not
        allowed to use jar version higher than the python tool itself.

        The implementation relies on parsing the "$MVN_REPO/maven-metadata.xml" which guarantees
        that any delays in updating the directory list won't block the python module
        from pulling the latest jar.

        :param url_base: the base url from which the jar file is downloaded. It can be mvn repo.
        :param loaded_version: the version from the python tools in string format
        :return: the string value of the jar that should be downloaded.
        """

        if loaded_version is None:
            loaded_version = cls.get_base_release()
        context = ssl.create_default_context(cafile=certifi.where())
        defined_version = Version(loaded_version)
        jar_version = Version(loaded_version)
        xml_path = f'{url_base}/maven-metadata.xml'
        with urllib.request.urlopen(cls.build_maven_url_request(xml_path), context=context) as resp:
            xml_content = resp.read()
            xml_root = elem_tree.fromstring(xml_content)
            for version_elem in xml_root.iter('version'):
                curr_version = Version(version_elem.text)
                if curr_version <= defined_version:
                    jar_version = curr_version
        # get formatted string
        return cls.reformat_release_version(jar_version)

    @classmethod
    def reformat_release_version(cls, defined_version: Version) -> str:
        # get the release from version
        version_tuple = defined_version.release
        version_comp = list(version_tuple)
        # release format is under url YY.MM.MICRO where MM is 02, 04, 06, 08, 10, and 12
        res = f'{version_comp[0]}.{version_comp[1]:02}.{version_comp[2]}'
        return res

    @classmethod
    def get_base_release(cls) -> str:
        """
        For now the tools_jar is always with major.minor.0.
        This method makes sure that even if the package version is incremented, we will still
        get the correct url.
        :return: a string containing the release number 22.12.0, 23.02.0, amd 23.04.0..etc
        """
        defined_version = Version(get_version(main=None))
        # get the release from version
        return cls.reformat_release_version(defined_version)

    @classmethod
    def get_valid_df_columns(cls, input_cols, input_df: pd.DataFrame) -> list:
        """
        Returns a subset of input_cols that are present in the input_df
        """
        return [col for col in input_cols if col in input_df.columns]

    @classmethod
    def calculate_jvm_max_heap_in_gb(cls) -> int:
        """
        Calculates the maximum heap size to pass to the java cmd based on the memory system.
        By default, the calculation should not be too aggressive because it would lead to reserving
        large memory from the system. In some environments, the OOM killer might kill the tools
        process when the OS runs out of resources.
        To achieve this, we calculate the heap based on the available memory
        (not total memory) capping the value to 32 GB.
        :return: The maximum JVM heap size in GB. It is in the range [min_jvm_xmx-max_jvm_xmx] GB.
        """
        ps_memory = psutil.virtual_memory()
        # get the available memory in the system
        available_sys_gb = ps_memory.available / (1024 ** 3)
        # set the max heap to 30% of total available memory
        heap_based_on_sys = int(0.3 * available_sys_gb)
        # enforce the xmx heap argument to be in the range [6, 32] GB
        return max(cls.min_jvm_xmx, min(heap_based_on_sys, cls.max_jvm_xmx))

    @classmethod
    def calculate_max_tools_threads(cls) -> int:
        """
        Calculates the maximum number of threads that can be passed to the tools' java cmd based on
        the cores of the system. We cap it to 8 threads to reduce teh risk of OOME on the java side.
        :return: The maximum thread pool size in the tools' java cmd in the range [1, 8].
        """
        # Get the number of physical cores in the system. The logical cores is usually higher,
        # but we are being a little bit conservative here to avoid running high number of threads concurrently.
        # Note that on MacOS, the result of both physical/logical count is the same.
        physical_cores = psutil.cpu_count(logical=False)
        # Enforce a safe range [1, 8]
        return max(1, min(cls.max_tools_threads, physical_cores))

    @classmethod
    def adjust_tools_resources(cls,
                               jvm_heap: int,
                               jvm_processes: int,
                               jvm_threads: Optional[int] = None) -> dict:
        """
        Calculate the number of threads to be used in the Rapids Tools JVM cmd.
        In concurrent mode, the profiler needs to have more heap, and more threads.
        """
        # The number of threads is calculated based on the total system memory and the JVM heap size
        # Each thread should at least be running within min_jvm_heap_per_thread GB of heap memory
        concurrent_mode = cls.conc_mode_enabled and jvm_processes > 1
        heap_unit = max(cls.min_jvm_heap_per_thread, jvm_heap // 3 if concurrent_mode else jvm_heap)
        # calculate the maximum number of threads.
        upper_threads = cls.calculate_max_tools_threads() // 3 if concurrent_mode else cls.calculate_max_tools_threads()
        if jvm_threads is None:
            # make sure that the qual threads cannot exceed maximum allowed threads
            num_threads_unit = min(upper_threads, max(1, heap_unit // cls.min_jvm_heap_per_thread))
        else:
            num_threads_unit = max(1, jvm_threads // 3) if concurrent_mode else jvm_threads

        # The Profiling will be assigned the remaining memory resources
        prof_heap = max(jvm_heap - heap_unit, cls.min_jvm_heap_per_thread) if concurrent_mode else heap_unit
        if jvm_threads is None:
            prof_threads = max(1, prof_heap // cls.min_jvm_heap_per_thread) if concurrent_mode else num_threads_unit
            # make sure that the profiler threads cannot exceed maximum allowed threads
            prof_threads = min(upper_threads, prof_threads)
        else:
            prof_threads = max(1, jvm_threads - num_threads_unit) if concurrent_mode else jvm_threads

        # calculate the min heap size based on the max heap size
        min_heap = max(1, heap_unit // 2)

        return {
            'qualification': {
                'jvmMaxHeapSize': heap_unit,
                'jvmMinHeapSize': min_heap,
                'rapidsThreads': num_threads_unit
            },
            'profiling': {
                'jvmMaxHeapSize': prof_heap,
                'jvmMinHeapSize': min_heap,
                'rapidsThreads': prof_threads
            }
        }

    @classmethod
    def squeeze_df_header(cls, df_row: pd.DataFrame, header_width: int) -> pd.DataFrame:
        for column in df_row.columns:
            if len(column) > header_width:
                new_column_name = textwrap.fill(column, header_width, break_long_words=False)
                if new_column_name != column:
                    df_row.columns = df_row.columns.str.replace(column,
                                                                new_column_name, regex=False)
        return df_row

    @classmethod
    def bytes_to_human_readable(cls, num_bytes: int) -> str:
        """
        Convert bytes to human-readable format up to PB
        """
        size_units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        i = 0
        while num_bytes >= 1024 and i < len(size_units) - 1:
            num_bytes /= 1024.0
            i += 1
        return f'{num_bytes:.2f} {size_units[i]}'

    @classmethod
    def parse_memory_size_in_gb(cls, memory_str: str) -> float:
        """
        Helper function to convert JVM memory string to float in gigabytes.
        E.g. '512m' -> 0.5, '2g' -> 2.0
        """
        if not memory_str or len(memory_str) < 2:
            raise ValueError("Memory size string must include a value and a unit (e.g., '512m', '2g').")

        unit = memory_str[-1].lower()
        size_value = float(memory_str[:-1])

        if unit == 'g':
            return size_value
        if unit == 'm':
            return size_value / 1024  # Convert MB to GB
        if unit == 'k':
            return size_value / (1024 ** 2)  # Convert KB to GB

        raise ValueError(f'Invalid memory unit {unit} in memory size: {memory_str}')

    @staticmethod
    def archive_directory(source_folder: str, base_name: str, archive_format: str = 'zip') -> str:
        """
        Archives the specified directory, keeping the directory at the top level in the
        archived file.

        Example:
        source_folder = '/path/to/directory'
        base_name = '/path/to/archived_directory'
        archive_format = 'zip'

        The above example will create a zip file at '/path/to/archived_directory.zip'
        with 'directory' at the top level.

        :param source_folder: Path to the directory to be zipped.
        :param base_name: Base name for the zipped file (without any format extension).
        :param archive_format: Format of the zipped file (default is 'zip').
        :return: Path to the zipped file.
        """
        # Create the zip file with the directory at the top level
        return shutil.make_archive(base_name=base_name,
                                   format=archive_format,
                                   root_dir=os.path.dirname(source_folder),
                                   base_dir=os.path.basename(source_folder))

    @staticmethod
    def string_to_bool(s: str = None) -> bool:
        """
        Converts a string to a boolean using a dictionary lookup, handling case insensitivity.
        Maps specific string values to True or False.
        :param s: The string to convert.
        :return: The corresponding boolean value.
        """
        if s is None:
            return False
        return {
            'true': True,
            'false': False,
            'on': True,
            'off': False,
            'yes': True,
            'no': False
        }.get(s.lower(), False)

    @staticmethod
    def str_to_camel(s: str) -> str:
        """
        Convert a string to camel case.
        Adopted from
        https://www.30secondsofcode.org/python/s/string-capitalize-camel-snake-kebab/#camel-case-string
        > To convert a string to camel case, you can use re.sub() to replace any - or _ with a space,
        > using the regexp r"(_|-)+". Then, use str.title() to capitalize every word and convert the
        > rest to lowercase. Finally, use str.replace() to remove any spaces between words.
        :param s: The input string.
        :return: The camel case version of the input string.
        """
        s = re.sub(r'([_\-])+', ' ', s).title().replace(' ', '')
        return ''.join([s[0].lower(), s[1:]])

    @staticmethod
    def scala_to_pandas_type(scala_type: str) -> str:
        """
        Converts a Scala data type string to its corresponding Pandas-compatible type.
        Note that it is important to use Pandas Nullable type in order to support null data
        passed from the core-tools. For example, Pandas cannot use 'int64' for an integer column if
        it has missing values. It always expect it to be a NaN which might not be the case for
        Scala.

        :param scala_type: The Scala data type string (e.g., 'Int', 'String', 'Double').
        :return: (str) The Pandas-compatible data type string (e.g., 'int64', 'object', 'float64').
        """
        scala_to_pandas_map = {
            'Int': 'Int64',         # Nullable integer
            'Long': 'Int64',        # Both Scala Int and Long map to int64 in Pandas for typical usage.
            'Float': 'float32',     # Float is already nullable (supports NaN)
            'Double': 'float64',    # Float is already nullable (supports NaN)
            'String': 'string',     # Pandas nullable string dtype
            'Boolean': 'boolean',   # Pandas nullable boolean dtype
            'Timestamp': 'datetime64[ns]',
            'Date': 'datetime64[ns]',  # Pandas represents both Date and Timestamp as datetime64[ns].
            'Decimal': 'object',  # Pandas may not have a direct equivalent for Decimal, so 'object' is used.
            # Add more mappings for other Scala types as needed
        }
        return scala_to_pandas_map.get(scala_type, 'object')  # Default to object for unknown types.
