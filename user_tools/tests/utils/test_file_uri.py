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

"""file URI normalization."""

from spark_rapids_tools.utils.util import get_path_as_uri


def test_localhost_file_uri_is_the_local_path():
    assert get_path_as_uri('file://localhost/tmp/x') == 'file:///tmp/x'
    assert get_path_as_uri('file://LocalHost/tmp/x') == 'file:///tmp/x'


def test_empty_host_file_uri_stays_local():
    assert get_path_as_uri('file:///tmp/x') == 'file:///tmp/x'
    assert get_path_as_uri('file:/tmp/x') == 'file:///tmp/x'


def test_other_host_is_not_a_local_directory():
    assert get_path_as_uri('file://remote/tmp/x') == 'file://remote/tmp/x'
