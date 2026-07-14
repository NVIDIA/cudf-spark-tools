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

"""init file of the spark_rapids_pytools package."""

from spark_rapids_pytools.build import get_version, get_spark_dep_version

VERSION = '26.04.8'
# defines the default runtime build version for the user tools environment
SPARK_DEP_VERSION = '357'
__version__ = get_version(VERSION)
__spark_dep_version__ = get_spark_dep_version()
