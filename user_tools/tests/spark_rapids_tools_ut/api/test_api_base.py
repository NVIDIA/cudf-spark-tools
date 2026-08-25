# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

"""Base class and utilities for API testing."""

import os
import shutil
import tempfile
import unittest
from typing import List, Optional


class APITestBase(unittest.TestCase):
    """
    Base class for API tests with minimal qual wrapper output structure.

    Creates the full wrapper output structure:
      qual_<timestamp>/
      ├── qualification_summary.csv
      ├── qual_core_output/
      │   ├── status.csv
      │   └── tuning_apps/<app-id>/

    This matches what users actually interact with via qualWrapperOutput.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures (runs once per test class)."""
        cls.sample_app_id_1 = 'application_1234567890_0001'
        cls.sample_app_id_2 = 'application_1234567890_0002'

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory with wrapper structure
        self.temp_dir = tempfile.mkdtemp()
        self.wrapper_output = os.path.join(self.temp_dir, 'qual_20250101000000_12345678')
        self.core_output = os.path.join(self.wrapper_output, 'qual_core_output')

        # Create base directory structure
        os.makedirs(self.wrapper_output, exist_ok=True)
        os.makedirs(self.core_output, exist_ok=True)

        # Create minimal required files
        self.create_wrapper_summary()
        self.create_core_status()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_wrapper_summary(self):
        """Create minimal qualification_summary.csv at wrapper level."""
        summary_path = os.path.join(self.wrapper_output, 'qualification_summary.csv')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('Vendor,Driver Host,App Name,App ID,Estimated GPU Speedup\n')
            f.write(f'test,localhost,TestApp1,{self.sample_app_id_1},2.5\n')
            f.write(f'test,localhost,TestApp2,{self.sample_app_id_2},1.8\n')

    def create_core_status(self, app_ids: Optional[List[str]] = None):
        """Create minimal status.csv at core level for app discovery."""
        if app_ids is None:
            app_ids = [self.sample_app_id_1, self.sample_app_id_2]

        status_csv_path = os.path.join(self.core_output, 'status.csv')
        with open(status_csv_path, 'w', encoding='utf-8') as f:
            f.write('Event Log,Status,App ID,Attempt ID,App Name,Description\n')
            for idx, app_id in enumerate(app_ids, 1):
                f.write(f'/path/to/eventlog{idx},SUCCESS,{app_id},0,TestApp{idx},Processing time: {idx*100}ms\n')

    def create_per_app_directory(self, subfolder: str, app_id: str) -> str:
        """
        Create a per-app directory under qual_core_output/<subfolder>/<app_id>/.

        Args:
            subfolder: The subfolder name (e.g., 'qual_metrics', 'tuning_apps')
            app_id: The application ID

        Returns:
            The created directory path
        """
        app_dir = os.path.join(self.core_output, subfolder, app_id)
        os.makedirs(app_dir, exist_ok=True)
        return app_dir

    def write_file(self, directory: str, filename: str, content: str):
        """
        Write a file to the specified directory.

        Args:
            directory: The directory path
            filename: The filename
            content: The file content
        """
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


class TuningAPITestBase(APITestBase):
    """
    Base class for API tests with tuning-specific setup.

    Extends APITestBase with tuning-specific setup:
    - Creates tuning_apps/ directory structure
    - Provides sample tuning file content
    - Helper methods to create tuning files
    """

    def setUp(self):
        """Set up test fixtures with tuning-specific structure."""
        super().setUp()

        # Create tuning_apps directory under core output
        self.tuning_apps_path = os.path.join(self.core_output, 'tuning_apps')
        os.makedirs(self.tuning_apps_path, exist_ok=True)

    def create_tuning_files(self, app_id: str,
                            include_bootstrap: bool = True,
                            include_recommendations: bool = True,
                            include_combined: bool = False):
        """
        Create tuning files for a specific application.

        Args:
            app_id: The application ID
            include_bootstrap: Whether to create bootstrap.conf
            include_recommendations: Whether to create recommendations.log
            include_combined: Whether to create combined.conf (optional file)

        Returns:
            The application's tuning directory path
        """
        app_dir = self.create_per_app_directory('tuning_apps', app_id)

        if include_bootstrap:
            bootstrap_content = self.get_sample_bootstrap_conf(app_id)
            self.write_file(app_dir, 'bootstrap.conf', bootstrap_content)

        if include_recommendations:
            recommendations_content = self.get_sample_recommendations_log(app_id)
            self.write_file(app_dir, 'recommendations.log', recommendations_content)

        if include_combined:
            combined_content = self.get_sample_combined_conf(app_id)
            self.write_file(app_dir, 'combined.conf', combined_content)

        return app_dir

    @staticmethod
    def get_sample_bootstrap_conf(app_id: str) -> str:  # pylint: disable=unused-argument
        """Generate sample bootstrap.conf content in Spark CLI format."""
        return """--conf spark.rapids.sql.enabled=true
--conf spark.executor.cores=16
--conf spark.executor.memory=32g
--conf spark.executor.resource.gpu.amount=1
--conf spark.rapids.memory.pinnedPool.size=4g
--conf spark.rapids.sql.concurrentGpuTasks=3
"""

    @staticmethod
    def get_sample_recommendations_log(app_id: str) -> str:
        """Generate sample recommendations.log content matching actual format."""
        return f"""### Recommended SPARK Configuration on GPU Cluster for App: {app_id} ###

Spark Properties:
--conf spark.executor.cores=16
--conf spark.executor.memory=32g
--conf spark.executor.resource.gpu.amount=1
--conf spark.plugins=com.nvidia.spark.SQLPlugin
--conf spark.rapids.memory.pinnedPool.size=4g
--conf spark.rapids.sql.concurrentGpuTasks=3
--conf spark.rapids.sql.enabled=true
--conf spark.sql.adaptive.enabled=true
--conf spark.task.resource.gpu.amount=0.001

Comments:
- 'spark.executor.resource.gpu.amount' should be set to allow Spark to schedule GPU resources.
- 'spark.plugins' should be set to the class name required for the cuDF plugin.
  Refer to: https://docs.nvidia.com/spark-rapids/user-guide/latest/getting-started/overview.html
- 'spark.rapids.memory.pinnedPool.size' was not set.
- 'spark.rapids.sql.concurrentGpuTasks' was not set.
- 'spark.task.resource.gpu.amount' was not set.
- Required jar for the NVIDIA cuDF plugin for Apache Spark is missing from the classpath entries.
  If the cuDF plugin jar is being bundled with your Spark distribution, this step is not needed.
- To enable Spark to discover and schedule GPU resources, set the
  'spark.executor.resource.gpu.discoveryScript' property according to cluster
  manager's documentation. Sample discovery script is available at
  '${{SPARK_HOME}}/examples/src/main/scripts/getGpusResources.sh'.
"""

    @staticmethod
    def get_sample_combined_conf(app_id: str) -> str:
        """Generate sample combined.conf content merging GPU and existing configs."""
        return f'--conf spark.app.id={app_id}\n' + """--conf spark.rapids.sql.enabled=true
--conf spark.executor.cores=16
--conf spark.executor.memory=32g
--conf spark.executor.resource.gpu.amount=1
--conf spark.rapids.memory.pinnedPool.size=4g
--conf spark.sql.adaptive.enabled=true
--conf spark.sql.adaptive.coalescePartitions.enabled=true
"""
