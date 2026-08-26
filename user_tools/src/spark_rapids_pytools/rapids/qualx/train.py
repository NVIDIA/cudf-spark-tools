# Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

"""Implementation class representing wrapper around the cuDF plugin QualX training tool."""

from dataclasses import dataclass

from spark_rapids_pytools.rapids.qualx.qualx_tool import QualXTool
from spark_rapids_tools.tools.qualx.qualx_main import train


@dataclass
class Train(QualXTool):
    """
    Wrapper layer around Training tool.

    Attributes
    ----------
    dataset:
        Path to a folder containing one or more dataset JSON files.
    model:
        Path to save the trained XGBoost model.
    n_trials:
        Number of trials for hyperparameter search.
    base_model:
        Path to pre-trained model to continue training from.
    features_csv_dir:
        Path to a directory containing one or more features.csv files.  These files are produced during prediction,
        and must be manually edited to provide a label column (Duration_speedup) and value.
    """

    dataset: str = None
    model: str = None
    n_trials: int = None
    base_model: str = None
    features_csv_dir: str = None

    name = 'train'

    def _run_rapids_tool(self):
        """
        Runs the QualX train tool, saves the trained model and training results.
        """
        try:
            train(
                output_dir=self.output_folder,
                dataset=self.wrapper_options['dataset'],
                model=self.wrapper_options['model'],
                n_trials=self.wrapper_options['n_trials'],
                base_model=self.wrapper_options['base_model'],
                features_csv_dir=self.wrapper_options['features_csv_dir'],
                config=self.wrapper_options['qualx_config'],
            )
            self.logger.info('Training completed successfully.')
            self.logger.info('Trained XGBoost model is saved at: %s', self.wrapper_options['model'])
            self.logger.info('Training results are generated at: %s', self.output_folder)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('Training failed with error: %s', e)
            raise e
