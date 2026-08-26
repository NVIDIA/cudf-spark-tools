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

"""Implementation class representing wrapper around the cuDF plugin Prediction tool."""

from dataclasses import dataclass

from spark_rapids_pytools.common.sys_storage import FSUtil
from spark_rapids_pytools.rapids.qualx.qualx_tool import QualXTool
from spark_rapids_tools.api_v1 import QualCore
from spark_rapids_tools.tools.qualx.qualx_main import predict
from spark_rapids_tools.tools.qualx.util import print_summary, print_speedup_summary


@dataclass
class Prediction(QualXTool):
    """
    Wrapper layer around Prediction Tool.

    Attributes
    ----------
    qual_output: str
        Path to a directory containing qualification tool output.
    """
    qual_output: str = None
    name = 'prediction'

    @property
    def qual_handler(self) -> QualCore:
        return QualCore(self.qual_output)

    def __prepare_prediction_output_info(self) -> dict:
        """
        Prepares the output information for the prediction results.
        :return: Dictionary with the full paths to the output files.
        """
        # build the full output path for the predictions output files
        predictions_info = self.ctxt.get_value('local', 'output', 'predictionModel')
        output_dir = FSUtil.build_path(self.ctxt.get_output_folder(), predictions_info['outputDirectory'])
        FSUtil.make_dirs(output_dir)

        files_info = predictions_info['files']
        # update files_info dictionary with full file paths
        for entry in files_info:
            file_name = files_info[entry]['name']
            file_path = FSUtil.build_path(output_dir, file_name)
            files_info[entry]['path'] = file_path
        return files_info

    def _run_rapids_tool(self):
        """
        Runs the QualX prediction tool, prints the summary and saves the output to a csv file.
        """
        try:
            output_info = self.__prepare_prediction_output_info()
            estimation_model_args = self.wrapper_options.get('estimationModelArgs')
            if estimation_model_args is not None and estimation_model_args:
                custom_model_file = estimation_model_args['customModelFile']
            else:
                custom_model_file = None
            df = predict(platform=self.platform_type.map_to_java_arg(),
                         qual=self.qual_output,
                         output_info=output_info,
                         model=custom_model_file,
                         config=self.wrapper_options.get('qualx_config'),
                         qual_handlers=[self.qual_handler])
            if not df.empty:
                print_summary(df)
                print_speedup_summary(df)
                df.to_csv(f'{self.output_folder}/prediction.csv', float_format='%.2f')
                self.logger.info('Prediction completed successfully.')
                self.logger.info('Prediction results are generated at: %s', self.output_folder)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error('Prediction failed with error: %s', e)
            raise e
