#!/usr/bin/env python3
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

"""NVIDIA cuDF plugin for Apache Spark qualification tool validation script"""

import argparse
import os
import glob
import subprocess

import pandas as pd
from tabulate import tabulate

parser = argparse.ArgumentParser(description="Qualification Tool Validation")
parser.add_argument("--cpu_log", type=str, help="Directory of CPU event log(s)", required=True)
parser.add_argument("--gpu_log", type=str, help="Directory of GPU event log(s)", required=True)
parser.add_argument("--output", type=str, help="Output folder for storing logs", required=True)
parser.add_argument("--platform", type=str, default="onprem", help="Platform name (e.g. onprem, dataproc, databricks-aws")
parser.add_argument("--cpu_profile", type=str, help="Directory of CPU profiler log(s)")
parser.add_argument("--gpu_profile", type=str, help="Directory of GPU profiler log(s)")
parser.add_argument("--jar", type=str, help="Custom tools jar")
parser.add_argument("--verbose", action="store_true", help="flag to generate full verbose output for logging raw node results")
args = parser.parse_args()

cpu_log = args.cpu_log
gpu_log = args.gpu_log
output = args.output
platform = args.platform
cpu_profile = args.cpu_profile
gpu_profile = args.gpu_profile
jar = args.jar
verbose = ""
if args.verbose:
    verbose = "--verbose"

print(f"Output folder = {output}")
print(f"CPU event log = {cpu_log}")
print(f"GPU event log = {gpu_log}")

subprocess.run(f"rm -rf {output}", shell=True)

jar_arg = ""
if jar is not None:
    jar_arg = f"--tools_jar {jar}"

# Generate speedup factors

### run GPU profiler if needed
gpu_profile_dir = ""
if gpu_profile is not None:
    gpu_profile_dir = gpu_profile
else:
    gpu_profile_dir = f"{output}/gpu_profile"
    subprocess.run(f"spark_rapids profiling --csv {jar_arg} --output_folder {gpu_profile_dir} --eventlogs {gpu_log} {verbose}", shell=True)

### run CPU profiler if needed
cpu_profile_dir = ""
if cpu_profile is not None:
    cpu_profile_dir = cpu_profile
else:
    cpu_profile_dir = f"{output}/cpu_profile"
    subprocess.run(f"spark_rapids profiling --csv {jar_arg} --output_folder {cpu_profile_dir} --eventlogs {cpu_log} {verbose}", shell=True)

### run CPU qualification with xgboost model
cpu_tmp_dir = f"{output}/cpu"
subprocess.run(f"spark_rapids qualification --platform {platform} {jar_arg} --estimation_model xgboost --output_folder {cpu_tmp_dir} --eventlogs {cpu_log} {verbose}", shell=True)

# Parse and validate results

### CPU log parsing
cpu_app_info = pd.read_csv(glob.glob(f"{cpu_tmp_dir}/*/qualification_summary.csv")[0])
cpu_query_info = cpu_app_info[["App Name", "App Duration", "Estimated GPU Duration", "Estimated GPU Speedup", "Unsupported Operators Stage Duration Percent", "Speedup Based Recommendation", "Estimated GPU Speedup Category"]]
cpu_query_info["Qualified"] = cpu_query_info["Estimated GPU Speedup Category"].apply(lambda x: x in {'Small', 'Medium', 'Large'})

### GPU log parsing
gpu_query_info = pd.DataFrame(columns = ['App Name', 'GPU Duration'])
counter = 0

for app in glob.glob(f"{gpu_profile_dir}/*/rapids_4_spark_profile/*/application_information.csv"):
    app_info = pd.read_csv(app)
    new_row = pd.DataFrame({'App Name': app_info.loc[0]["appName"], 'GPU Duration': app_info.loc[0]["duration"]}, index=[counter])
    gpu_query_info = pd.concat([gpu_query_info, new_row])
    counter = counter+1

merged_info = cpu_query_info.merge(gpu_query_info, left_on='App Name', right_on='App Name')
merged_info["GPU Speedup"] = (merged_info["App Duration"]/merged_info["GPU Duration"]).apply(lambda x: round(x,2))

speedup_threshold = 1.3
merged_info["True Positive"] = ((merged_info["Qualified"] == True) & (merged_info["GPU Speedup"] > speedup_threshold))
merged_info["False Positive"] = ((merged_info["Qualified"] == True) & (merged_info["GPU Speedup"] <= speedup_threshold))
merged_info["True Negative"] = ((merged_info["Qualified"] != True) & (merged_info["GPU Speedup"] <= speedup_threshold))
merged_info["False Negative"] = ((merged_info["Qualified"] != True) & (merged_info["GPU Speedup"] > speedup_threshold))

tp_count = merged_info["True Positive"].sum()
fp_count = merged_info["False Positive"].sum()
tn_count = merged_info["True Negative"].sum()
fn_count = merged_info["False Negative"].sum()
total = len(merged_info)

print("==================================================")
print("              Application Details")
print("==================================================")
print(tabulate(merged_info, headers='keys', tablefmt='psql'))

print("\n")
print("==================================================")
print("              Classification Metrics")
print("==================================================")
print(f"Total count          = {total}")
print(f"True Positive count  = {tp_count}")
print(f"False Positive count = {fp_count}")
print(f"True Negative count  = {tn_count}")
print(f"False Negative count = {fn_count}")
if (tp_count + fp_count + tn_count + fn_count) != 0:
    print(f"Accuracy             = {round(100.0*(tp_count+tn_count)/(tp_count+fp_count+fn_count+tn_count),2)}")
else:
    print(f"Accuracy             = N/A (no classified apps)")
if (tp_count + fp_count) != 0:
    print(f"Precision            = {round(100.0*tp_count/(tp_count+fp_count),2)}")
else:
    print(f"Precision            = N/A (no predicted positive apps)")
if (tp_count + fn_count) != 0:
    print(f"Recall               = {round(100.0*tp_count/(tp_count+fn_count),2)}")
else:
    print(f"Recall               = N/A (no actual positive apps)")
