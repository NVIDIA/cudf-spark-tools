#!/bin/bash
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


# Usage: ./build.sh [build_mode] [jar_url]
# This script takes a build_mode ("fat" or "non-fat") and the jar url as parameter.
# Build mode is a mandatory parameter while the jar url is optional.
# Additional Maven arguments can be passed via the TOOLS_MAVEN_ARGS environment variable.
# Example: TOOLS_MAVEN_ARGS="-Dbuildver=350 -Dhadoop.version=3.3.6" ./build.sh non-fat
# If the build_mode is "fat" and jar_url is provided, it downloads the JAR from the URL and packages the CSP dependencies with the whl.
# If the build_mode is "fat" and jar_url is not provided, it builds the JAR from source and packages the CSP dependencies with the whl.
# If the build_mode is "non-fat" and jar_url is provided, it downloads the JAR from the URL and packages it with the wheel.
# If the build_mode is "non-fat" and jar_url is not provided, it builds the JAR from source and packages it with the wheel.

###############################
# define utilities and methods
###############################

# define coloring for logging
UNDER='\033[4m'
RED='\033[31;1m'
GREEN='\033[32;1m'
YELLOW='\033[33;1m'
BLUE='\033[34;1m'
MAGENTA='\033[35;1m'
CYAN='\033[36;1m'
WHITE='\033[37;1m'
ENDCOLOR='\033[0m'

PRODUCT_NAME="cuDF plugin user tools"

log_msg() {
  echo -e "${1}${2}${ENDCOLOR}"
}

log_error() {
  log_msg "${RED}" "Error: $1"
}

log_section() {
  echo
  log_msg "${MAGENTA}" "$1"
  echo
}

log_info() {
  log_msg "${YELLOW}" "$1"
}

log_debug() {
  echo -e "$1"
}

bail() {
 log_error "$1"
 exit 1
}

###############################
# Process Arguments
###############################

# Get the build mode argument
# Check if build_mode is provided and valid
if [ -z "$1" ]; then
  bail "build_mode parameter is required. Use either 'fat' or 'non-fat'."
fi

# Validate build_mode is either "fat" or "non-fat"
if [ "$1" != "fat" ] && [ "$1" != "non-fat" ]; then
  bail "build_mode must be either 'fat' or 'non-fat'. Got '$1' instead."
fi

build_mode="$1"
jar_url="$2"

# Optional mode to only prepare resources (generate Tools JAR, copy reports) without building the wheel
# Usage: SKIP_WHEEL_PACKAGE_BUILD=1 ./build.sh non-fat [jar_url]
SKIP_WHEEL_PACKAGE_BUILD=${SKIP_WHEEL_PACKAGE_BUILD:-0}

# get the directory of the script
WORK_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# Define wrapper module directories
WRAPPER_RESOURCES_DIR="src/spark_rapids_pytools/resources"
# define the destination folder where the core files will be located.
GENERATED_RES_REL_PATH="generated_files/core"
DEST_CORE_REPORTS_REL_PATH="${GENERATED_RES_REL_PATH}/reports"
DEST_CORE_JARS_REL_PATH="${GENERATED_RES_REL_PATH}/jars"

PREPACKAGED_FOLDER="csp-resources"

# Define core module directories
# Constants and variables of core module
CORE_DIR="$WORK_DIR/../core"
CORE_RESOURCES_DIR="${CORE_DIR}/src/main/resources"
SRC_CORE_REPORTS_REL_PATH="configs/reports"
TOOLS_JAR_FILE=""
BUILD_DIR="/var/tmp/spark_rapids_user_tools_cache/build_$(date +%Y%m%d_%H%M%S)"

# Function to download JAR from URL
download_jar_from_url() {
  local url="$1"
  local output_dir="$2"

  # Create download directory if it doesn't exist
  mkdir -p "${output_dir}"

  # Extract filename from URL
  local filename
  filename=$(basename "$url")
  local output_path="${output_dir}/$filename"

  # Validate the JAR filename to ensure it is not a source, javadoc, or test JAR
  if [[ "$filename" == *"sources.jar" || "$filename" == *"javadoc.jar" || "$filename" == *"tests.jar" ]]; then
    bail "Invalid JAR file: ${filename}. Source, javadoc, or test JARs are not allowed."
  fi

  log_debug "Downloading JAR from $url to ${output_path}"

  curl_exit_code=$(curl -L -f -o "${output_path}" "$url"; echo $?)
  if [ "$curl_exit_code" -ne 0 ]; then
    bail "Failed to download JAR from URL: $url"
  fi

  TOOLS_JAR_FILE="${output_path}"
  log_info "Tools JAR download complete from url into: ${TOOLS_JAR_FILE}"
}

clean_up_build_jars() {
  if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
  fi
}

# Function to run mvn command to build the tools jar
# This function skips the test cases and builds the jar file and only
# picks the jar file without sources/javadoc/tests..
# Maven arguments can be passed via TOOLS_MAVEN_ARGS environment variable
build_jar_from_source() {
  # store teh current directory
  local curr_dir
  curr_dir=$(pwd)
  local jar_dir="${CORE_DIR}"/target
  cd "${CORE_DIR}" || exit

  # Construct Maven command with optional arguments from environment
  local mvn_cmd="mvn clean package -DskipTests"

  # Add additional Maven arguments if provided via environment variable
  if [ -n "${TOOLS_MAVEN_ARGS}" ]; then
    log_debug "Using additional Maven arguments: ${TOOLS_MAVEN_ARGS}"
    mvn_cmd="${mvn_cmd} ${TOOLS_MAVEN_ARGS}"
  fi

  log_debug "Running Maven command: ${mvn_cmd}"

  # build mvn
  eval "${mvn_cmd}"
  mvn_exit_code=$?
  if [ $mvn_exit_code -ne 0 ]; then
    bail "Failed to build the tools jar from source"
  fi
  # rename jar file stripping snapshot
  TOOLS_JAR_FILE=$(find "${jar_dir}" -type f \( -iname "rapids-4-spark-tools_*.jar" ! -iname "*sources.jar" ! -iname "*tests.jar" ! -iname "original-rapids-4*.jar" ! -iname "*javadoc.jar" \) | head -n 1)

  if [ -z "$TOOLS_JAR_FILE" ]; then
    bail "Failing because tools jar could not be located"
  else
    log_info "Tools jar file: $TOOLS_JAR_FILE"
  fi
  # restore the current directory
  cd "${curr_dir}" || exit
}

# Function to run the dependency downloader script for non-fat/fat mode
# prepackage_mgr.py file downloads the dependencies for the csp-related resources
# in case of fat mode.
# In case of non-fat mode, it just copies the tools jar into the tools-resources folder
# --fetch_all_csp=True toggles the fat/non-fat mode for the script
download_web_dependencies() {
  local res_dir="$1"
  local is_fat_mode="$2"
  local web_downloader_script="$res_dir/dev/prepackage_mgr.py"
  log_section "Downloading dependencies"
  python "$web_downloader_script" run --resource_dir="$res_dir" --tools_jar="${TOOLS_JAR_FILE}" --fetch_all_csp="$is_fat_mode"
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    bail "Dependency download failed. Exiting"
  fi
}

# Function to remove dependencies from the fat directory
remove_web_dependencies() {
  local res_dir="$1"
  log_info "Cleaning-up the generated-core files and prepackaged CSPs"
  # remove all subdirectories and files from generated_files core
  rm -rf "${res_dir:?}"/"${GENERATED_RES_REL_PATH}"/*
  # remove csp resources recursively
  rm -rf "${res_dir:?}"/"$PREPACKAGED_FOLDER"
  # remove compressed file in case archive-mode was enabled
  rm -f "${res_dir:?}"/"$PREPACKAGED_FOLDER".tgz
}

# Function to copy reports configurations from core module to generated_files folder
copy_reports_from_core() {
  local wrapper_res_dir="$1"
  local src_core_reports="${CORE_RESOURCES_DIR}/${SRC_CORE_REPORTS_REL_PATH}"
  local wrapper_gen_dir="${wrapper_res_dir}/$GENERATED_RES_REL_PATH"

  if [ -d "$src_core_reports" ]; then
    log_debug "Copying core reports to the wrapper module"
    mkdir -p "${wrapper_gen_dir}"
    cp -r "${src_core_reports}" "${wrapper_gen_dir}"
    cp_exit_code=$?
    if [ $cp_exit_code -ne 0 ]; then
      bail "Failed to copy report files"
    fi
    log_info "Finished copying report files to generated files directory"
  else
    bail "directories were not found ${src_core_reports}"
  fi
}

# Prepare-only: generate the Tools JAR and copy the core reports into generated resources
# This avoids building the Python wheel and can be used by CI/tox prior to isolated builds
prepare_resources() {
  # Clean any previously generated resources
  remove_web_dependencies "$WRAPPER_RESOURCES_DIR"

  # Build or download the tools JAR
  log_section "Building the tools jar...."
  if [ -n "$jar_url" ]; then
    log_info "Using provided JAR URL"
    download_jar_from_url "$jar_url" "$BUILD_DIR"
  else
    log_info "Building JAR from source"
    build_jar_from_source
  fi

  # Create destination jars directory and copy the built tools jar
  local dest_jars_dir="${WRAPPER_RESOURCES_DIR}/${DEST_CORE_JARS_REL_PATH}"
  mkdir -p "${dest_jars_dir}"
  cp -f "${TOOLS_JAR_FILE}" "${dest_jars_dir}/"

  # Copy core reports into generated_files
  copy_reports_from_core "$WRAPPER_RESOURCES_DIR"

  # Cleanup temporary build artifacts
  clean_up_build_jars
}

# Pre-build setup
pre_build() {
  log_info "upgrade pip"
  pip install --upgrade pip
  log_debug "rm previous build and dist directories"
  rm -rf build/ dist/
  log_info "install build dependencies using pip"
  pip install build
  # Note: Removed -e .[dev-env] to avoid overriding existing package installations
  if ! pip install ".[dev-env]"; then
    bail "Failed to download the package with dependencies"
  fi
}

# Build process
build() {
  # Deletes pre-existing csp-resources.tgz folder
  remove_web_dependencies "$WRAPPER_RESOURCES_DIR"
  # Build the tools jar from source
  log_section "Building the tools jar...."
  if [ -n "$jar_url" ]; then
    log_info "Using provided JAR URL"
    download_jar_from_url "$jar_url" "$BUILD_DIR"
  else
    log_info "Building JAR from source"
    build_jar_from_source
  fi
  if [ "$build_mode" = "fat" ]; then
    log_info "Building in fat mode"
    # This will download the dependencies and create the csp-resources
    # and copy the dependencies into the csp-resources folder
    # Tools resources are copied into the tools-resources folder
    download_web_dependencies "$WRAPPER_RESOURCES_DIR" "True"
  else
    log_info "Building in non-fat mode"
    # This will just copy the tools jar built from source into the tools-resources folder
    download_web_dependencies "$WRAPPER_RESOURCES_DIR" "False"
  fi
  # Copy qualOutputTable.yaml from core module
  copy_reports_from_core "$WRAPPER_RESOURCES_DIR"
  # Builds the python wheel file
  # Look into the pyproject.toml file for the build system requirements
  python -m build --wheel
  clean_up_build_jars
}


# Main Script execution starts here
echo
log_msg "$CYAN" " Product                   : ${PRODUCT_NAME}"
log_msg "$CYAN" " Build Mode                : ${build_mode}"
log_msg "$CYAN" " Build Dir                 : ${BUILD_DIR}"
log_msg "$CYAN" " Jar URL                   : ${jar_url}"
log_msg "$CYAN" " Wrapper ResourceS Dir     : ${WRAPPER_RESOURCES_DIR}"
echo

if [ "$SKIP_WHEEL_PACKAGE_BUILD" = "1" ]; then
  log_msg "$BLUE" " Running in SKIP_WHEEL_PACKAGE_BUILD mode (no wheel build)"
  prepare_resources
  build_exit_code=$?
else
  pre_build
  build
  build_exit_code=$?
fi
if [ $build_exit_code -eq 0 ]; then
  echo
  if [ "$SKIP_WHEEL_PACKAGE_BUILD" = "1" ]; then
    log_msg "$UNDER" "Resources prepared successfully. Wheel build was skipped (SKIP_WHEEL_PACKAGE_BUILD=1)."
  else
    log_msg "$UNDER" "Build successful. To install, use: pip install dist/<wheel-file>"
  fi
  echo
  log_msg "$GREEN" " Product                   : ${PRODUCT_NAME}"
  log_msg "$GREEN" " Jar URL                   : ${jar_url}"
  log_msg "$GREEN" " Build Mode                : ${build_mode}"
  log_msg "$GREEN" " Tools Jar                 : ${TOOLS_JAR_FILE}"
  log_msg "$GREEN" " Wrapper generated reports : ${DEST_CORE_REPORTS_REL_PATH}"
  echo
else
  bail "Build failed."
fi
