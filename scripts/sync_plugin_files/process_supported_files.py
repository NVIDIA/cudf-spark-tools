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

"""
Description:
    This Python script goes over all directories of different spark versions under the
    spark-rapids generated_files_path folder, constructs unions of the supported CSV
    files (supportedDataSource.csv, supportedExecs.csv, supportedExprs.csv), overrides
    any custom configurations, generates a report for the changes between spark-rapids
    and tools supported CSV files, and writes the final results to new CSV files.

Dependencies:
    - numpy >= 1.23.3
    - pandas >= 2.0.3

Usage:
    python process_supported_files.py generated_files_path [--configs configs_file] \
        [--output output_directory] [--tools-csv tools_csv_dir]
"""

import argparse
import json
import logging
import os
from enum import IntEnum

import pandas as pd


class SupportLevel(IntEnum):
    """
    IntEnum Class for support level types to facilitate easy comparison.

    Enum members:
    - NA: Not Applicable.
    - NS: Not Supported.
    - CO: Configured Off.
    - PS: Partially Supported.
    - S: Supported.
    - TOFF: Tools Off (Explicitly disabled in tools).
    - TNEW: Tools New (New ops from plugin which have not been tested in tools).
    - TON: Tools On (Explicitly enabled in tools).
    """
    NA = 1
    NS = 2
    CO = 3
    PS = 4
    S = 5
    TOFF = 6
    TNEW = 7
    TON = 8


def is_support_level(elem):
    """
    Check if elem is a valid name of a SupportLevel member.
    """
    return elem in SupportLevel.__members__


def is_greater(elem1, elem2):
    """
    Compare two entries, which might be names of SupportLevel members or strings.

    1) If they are both valid names of SupportLevel members,
       compare using their integer values.
    2) If not, compare their length (assuming strings with greater length contain more info).
    """
    if is_support_level(elem1) and is_support_level(elem2):
        return SupportLevel[elem1] > SupportLevel[elem2]

    return len(elem1) > len(elem2)


def get_sql_func_aliases(sql_func_value):
    """
    Return SQL function aliases from a semicolon-separated SQL Func cell.
    """
    if pd.isna(sql_func_value):
        return set()
    sql_func_str = str(sql_func_value).strip()
    if sql_func_str == "" or sql_func_str.lower() in ("none", "nan"):
        return set()
    return {
        alias.strip() for alias in sql_func_str.split(";")
        if alias.strip() and alias.strip() not in SupportLevel.__members__
    }


def should_preserve_tools_string(column_name, tools_value, plugin_value):
    """
    Preserve tools-side expression aliases when plugin generated metadata is less informative.
    SQL function aliases are parser lookup keys in tools, and the generated plugin CSV can omit
    aliases for newer Spark versions even though older-version aliases should still be recognized.
    Preserve only when the plugin aliases are blank or a strict subset of the tools aliases, so an
    intentional plugin alias rename is accepted instead of being hidden by a string-length heuristic.
    """
    if column_name != "SQL Func":
        return False

    tools_aliases = get_sql_func_aliases(tools_value)
    plugin_aliases = get_sql_func_aliases(plugin_value)
    if not tools_aliases:
        return False
    return not plugin_aliases or plugin_aliases < tools_aliases


def check_df_rows(row1, row2, keys):
    """
    Given two DataFrame rows (pandas Series) and a list of keys (column names),
    check if the rows have the same values for all specified keys.
    """
    for key in keys:
        if row1[key] != row2[key]:
            return False
    return True


def unify_all_files(root_dir, file_name, key_names):
    """
    Search for file_name in all spark-version folders and unify the CSV files into a
    single Dataframe, using the following rules:
    1) If a data source/exec/expr exists in any spark version, include in the final output.
    2) For each value in a row, escalate the support level: CO > S > PS > NA > NS.

    Parameters:
    - root_dir: The directory containing folders of CSV data for every spark version.
                > root_dir
                    > 311
                    ...
                    > 350
                        > supportedExecs.csv
                        > supportedExprs.csv
                        ...
    - file_name: CSV file to process.
    - key_names: List of column names which uniquely identifies a row in the CSV file.
    """
    final_df = pd.DataFrame()

    for dir_name in os.listdir(root_dir):  # List entries in root_dir
        if os.path.isdir(os.path.join(root_dir, dir_name)):
            csv_file_path = os.path.join(root_dir, dir_name, file_name)
            try:
                cur_df = pd.read_csv(csv_file_path, keep_default_na=False)
            except Exception as e:
                logging.debug(f"Encountered exception when reading file {csv_file_path}: {e}")
                continue

            if final_df.empty:
                final_df = pd.DataFrame(columns=cur_df.columns.tolist())

            # expand final_df if cur_df has more columns
            for col_name in cur_df.columns:
                if col_name not in final_df.columns:
                    logging.debug(f"Expanding final_df with new column name: {col_name}")
                    final_df[col_name] = ["NS" for _ in range(final_df.shape[0])]

            # iterate through every row in the current df and update the final df correspondingly
            for _, cur_row in cur_df.iterrows():
                # expand current_row if final_df has more columns
                for col_name in final_df.columns:
                    if col_name not in cur_row:
                        logging.debug(f"Expanding cur_row with entry: ({col_name}, NS)")
                        cur_row.loc[col_name] = "NS"

                updated_flag = False
                for final_idx, final_row in final_df.iterrows():
                    # current row exists in final df, unify them and update final df
                    if check_df_rows(cur_row, final_row, key_names):
                        updated_flag = True
                        for col_name in final_df.columns:
                            # if cur_row[col_name] has higher support level, update final_row[col_name]
                            if (col_name not in key_names) and is_greater(cur_row[col_name], final_row[col_name]):
                                logging.debug(f"Updating final_df at ({final_idx}, {col_name}) = {cur_row[col_name]}")
                                final_df.at[final_idx, col_name] = cur_row[col_name]
                # current_row does not exist in final df, append it
                if not updated_flag:
                    logging.debug("Appending row to final_df: {cur_row.values}")
                    final_df = pd.concat([final_df, cur_row.to_frame().T], ignore_index=True)

    logging.debug(f"final_df = {final_df}")
    return final_df


def check_override(entry, row_data, keys):
    """
    Given a JSON entry and a DataFrame row (pandas Series) and a list of keys,
    check if the entry and row have the same values for all specified keys.
    """
    for key in keys:
        if not entry[key] == row_data[key]:
            return False
    return True


def override_supported_configs(json_data, file_name, df, keys):
    """
    Override dataframe df with input json_data.

    Parameters:
    - json_data: Data in JSON format to explicitly override input DataFrame.
    - file_name: Key within json_data containing override information.
    - df: Pandas DataFrame to be modified.
    - keys: A list of column names which identifies a row in df.

    Returns:
    - The modified DataFrame.

    Example JSON input:
    json_data = {
        "supportedExprs.csv": [
            {
                "Expression": "PromotePrecision",
                "Context": "project",
                "Params": "input",
                "override": [{'key': 'SQL Func', 'value': "`promote_precision`"}]
            }
        ]
    }
    """
    if file_name in json_data:
        for entry in json_data[file_name]:
            for idx, row_data in df.iterrows():
                if check_override(entry, row_data, keys):
                    for config in entry["override"]:
                        df.at[idx, config["key"]] = config["value"]
    return df


def compare_csv_file(union_df, tools_df, keys, report_file, override_configs_json, csv_file_name,
                     new_operators_file=None):
    """
    Compare the plugin union dataframe with tools dataframe and write the differences to report file.
    For added rows in the plugin union Dataframe, update the 'Supported' or first support level column
    to 'TNEW' for further testing.

    Parameters:
    - union_df: Pandas DataFrame which represents the union from plugin CSV files.
    - tools_df: Pandas DataFrame which represents the tools CSV file.
    - keys: A list of column names which identifies a row in the Dataframes.
    - report_file: A file to write the differences into.
    - override_configs_json: A json object which stores the override configs information.
    - csv_file_name: File name to identify which supported CSV type is processing.
    - new_operators_file: File to write the new operators in the plugin CSV files.

    Returns:
    - Modified union_df which may contain 'TNEW' entries.
    """

    # pre-process union dataframe to resolve any inconsistencies
    for union_idx, union_row in union_df.iterrows():
        if "Supported" in union_row and union_row["Supported"] == "S" and union_row["Notes"] != "None":
            union_df.at[union_idx, "Notes"] = "None"

    # added columns in plugin union dataframe (compared with tools)
    for union_column_name in union_df.columns:
        if union_column_name not in tools_df.columns:
            report_file.write(f"Column is added: \"{union_column_name}\"\n")

    # removed columns in plugin union dataframe (compared with tools)
    for tools_column_name in tools_df.columns:
        if tools_column_name not in union_df.columns:
            report_file.write(f"Column is removed: \"{tools_column_name}\"\n")

    # added/changed rows in plugin union dataframe (compared with tools)
    for union_idx, union_row in union_df.iterrows():
        exists_in_tools = False
        for _, tools_row in tools_df.iterrows():
            if check_df_rows(union_row, tools_row, keys):
                exists_in_tools = True
                for tools_column in tools_df.columns:
                    if (tools_column in union_row) and (not tools_row[tools_column] == union_row[tools_column]):
                        if should_preserve_tools_string(
                                tools_column, tools_row[tools_column], union_row[tools_column]):
                            report_file.write(
                                f"Row keeps tools value: {', '.join(tools_row.astype(str))}\n    " +
                                f"{tools_column}: preserving {tools_row[tools_column]} instead of " +
                                f"{union_row[tools_column]}\n")
                            union_df.at[union_idx, tools_column] = tools_row[tools_column]
                        else:
                            report_file.write(f"Row is changed: {', '.join(tools_row.astype(str))}\n    " +
                                              f"{tools_column}: {tools_row[tools_column]} -> " +
                                              f"{union_row[tools_column]}\n")
        if not exists_in_tools:
            for column_name in union_df.columns:
                if is_support_level(union_row[column_name]):
                    union_df.at[union_idx, column_name] = "TNEW"
                    # update override configs json object for new ops
                    json_entry = {}
                    for key in keys:
                        json_entry[key] = union_row[key]
                    json_entry["override"] = [{"key": column_name, "value": "TNEW"}]
                    if csv_file_name in override_configs_json:
                        json_entry_exists = False
                        for entry in override_configs_json[csv_file_name]:
                            if entry == json_entry:
                                json_entry_exists = True
                                break
                        if not json_entry_exists:
                            override_configs_json[csv_file_name].append(json_entry)
                    else:
                        override_configs_json[csv_file_name] = [json_entry]
                    break
            new_operator = ""
            if "Exec" in union_row:
                new_operator = union_row["Exec"]
            elif "Expression" in union_row:
                new_operator = union_row["Expression"]
            if new_operator != "":
                new_operators_file.write(f"{new_operator}\n")
            report_file.write(f"Row is added: {', '.join(union_row.astype(str))}\n")

    # removed rows in plugin union dataframe (compared with tools)
    for _, tools_row in tools_df.iterrows():
        exists_in_union = False
        for _, union_row in union_df.iterrows():
            if check_df_rows(tools_row, union_row, keys):
                exists_in_union = True
                break
        if not exists_in_union:
            # append removed tools_row to union_df to preserve in final output
            union_df.loc[len(union_df)] = tools_row
            report_file.write(f"Row is removed: {', '.join(tools_row.astype(str))}\n")

    return union_df


def main(argvs):
    """
    Main function of the script.

    Parameters:
    args: Namespace containing the command-line arguments
    """

    generated_files_dir = argvs.path
    override_configs_file = argvs.configs
    output_dir = argvs.output
    tools_csv_dir = argvs.tools_csv

    # load override configs into a json object
    if override_configs_file:
        with open(override_configs_file, "r") as f:
            override_configs_json = json.load(f)
    else:
        override_configs_json = {}

    # generate the union of supported files as pandas dataframes
    logging.info("Generating the union of plugin supportedDataSource.csv files")
    data_source_union_df = unify_all_files(generated_files_dir, "supportedDataSource.csv", ["Format", "Direction"])
    logging.info("Generating the union of plugin supportedExecs.csv files")
    execs_union_df = unify_all_files(generated_files_dir, "supportedExecs.csv", ["Exec", "Params"])
    logging.info("Generating the union of plugin supportedExprs.csv files")
    exprs_union_df = unify_all_files(generated_files_dir, "supportedExprs.csv", ["Expression", "Context", "Params"])

    # post-process the dataframes to override custom configs
    logging.info("Post-processing supportedDataSource union dataframe to override custom configs")
    data_source_union_df = override_supported_configs(override_configs_json, "supportedDataSource.csv",
                                                      data_source_union_df, ["Format", "Direction"])
    logging.info("Post-processing supportedExecs union dataframe to override custom configs")
    execs_union_df = override_supported_configs(override_configs_json, "supportedExecs.csv", execs_union_df,
                                                ["Exec", "Params"])
    logging.info("Post-processing supportedExprs union dataframe to override custom configs")
    exprs_union_df = override_supported_configs(override_configs_json, "supportedExprs.csv", exprs_union_df,
                                                ["Expression", "Context", "Params"])

    # generate report for changes from tools existing CSV files to those in plugin
    logging.info("Generating report for this sync up process")
    report_file = open("operators_plugin_sync_report.txt", "w+")
    report_file.write("""This report documents the differences between the tools existing CSV files and those processed from the plugin.
    Notes:
      1. For new data source/exec/expression from plugin, the first column with supported level will be updated to 'TNEW' for future testing.
      2. Rows marked as "is removed" will be preserved in the final output.
      3. The "Notes" column for rows with "S" for "Supported" will be updated to "None" in the final output.\n\n""")

    if not tools_csv_dir:
        data_source_final_df = data_source_union_df
        execs_final_df = execs_union_df
        exprs_final_df = exprs_union_df
        report_file.write("Report is not generated: no input tools CSV directory.")
    else:
        # Construct a file to store all new Exec/Expressions from plugin CSV
        new_operators_file = open("new_operators.txt", "w+")

        logging.info("Writing report for supportedDataSource.csv")
        tools_data_source_file = os.path.join(tools_csv_dir, "supportedDataSource.csv")
        if os.path.exists(tools_data_source_file):
            report_file.write("\n**supportedDataSource.csv (FROM TOOLS TO PLUGIN)**\n")
            tools_data_source_df = pd.read_csv(tools_data_source_file, keep_default_na=False)
            data_source_final_df = compare_csv_file(data_source_union_df, tools_data_source_df, ["Format", "Direction"],
                                                    report_file, override_configs_json, "supportedDataSource.csv")

        logging.info("Writing report for supportedExecs.csv")
        tools_execs_file = os.path.join(tools_csv_dir, "supportedExecs.csv")
        if os.path.exists(tools_execs_file):
            report_file.write("\n**supportedExecs.csv (FROM TOOLS TO PLUGIN)**\n")
            tools_execs_df = pd.read_csv(tools_execs_file, keep_default_na=False)
            execs_final_df = compare_csv_file(execs_union_df, tools_execs_df, ["Exec", "Params"], report_file,
                                              override_configs_json, "supportedExecs.csv", new_operators_file)

        logging.info("Writing report for supportedExprs.csv")
        tools_exprs_file = os.path.join(tools_csv_dir, "supportedExprs.csv")
        if os.path.exists(tools_exprs_file):
            report_file.write("\n**supportedExprs.csv (FROM TOOLS TO PLUGIN)**\n")
            tools_exprs_df = pd.read_csv(tools_exprs_file, keep_default_na=False)
            exprs_final_df = compare_csv_file(exprs_union_df, tools_exprs_df, ["Expression", "Context", "Params"],
                                              report_file, override_configs_json, "supportedExprs.csv",
                                              new_operators_file)
        new_operators_file.close()

    report_file.close()

    # write the final result dataframes to output CSV files
    logging.info(f"Writing the final result dataframes to output directory: {output_dir}")
    data_source_final_df.to_csv(f"{output_dir}/supportedDataSource.csv", index=False)
    execs_final_df.to_csv(f"{output_dir}/supportedExecs.csv", index=False)
    exprs_final_df.to_csv(f"{output_dir}/supportedExprs.csv", index=False)

    # write the processed override json data to input configs file
    logging.info("Writing the processed override json data to input configs file")
    with open(override_configs_file, "w") as f:
        json.dump(override_configs_json, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to generated_files directory.")
    parser.add_argument("--configs", type=str, help="Path to configs file for overriding current data.")
    parser.add_argument("--output", type=str, help="Path to output directory.", default=".")
    parser.add_argument("--tools-csv", type=str,
                        help="Path to directory which contains the original CSV files in the tools repo.")

    args = parser.parse_args()

    main(args)
