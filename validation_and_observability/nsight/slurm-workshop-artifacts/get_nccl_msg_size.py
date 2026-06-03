import argparse
import subprocess
import sqlite3
import pandas as pd
import ast

def replace_id_with_value(main_df, str_df, id_column, value_col_name=None):
    """Replace the values in 'id_column' of 'main_df' with the corresponding
    string value stored in 'str_df'.

    Parameters
    ----------
    main_df : dataframe
        Dataframe containing 'id_column'.
    str_df : dataframe
        Dataframe 'StringId' that maps IDs to string values.
    id_column : str
        Name of the column that should be replaced with the corresponding
        string values.
    value_col_name : str, optional
        Name of the column that contains the string value of 'id_column'.
        If not specified, the 'id_column' will be retained as the column name.
    """
    renamed_str_df = str_df.rename(columns={"id": id_column})
    merged_df = main_df.merge(renamed_str_df, on=id_column, how="left")

    # Drop the original 'id_column' column.
    merged_df = merged_df.drop(columns=[id_column])
    # Rename the 'value' column.
    value_col_name = value_col_name or id_column
    return merged_df.rename(columns={"value": value_col_name})

def combine_text_fields(nvtx_df, str_df):
    """Combine the 'text' and 'textId' fields of the NVTX dataframe.

    This function simplifies the lookup process for accessing the event
    message. The 'text' field corresponds to the NVTX event message passed
    through 'nvtxDomainRegisterString', while the 'textId' field corresponds
    to the other case. By merging these fields, we streamline the process of
    retrieving the message.
    """
    if not nvtx_df["textId"].notnull().any():
        return nvtx_df.copy()

    nvtx_textId_df = replace_id_with_value(
        nvtx_df, str_df, "textId", "textStr"
    )
    mask = ~nvtx_textId_df["textStr"].isna()
    nvtx_textId_df.loc[mask, "text"] = nvtx_textId_df.loc[mask, "textStr"]
    return nvtx_textId_df.drop(columns=["textStr"])


# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', type=str, help='Nsight Report Name')

args = parser.parse_args()

subprocess.run(["/fsxl/nsight-efa/target-linux-x64/nsys", "export",
                "--type", "sqlite",
                "--force-overwrite", "true",
                "--include-blobs", "true",
                "--include-json", "true",
                args.name ])

sqlite_file_name = args.name.split('.')[0]+'.sqlite'

print(sqlite_file_name)

conn = sqlite3.connect(sqlite_file_name)

nvtx_df = pd.read_sql_query("SELECT * FROM NVTX_EVENTS", conn)

str_df = pd.read_sql_query("SELECT * FROM StringIds", conn)

df = combine_text_fields(nvtx_df, str_df)

nccl_msg_size_df = df.loc[df['eventType']==59,['text','jsonText']].drop_duplicates()

#print(nccl_msg_size_df)

nccl_operations = ['ncclAllReduce','ncclAllGather','ncclReduceScatter','ncclBroadcast']

final_df = pd.DataFrame(columns=['NCCL Operation','Message Size Bytes', 'Reduction Operation'])
rows_list = []
for one_operation in nccl_operations:
    one_df = nccl_msg_size_df.loc[nccl_msg_size_df['text']==one_operation,]

    for index, row in one_df.iterrows():
        tmp_dict =  ast.literal_eval(row['jsonText'])

        if one_operation == 'ncclAllGather':
            tmp_dict['Reduction operation'] = 'None'

        if one_operation == 'ncclBroadcast':
            tmp_dict['Reduction operation'] = 'None'
            tmp_dict['Message size [bytes]'] = tmp_dict['Bytes']

        new_row = {'NCCL Operation': one_operation,
                   'Message Size Bytes': tmp_dict['Message size [bytes]'],
                   'Reduction Operation': tmp_dict['Reduction operation']
                   }
        rows_list.append(new_row)

final_df = pd.DataFrame(rows_list)

print(final_df)

conn.close()