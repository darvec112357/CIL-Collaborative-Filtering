import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

def _read_df_in_format(root):
    def reformat_id(id):
        row, col = id.split('_')
        return int(row[1:]), int(col[1:])  
    df = pd.read_csv(root)
    df['row'], df['col'] = zip(*df['Id'].apply(reformat_id))
    df = df.drop(columns=['Id'], axis=1)
    return df

def convert_df_to_matrix(df, n_row=10000, n_col=1000):
    row_id = df['row'].to_numpy() - 1
    col_id = df['col'].to_numpy() - 1

    data_matrix = np.zeros((n_row, n_col), dtype=np.int8)
    data_matrix[row_id, col_id] = df['Prediction'].to_numpy()
    return pd.DataFrame(data_matrix)

def get_train_test(root, split_num=0):
    df = _read_df_in_format(root)
    df = df[['row', 'col', 'Prediction']]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(kf.split(df))[split_num]
    # print(train_idx)
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    train_matrix = convert_df_to_matrix(train_df)
    train_matrix.replace(0, np.nan, inplace=True)
    train_matrix.to_csv('train_matrix.csv',index = False)
    test_df.to_csv('test_df.csv',index = False)
    return train_df, test_df

"""
Usage:
train_df, test_df = get_train_test('data/data_train.csv', split_num=0)
train_matrix = convert_df_to_matrix(train_df)

NOTE: The ids are 1-indexed (rows go from 1 to 10000, cols go from 1 to 1000)
"""

get_train_test('data_train.csv', split_num=0)
