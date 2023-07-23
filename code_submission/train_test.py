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
    df = df[['row', 'col', 'Prediction']]
    return df

def convert_df_to_matrix(df, n_row=10000, n_col=1000, dtype=np.int8):
    row_id = df['row'].to_numpy() - 1
    col_id = df['col'].to_numpy() - 1

    data_matrix = np.zeros((n_row, n_col), dtype=dtype)
    data_matrix[row_id, col_id] = df['Prediction'].to_numpy()
    return pd.DataFrame(data_matrix)

def get_train_test(root, split_num=0):
    df = _read_df_in_format(root)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = list(kf.split(df))[split_num]
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    return train_df, test_df

def generate_anti_train(matrix):
    return pd.DataFrame(np.array(np.where(matrix == 0)).T, columns=['row', 'col'])

def store_dense_matrix_to_submission(sub_sample_path, store_path, data_matrix, clip_min=1, clip_max=5):
    df = _read_df_in_format(sub_sample_path)

    row_id = df['row'].to_numpy() - 1
    col_id = df['col'].to_numpy() - 1
    data_matrix = np.clip(data_matrix, clip_min, clip_max)
    df['Prediction'] = data_matrix[row_id, col_id]

    def reformat_id(record):
        return f"r{record['row']:.0f}_c{record['col']:.0f}"
    
    df['Id'] = df.apply(reformat_id, axis=1)
    df = df.drop(columns=['row', 'col'], axis=1)
    df.to_csv(store_path, columns=['Id', 'Prediction'], index=False)
 
"""
Usage:
train_df, test_df = get_train_test('data/data_train.csv', split_num=0)
train_matrix = convert_df_to_matrix(train_df)

NOTE: The ids are 1-indexed (rows go from 1 to 10000, cols go from 1 to 1000)
"""
