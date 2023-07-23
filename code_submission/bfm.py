import numpy as np
import pandas as pd 

from sklearn.preprocessing import OneHotEncoder
from surprise import KNNBaseline
from myfm import MyFMRegressor, MyFMOrderedProbit, VariationalFMRegressor

def run_bfm(train_df, test_df, rank, fm_kind='classifier'):
    """
    Run Bayesian Factorization Machine on the train and predict on the test

    Parameters
    ----------
    train_df : pandas DataFrame
        The training data, must have columns 'row', 'col', and 'Prediction'
    test_df : pandas DataFrame
        The testing data, must have columns 'row' and 'col'
    rank : int
        The rank of the factorization
    fm_kind : str
        The kind of FM to run, either 'classifier' or 'regressor'

    Returns
    -------
    expected_rating : numpy array
        The expected rating for each row in test_df
    fm : MyFMRegressor or MyFMOrderedProbit
        The trained FM model
    """
    if fm_kind.lower() == 'classifier':
        fm = MyFMOrderedProbit(rank=rank, random_seed=42)
    elif fm_kind.lower() == 'regressor':
        fm = MyFMRegressor(rank=rank, random_seed=42)
    else:
        raise ValueError(f"fm_kind must be either 'classifier' or 'regressor', got {fm_kind}")
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train = ohe.fit_transform(train_df[['row', 'col']]) # makes it into sparse matrix of number of ratings x (number of users + number of items)
    y_train = train_df['Prediction']

    X_test = ohe.transform(test_df[['row', 'col']])

    fm.fit(X_train, y_train-1, n_iter=200, n_kept_samples=200, group_shapes=[len(group) for group in ohe.categories_])    
    if fm_kind.lower() == 'classifier':
        y_pred = fm.predict_proba(X_test)
        expected_rating = y_pred.dot(np.arange(1, 6))
    elif fm_kind.lower() == 'regressor':
        y_pred = fm.predict(X_test) + 1
        expected_rating = np.clip(y_pred, 1, 5)

    return expected_rating, fm

def generate_clusters(trainset, anti_trainset, n_clusters=30):
    """
    Utilises KNNBaseline to generate clusters for all ratings

    Parameters
    ----------
    trainset : surprise Trainset
        The training data
    anti_trainset : surprise AntiTrainset
        The anti-training data
    n_clusters : int
        The number of clusters to generate

    Returns
    ------- 
    anti_pred_df : pandas DataFrame
        The anti-training data with the predicted ratings and the cluster number
    """
    knn = KNNBaseline(k=n_clusters, sim_options={'name': 'pearson_baseline', 'user_based': False})
    knn.fit(trainset)
    anti_preds = knn.test(anti_trainset)
    anti_pred_df = pd.DataFrame(map(lambda x: (x.uid, x.iid, x.est, x.details['actual_k']), anti_preds), columns=['row', 'col', 'Prediction', 'cluster'])
    return anti_pred_df

def create_augmented_dataset(train_df, antitrain_df, n_samples_per_cluster, seed=42):
    """
    Creates an augmented dataset by sampling n_samples_per_cluster from each cluster in antitrain_df

    Parameters
    ----------
    train_df : pandas DataFrame
        The training data, must have columns 'row', 'col', and 'Prediction'
    antitrain_df : pandas DataFrame
        The anti-training data, must have columns 'row', 'col', 'Prediction', and 'cluster'
    n_samples_per_cluster : int
        The number of samples to take from each cluster
    seed : int
        The random seed to use
    
    Returns
    ------- 
    augmented_train_df : pandas DataFrame
        The augmented training data
    """
    n_clusters = antitrain_df.cluster.nunique()

    added_rows = []

    for i in range(n_clusters):
        cluster_filtered = antitrain_df[antitrain_df.cluster == i]
        if len(cluster_filtered) < n_samples_per_cluster:
            added_rows.append(cluster_filtered)
        else:
            added_rows.append(antitrain_df[antitrain_df.cluster == i].sample(n=n_samples_per_cluster, random_state=seed))
    added_rows = pd.concat(added_rows, ignore_index=True)

    return pd.concat([train_df, added_rows], ignore_index=True)

def run_bfm_augmented(train_df, antitrain_df, test_df, n_samples_per_cluster, rank, seed_lst = [42]):
    """
    Run Bayesian Factorization Machine on the train and predict on the test. The training data is augmented with the anti-training data.
    To account for randomness, the training is run multiple times with different random seeds and the predictions are averaged.

    Parameters
    ----------
    train_df : pandas DataFrame
        The training data, must have columns 'row', 'col', and 'Prediction'
    antitrain_df : pandas DataFrame
        The anti-training data, must have columns 'row', 'col', 'Prediction', and 'cluster'
    test_df : pandas DataFrame
        The testing data, must have columns 'row' and 'col'
    n_samples_per_cluster : int
        The number of samples to take from each cluster
    rank : int
        The rank of the factorization
    seed_lst : list
        The list of random seeds to use
        
    Returns
    -------
    pred_df : pandas DataFrame
        The testing data with the predictions
    """
    pred_df = test_df.reset_index(drop=True)
    for seed in seed_lst:
        augmented_train_df = create_augmented_dataset(train_df, antitrain_df, n_samples_per_cluster, seed=seed)
        expected_rating, _ = run_bfm(augmented_train_df, pred_df, rank, fm_kind='regressor')
        pred_df[f'Prediction_{seed}'] = expected_rating
    pred_df['Prediction_avg'] = pred_df[[f'Prediction_{seed}' for seed in seed_lst]].mean(axis=1)
    return pred_df