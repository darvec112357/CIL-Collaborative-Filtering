import pandas as pd
import numpy as np
import optuna

from surprise import KNNWithMeans,SVD,NMF,SlopeOne,BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

from train_test import get_train_test
from sklearn.metrics import mean_squared_error
from bfm import  get_RelationBlocks

from sklearn.linear_model import ElasticNet
import sys

import myfm



def objectiveKNNWithMeans(trial):
    # Define hyperparameters
    k = trial.suggest_int("k", 5, 300, 5)
    min_k = trial.suggest_int("min_k", 1, 300, 5)
    name = trial.suggest_categorical("name", ["cosine", "msd", "pearson", "pearson_baseline"])
    user_based = trial.suggest_categorical("user_based", [True, False])
    sim_options = {'name': name, 'user_based': user_based}


    algo = KNNWithMeans(k = k, min_k= min_k, sim_options=sim_options, verbose= False)
    # Perform 5-fold cross-validation and compute RMSE
    cross_val_results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)

    # Return mean RMSE
    return cross_val_results['test_rmse'].mean()


def objectiveBaselineOnly(trial):
    # Define hyperparameters
    bsl_options = {'method': "als",
                   'reg_i': trial.suggest_float("reg_i", 0, 300),
                   'reg_u': trial.suggest_float("reg_u", 0, 300),
                   'n_epochs': trial.suggest_int("n_epochs", 2, 200, 10)}


    algo = BaselineOnly(bsl_options=bsl_options)

    # Perform 5-fold cross-validation and compute RMSE
    cross_val_results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)

    # Return mean RMSE
    return cross_val_results['test_rmse'].mean()


def objectiveSVD(trial):
    # Define hyperparameters
    n_factors = trial.suggest_int("n_factors", 1, 50, 10)
    n_epochs = trial.suggest_int("n_epochs", 1, 300, 10)
    lr_all = trial.suggest_loguniform("lr_all", 1e-10, 1)
    reg_all = trial.suggest_loguniform("reg_all", 1e-10, 20)

    # Train SVD algorithm with specified hyperparameters
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)


    # Perform 5-fold cross-validation and compute RMSE
    cross_val_results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)

    # Return mean RMSE
    return cross_val_results['test_rmse'].mean()


def objectiveNMF(trial):
    n_factors = trial.suggest_int("n_factors", 10, 50, 10)
    n_epochs = trial.suggest_int("n_epochs", 5, 300, 10)
    reg_pu = trial.suggest_loguniform("lr_all", 1e-10, 20)
    reg_qi = trial.suggest_loguniform("reg_all", 1e-10, 20)
    biased = trial.suggest_categorical("biased", [True, False])

    algo = NMF(n_factors=n_factors, n_epochs=n_epochs, reg_pu=reg_pu, reg_qi=reg_qi, biased = biased)

    # Perform 5-fold cross-validation and compute RMSE
    cross_val_results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)

    # Return mean RMSE
    return cross_val_results['test_rmse'].mean()


def objective_bfm_OrderProbit_6(trial):
    rank = trial.suggest_int("rank", 1, 100, 1)
    n_iter = trial.suggest_int("n_iter", 1, 1000, 1)
    n_kept_samples = trial.suggest_int("n_kept_samples", 1, n_iter, 1)

    algo = myfm.MyFMOrderedProbit(rank=rank)
    algo.fit(
    None, train_df.Prediction.values, X_rel=train_blocks,
    group_shapes=feature_group_sizes,
    n_iter=n_iter, n_kept_samples=n_kept_samples);

    prediction = algo.predict_proba(None, test_blocks)
    prediction = prediction.dot(np.arange(6)) 

    return (np.sqrt(mean_squared_error(prediction, test_df.Prediction)))

def objective_bfm_OrderProbit_5(trial):
    rank = trial.suggest_int("rank", 1, 100, 1)
    n_iter = trial.suggest_int("n_iter", 1, 1000, 1)
    n_kept_samples = trial.suggest_int("n_kept_samples", 1, n_iter, 1)

    algo = myfm.MyFMOrderedProbit(rank=rank)
    algo.fit(
    None, train_df.Prediction.values -1, X_rel=train_blocks,
    group_shapes=feature_group_sizes,
    n_iter=n_iter, n_kept_samples=n_kept_samples);

    prediction = algo.predict_proba(None, test_blocks)
    prediction = prediction.dot(np.arange(5))  + 1

    return (np.sqrt(mean_squared_error(prediction, test_df.Prediction)))

def objective_bfm_variational(trial):
    rank = trial.suggest_int("rank", 1, 100, 1)
    n_iter = trial.suggest_int("n_iter", 1, 1000, 1)
   

    algo = myfm.VariationalFMRegressor(rank=rank)
    algo.fit(
    None, train_df.Prediction.values -1, X_rel=train_blocks,
    group_shapes=feature_group_sizes,
    n_iter=n_iter);

    prediction = algo.predict(None, test_blocks)

    return (np.sqrt(mean_squared_error(prediction, test_df.Prediction)))

def create_objective_ElasticNet(training,testing,train_rating,test_rating):
    
    def objective(trial):
        # Defining the parameters space for the ElasticNet model
        alpha = trial.suggest_float("alpha", 0.01, 2.0,log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0, 0.99999, step=0.00001)


        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0,max_iter = 10000)
        model.fit(training, train_rating)

        y_pred = model.predict(testing)
        y_pred = np.clip(y_pred, 1, 5)  
        rmse = (np.sqrt(mean_squared_error(test_rating, y_pred)))

        return rmse
    return objective


def runOpt(training,testing,train_rating,test_rating):
    objec = create_objective_ElasticNet(training,testing,train_rating,test_rating)
    study = optuna.create_study(direction='minimize')
    study.optimize(objec, n_trials=150)
    return study.best_value


def morePred(indexes,base_train,base_test,train_rating,test_rating,train_blend,test_blend):
    currentBestRsme = 100
    for i in indexes:
        new_base_train = pd.concat([base_train,train_blend.iloc[:,[i]]],axis=1)
        new_base_test = pd.concat([base_test,test_blend.iloc[:,[i]]],axis=1)
        rmse = runOpt(new_base_train,new_base_test,train_rating.Prediction,test_rating.Prediction)
        if rmse < currentBestRsme:
            currentBestRsme = rmse
            currentIndex = i
    return (currentBestRsme,currentIndex)




def optimize_blend(train_models,test_models,train_df,test_df):
    """
    This function performs forward selection to optimize the blend of models.
    It takes in the training and testing dataframes for the models, and returns
    the indexes of the models used in the blend and the best root mean squared error (RMSE) achieved.
    
    Args:
    train_models (pandas.DataFrame): The training dataframes for the models.
    test_models (pandas.DataFrame): The testing dataframes for the models.
    train_df (pandas.DataFrame): The training dataframe.
    test_df (pandas.DataFrame): The testing dataframe.
    
    Returns:
    tuple: A tuple containing the indexes of the models used in the blend and the best RMSE achieved.
    """
    indexes = list(range(2,train_blocks.size[1]))
    currentIndex = 1
    bestRsme = 10
    for i in indexes:

        if np.sqrt(mean_squared_error(test_df.Prediction,  test_models.iloc[:,[i]])) < bestRsme:
            bestRsme = rmse
            currentIndex = i
    usedIndex = [currentIndex]
    base_train = train_models.iloc[:,[currentIndex]]
    base_test = test_models.iloc[:,[currentIndex]]

    while len(indexes) > 0:
    
        rmse,currentIndex = morePred(indexes,base_train,base_test,train_df,test_df,train_models,test_models)
        if rmse < bestRsme:
            usedIndex.append(currentIndex)
            indexes.remove(currentIndex)
            base_train = pd.concat([base_train,train_models.iloc[:,[currentIndex]]],axis=1)
            base_test = pd.concat([base_test,test_models.iloc[:,[currentIndex]]],axis=1)
            bestRsme = rmse
        else:
            print("No more predictions")
            break
    return (usedIndex,bestRsme)    


args = sys.argv[1]
model_dict = {"KNNWithMeans":objectiveKNNWithMeans,"BaselineOnly":objectiveBaselineOnly,"SVD":objectiveSVD,"NMF":objectiveNMF,"bfm_OrderProbit_6":objective_bfm_OrderProbit_6,"bfm_OrderProbit_5":objective_bfm_OrderProbit_5,"bfm_variational":objective_bfm_variational}

if args in model_dict.keys():
    df = pd.read_csv('data_train.csv')

    # Split the 'ID' column into 'user' and 'item' columns
    df[['user', 'item']] = df['Id'].str.split('_', expand=True)

    # Remove 'r' and 'c' from 'user' and 'item' columns
    df['user'] = df['user'].str.replace('r', '')
    df['item'] = df['item'].str.replace('c', '')
    df = df.rename(columns={'Prediction': 'rating'})
    df = df.drop(['Id'], axis=1)

    # Using the surprise library's Reader class to parse the file
    reader = Reader(rating_scale=(1, 5))

    # Creating a Dataset object for surprise to use
    data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    if args in ["bfm_OrderProbit_6","bfm_OrderProbit_5","bfm_variational"]:
        train_df, test_df = get_train_test(('./data_train.csv'), split_num=0)
        train_blocks, test_blocks, feature_group_sizes = get_RelationBlocks(train_df, test_df)

    objective = model_dict[args]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50 )
    print("Best parameters: ", study.best_params)
    print("Best RMSE: ", study.best_value)
else:
    print("Model not found")
    print("Available models: KNNWithMeans,BaselineOnly,SVD,NMF,bfm_OrderProbit_6,bfm_OrderProbit_5,bfm_variational")

