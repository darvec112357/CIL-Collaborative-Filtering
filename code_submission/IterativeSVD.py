import numpy as np
import pandas as pd
from utils import *

class IterativeSVD:
    def __init__(self, A, A_mask, rank_svd = 10, num_iterations = 20, shrinkage = 50, eta = 0.2):
        A, self.col_mean, self.col_std = standard_norm(A)
        A = A.fillna(0).to_numpy()
        self.A = A
        self.A_mask = A_mask
        self.rank_svd = rank_svd
        self.num_iterations = num_iterations
        self.shrinkage = shrinkage
        self.eta = eta
        self.users = 10000
        self.movies = 1000
        self.test_df = pd.read_csv('test_df.csv')  
    
    def fit(self):
        X = np.zeros((self.users,self.movies))
        # X = self.A.copy()
        for i in range(self.num_iterations):
            print('Starting {}th iteration'.format(i))
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            s_ = (s - self.shrinkage).clip(min=0)
            shrinkA = (U*s_).dot(Vt)
            diff = (self.A - shrinkA)*self.A_mask
            X += self.eta * diff
        return shrinkA
    
if __name__ == '__main__':
    full_matrix = pd.read_csv('full_matrix.csv').to_numpy()
    test_df = pd.read_csv('test_df.csv')
    A_train = pd.read_csv('train_matrix.csv')
    A_mask = ~np.isnan(A_train)
    mask = ~np.isnan(full_matrix)
    svd = IterativeSVD(A_train, A_mask)
    X = svd.fit()
    X = predict(X,svd.col_mean,svd.col_std)
    print('RMSE: {}'.format(score(X,test_df)))
    X[mask] = full_matrix[mask]
    pd.DataFrame(X).to_csv('predictions/ISVD.csv', index = False)
    store_dense_matrix_to_submission('sampleSubmission.csv', 'submission.csv', X, clip_min=1, clip_max=5)
