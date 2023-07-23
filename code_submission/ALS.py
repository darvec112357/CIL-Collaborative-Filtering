import numpy as np
import pandas as pd
from utils import *

class ALS:
    def __init__(self, A, A_mask, num_iterations = 10, lambda_ = 0.15, k = 3):
        A, self.col_mean, self.col_std = standard_norm(A)
        A = A.fillna(0).to_numpy()
        self.A = A
        self.A_mask = A_mask.to_numpy()
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.k = k

    def fit(self):
            print("Running ALS")
            n, m = self.A.shape
            U, s, Vt = np.linalg.svd(self.A, full_matrices=False)
            U = np.copy(U[:,:self.k])
            V = np.copy(Vt[:self.k,:])
            for iter in range(self.num_iterations):
                print("Starting {}th Iterations".format(iter))
                for i, wi in enumerate(self.A_mask):
                    temp1 = V@(wi[:,None] * V.T) + self.lambda_ * np.eye(self.k)
                    temp2 = np.dot(V, np.dot(np.diag(wi), self.A[i].T))
                    U[i] = np.linalg.solve(temp1, temp2).T
                error = score(predict(np.dot(U,V),self.col_mean,self.col_std),test_df)
                print("Error after optimizing U: ", error)

                for j, wj in enumerate(self.A_mask.T):
                    temp1 = U.T@(wj[:,None] * U) + self.lambda_ * np.eye(self.k)
                    temp2 = U.T@(wj * self.A[:, j])
                    V[:,j] = np.linalg.solve(temp1,temp2)
                error = score(predict(np.dot(U,V),self.col_mean,self.col_std),test_df)
                print("Error after optimizing V: ", error)
            return U@V
    

if __name__ == '__main__':
    full_matrix = pd.read_csv('full_matrix.csv').to_numpy()
    test_df = pd.read_csv('test_df.csv')
    A_train = pd.read_csv('train_matrix.csv')
    A_mask = ~np.isnan(A_train)
    mask = ~np.isnan(full_matrix)
    als = ALS(A_train, A_mask)
    X = als.fit()
    X = predict(X,als.col_mean,als.col_std)
    print('RMSE: {}'.format(score(X,test_df)))
    X[mask] = full_matrix[mask]
    pd.DataFrame(X).to_csv('predictions/ALS.csv', index = False)
    store_dense_matrix_to_submission('sampleSubmission.csv', 'submission.csv', X, clip_min=1, clip_max=5)
