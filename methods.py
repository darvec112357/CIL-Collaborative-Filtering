import numpy as np

from surprise import AlgoBase

from utils import de_norm

class UserAverage(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.global_mean = trainset.global_mean
        self.user_mean = dict() 
        for u, ratings in trainset.ur.items():
            self.user_mean[u] = np.mean([r for (_, r) in ratings])
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u)):
            return self.trainset.global_mean
        return self.user_mean[u]
    
class ItemAverage(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.global_mean = trainset.global_mean
        self.item_mean = dict() 
        for i, ratings in trainset.ir.items():
            self.item_mean[i] = np.mean([r for (_, r) in ratings])
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_item(i)):
            return self.trainset.global_mean
        return self.item_mean[i]
    
class UserItemAverage(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.global_mean = trainset.global_mean
        self.user_mean = dict() 
        for u, ratings in trainset.ur.items():
            self.user_mean[u] = np.mean([r for (_, r) in ratings])
        self.item_mean = dict() 
        for i, ratings in trainset.ir.items():
            self.item_mean[i] = np.mean([r for (_, r) in ratings])
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return self.trainset.global_mean
        if not (self.trainset.knows_user(u)):
            return self.item_mean[i]
        if not (self.trainset.knows_item(i)):
            return self.user_mean[u]
        return (self.user_mean[u] + self.item_mean[i]) / 2


class Baseline:
    """
    SVD / IterSVD / ALS
    """
    def __init__(self, rank_svd = 9, rank_als = 3, num_iterations = 20, lambda_als = 0.1):
        self.rank_svd = rank_svd
        self.rank_als = rank_als
        self.num_iterations = num_iterations
        self.lambda_als = lambda_als
        self.num_movies = 1000

    def SVD(self, A, num_movies, k=9):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

		# using the top k eigenvalues
        S = np.zeros((num_movies, num_movies))
        S[:k, :k] = np.diag(s[:k])

		# reconstruct matrix
        return U, S, Vt   
    
    def IterSVD(self, A, mask_A, shrinkage=38, n_itr=15):
        X = A.copy()
        for i in range(n_itr):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            s_ = (s - shrinkage).clip(min=0)
            X = U.dot(np.diag(s_)).dot(Vt)
            X[mask_A] = A[mask_A]
            print("%sth iteration is complete." % i, end='\r')

        return X

    def ALS(self, A, mask_A, k=3, n_itr=20, lambda_=0.1):
        print("Initializing ALS")
        n, m = A.shape
        U, _, Vt = self.SVD(A, self.num_movies, self.rank_svd) # Initialize U, Vt
        U = np.copy(U[:,:k])
        V = np.copy(Vt[:k,:])

        print("Starting Iterations")

        for iter in range(n_itr):
            print(f"Iteration {iter+1}", end='\t\t')
            for i, Ri in enumerate(mask_A):
                temp1 = V@(Ri[:,None] * V.T) + lambda_ * np.eye(k)
                temp2 = np.dot(V, np.dot(np.diag(Ri), A[i].T))
                U[i] = np.linalg.solve(temp1, temp2).T
            print("Error after solving for U matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A), end='\t\t')

            for j, Rj in enumerate(mask_A.T):
                temp1 = U.T@(Rj[:,None] * U) + lambda_ * np.eye(k)
                temp2 = U.T@(Rj * A[:, j])
                V[:,j] = np.linalg.solve(temp1,temp2)
            print("Error after solving for V matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A))
        return U, V
    
    def predict(self, U, V, col_mean, col_std):
        return de_norm(U.dot(V), col_mean, col_std)

class ALS:
    """
    ALS - Edited to combine the Iterative SVD initialization and ALS together into one function
    """
    def __init__(self):
        pass

    def _iterSVD(self, A, mask_A, shrinkage, n_iter):
        """
        Parameters
        ----------
        A : numpy array
            The matrix to be factorized, dimension (n_movies, n_users)
        mask_A : numpy array
            The mask of A, dimension (n_movies, n_users)
        shrinkage : int
            The shrinkage parameter for IterSVD
        n_iter : int
            The number of iterations for IterSVD

        Returns
        -------
        U : numpy array
            The left factor matrix, dimension (n_movies, k)
        V : numpy array
            The right factor matrix, dimension (k, n_users)
        """
        print("Initializing IterSVD")
        X = A.copy()
        for i in range(n_iter):
            U, s, V = np.linalg.svd(X, full_matrices=False)
            s_ = (s - shrinkage).clip(min=0)
            X = U.dot(np.diag(s_)).dot(V)
            X[mask_A] = A[mask_A]
            print(f"Iteration {i} complete", end='\r')

        print("IterSVD complete")
        return U, V
    
    def ALS(self, A, mask_A, k, shrinkage, lambd, n_iter_svd, n_iter_als):
        """
        Parameters
        ----------
        A : numpy array
            The matrix to be factorized, dimension (n_movies, n_users)
        mask_A : numpy array
            The mask of A, dimension (n_movies, n_users)
        k : int
            The rank of the factorization
        shrinkage : int
            The shrinkage parameter for IterSVD
        lambd : float
            The regularization parameter for ALS
        n_iter_svd : int
            The number of iterations for IterSVD
        n_iter_als : int
            The number of iterations for ALS

        Returns
        -------
        U : numpy array
            The left factor matrix, dimension (n_movies, k)
        V : numpy array
            The right factor matrix, dimension (k, n_users)
        """
        U, V = self._iterSVD(A, mask_A, shrinkage, n_iter_svd) # Initialize U, V using iterative SVD
        U = np.copy(U[:,:k])
        V = np.copy(V[:k,:])

        for iter in range(n_iter_als):
            print(f"Iteration {iter+1}", end='\t\t')
            for i, Ri in enumerate(mask_A):
                temp1 = V@(Ri[:, None] * V.T) + lambd * np.eye(k)
                temp2 = np.dot(V, np.dot(np.diag(Ri), A[i].T))
                U[i] = np.linalg.solve(temp1, temp2).T
            print("Error after solving for U matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A), end='\t\t')

            for j, Rj in enumerate(mask_A.T):
                temp1 = U.T@(Rj[:,None] * U) + lambd * np.eye(k)
                temp2 = U.T@(Rj * A[:, j])
                V[:,j] = np.linalg.solve(temp1,temp2)
            print("Error after solving for V matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A))
        return U, V
    
    def predict(self, U, V, col_mean, col_std):
        return de_norm(U.dot(V), col_mean, col_std)
    
class NMF(ALS):
    def NMF(self, A, mask_A, k, shrinkage, lambd, n_iter_svd, n_iter_als):
        """
        Self-defined NMF algorithm

        Parameters
        ----------
        A : numpy array
            The matrix to be factorized, dimension (n_movies, n_users)
        mask_A : numpy array
            The mask of A, dimension (n_movies, n_users)
        k : int
            The rank of the factorization
        shrinkage : int
            The shrinkage parameter for IterSVD
        lambd : float
            The regularization parameter for ALS
        n_iter_svd : int
            The number of iterations for IterSVD
        n_iter_als : int
            The number of iterations for ALS

        Returns
        -------
        U : numpy array
            The left factor matrix, dimension (n_movies, k)
        V : numpy array
            The right factor matrix, dimension (k, n_users)
        """
        U, V = self._iterSVD(A, mask_A, shrinkage, n_iter_svd) # Initialize U, V using iterative SVD
        U = np.copy(U[:,:k])
        V = np.copy(V[:k,:])

        for iter in range(n_iter_als):
            print(f"Iteration {iter+1}", end='\t\t')
            for i, Ri in enumerate(mask_A):
                temp1 = V@(Ri[:, None] * V.T) + lambd * np.eye(k)
                temp2 = np.dot(V, np.dot(np.diag(Ri), A[i].T))
                U[i] = np.linalg.solve(temp1, temp2).T
            U = np.clip(U, 0, None)
            print("Error after solving for U matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A), end='\t\t')


            for j, Rj in enumerate(mask_A.T):
                temp1 = U.T@(Rj[:,None] * U) + lambd * np.eye(k)
                temp2 = U.T@(Rj * A[:, j])
                V[:,j] = np.linalg.solve(temp1,temp2)
            V = np.clip(V, 0, None)
            print("Error after solving for V matrix:", np.sum((mask_A * (A - np.dot(U, V))) ** 2) / np.sum(mask_A))
        return U, V