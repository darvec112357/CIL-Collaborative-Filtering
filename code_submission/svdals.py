import numpy as np


def normalize(A):
    """
    Normalize A by subtracting the mean and dividing by the standard deviation
    """
    col_mean = np.nanmean(A, axis=0)
    col_std = np.nanstd(A, axis=0)
    A_n = (A-col_mean)/col_std
    return A_n, col_mean, col_std


def de_normalize(A, mean, std):
    """
    Reverse the normalization of A
    """
    return (A*std) + mean

class ALS:
    """
    Combines the Iterative SVD initialization and ALS together into one function
    """
    def __init__(self):
        pass

    def _iterSVD(self, A, mask_A, shrinkage, n_iter):
        """
        Parameters
        ----------
        A : numpy array
            The matrix to be factorized, dimension (n_users, n_items)
        mask_A : numpy array
            The mask of A, dimension (n_users, n_items)
        shrinkage : int
            The shrinkage parameter for IterSVD
        n_iter : int
            The number of iterations for IterSVD

        Returns
        -------
        U : numpy array
            The left factor matrix, dimension (n_users, k)
        V : numpy array
            The right factor matrix, dimension (k, n_items)
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
            The matrix to be factorized, dimension (n_users, n_items)
        mask_A : numpy array
            The mask of A, dimension (n_users, n_items)
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
            The left factor matrix, dimension (n_users, k)
        V : numpy array
            The right factor matrix, dimension (k, n_items)
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
        return de_normalize(U.dot(V), col_mean, col_std)