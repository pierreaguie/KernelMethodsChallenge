import numpy as np
from scipy.linalg import eigh 
from src.kernel import Kernel

class PCA():

    def __init__(self, kernel : Kernel, n_components : int):
        self.kernel = kernel
        self.n_components = n_components

        self.alpha = None
        self.X_fit = None

    def fit(self, X_fit : np.ndarray):
        self.X_fit = X_fit
        K = self.kernel(X_fit, X_fit)
        n = len(K)
        ones = np.ones((n,n))/n
        K_c = K - ones @ K - K @ ones + ones @ K @ ones

        Delta, Beta = eigh(K_c, subset_by_index=[n-self.n_components, n-1])
        self.alpha = Beta[:, ::-1] / np.sqrt(Delta[::-1])

    def transform(self, X : np.ndarray) -> np.ndarray:
        K = self.kernel(X, self.X_fit)
        K_fit = self.kernel(self.X_fit, self.X_fit)
        n = len(K_fit)
        ones = np.ones((n, n)) / n

        K_c = K - ones @ K_fit - K @ ones + ones @ K_fit @ ones

        return K_c @ self.alpha