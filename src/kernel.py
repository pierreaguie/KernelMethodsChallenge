import numpy as np
from itertools import product, combinations
from typing import Tuple, List
from tqdm import tqdm



class Kernel():
    """ 
    A generic class for kernels.
    """

    def __init__(self):
        pass

    def __call__(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        """ 
        Returns the kernel matrix K = [ K(X_i, Y_j) ]_{i, j}

        Parameters:
            X (np.ndarray) : Array of shape (n,)
            Y (np.ndarray) : Array of shape (m,)

        Returns:
            K (np.ndarray) : The kernel matrix of shape (n, m)
        """
        pass

    def norm(self, X : np.ndarray) -> np.ndarray:
        """ 
        Returns the norm of the feature vectors of the elements in X

        Parameters:
            X (np.ndarray) : Array of shape (n,)

        Returns:
            norm (np.ndarray) : The norm of each element's feature vector
        """
        pass


class NormalizedKernel(Kernel):

    def __init__(self, kernel : Kernel):
        self.kernel = kernel

    def __call__(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        K = self.kernel(X, Y)
        if X is Y:
            K /= np.outer(np.sqrt(np.diag(K)), np.sqrt(np.diag(K)))
        else:
            K /= np.outer(self.kernel.norm(X), self.kernel.norm(Y))
        return K
    
    def norm(self, X : np.ndarray) -> np.ndarray:
        return np.ones(len(X))


class SumKernel(Kernel):

    def __init__(self, kernels : List[Kernel], weights : List[float]):
        self.kernels = kernels
        self.weights = weights

    def __call__(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        K = np.zeros((len(X), len(Y)))
        for kernel, weight in zip(self.kernels, self.weights):
            K += weight * kernel(X, Y)
        return K
    
    def norm(self, X : np.ndarray) -> np.ndarray:
        norms = [kernel.norm(X) for kernel in self.kernels]
        return sum([weight * norm for weight, norm in zip(self.weights, norms)])
    

class LinearKernel(Kernel):

    def __init__(self):
        pass

    def __call__(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        return X @ Y.T
    
    def norm(self, X : np.ndarray) -> np.ndarray:
        return np.linalg.norm(X, axis = 1)



class SpectrumKernel(Kernel):

    def __init__(self, k : int):
        self.k = k
        self.kmers = [''.join(x) for x in product('ACGT', repeat=k)]
        self.kmer2idx = {kmer: i for i, kmer in enumerate(self.kmers)}
        
        # If k is small, we evaluate kernels thorugh numpy dot product. If not, we use pre-indexation to only sum up non-zero features.
        self.largek = self.k > 8
        # For small ks, self.seq2phi is {sequence : numpy feature vector}
        # For large ks, self.seq2phi is {sequence : {kmer : count} dict} 
        self.seq2phi = {}        

    def __call__(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        self.save_phi(X)
        self.save_phi(Y)
        if not self.largek:
            X_w = np.array([self.seq2phi[seq] for seq in X])
            Y_w = np.array([self.seq2phi[seq] for seq in Y])
            K = X_w @ Y_w.T
            return K
        else:
            K = np.zeros((len(X), len(Y)))
            for i in range(len(X)):
                for j in range(len(Y)):
                    K[i, j] = sum([self.seq2phi[X[i]][kmer] * self.seq2phi[Y[j]].get(kmer, 0) for kmer in self.seq2phi[X[i]]])
            return K
    
    def norm(self, X : np.ndarray) -> np.ndarray:
        self.save_phi(X)
        return np.linalg.norm([self.seq2phi[seq] for seq in X], axis = 1)

    def save_phi(self, X : np.ndarray):
        """ 
        Saves in self.seq2phi the weight vector for each sequence in X

        Parameters:
            X (np.ndarray) : The array of sequences
        """
        for seq in X:
            if seq not in self.seq2phi:
                if not self.largek:
                    w = np.zeros(len(self.kmers))
                    for i in range(len(seq) - self.k + 1):
                        kmer = seq[i:i+self.k]
                        w[self.kmer2idx[kmer]] += 1
                    self.seq2phi[seq] = w
                else:
                    w = {}
                    for i in range(len(seq) - self.k + 1):
                        kmer = seq[i:i+self.k]
                        if kmer in w:
                            w[kmer] += 1
                        else:
                            w[kmer] = 1
                    self.seq2phi[seq] = w




class MismatchKernel(Kernel):

    def __init__(self, k : int, m : int):
        self.k = k
        self.m = m
        self.kmers = [''.join(x) for x in product('ACGT', repeat=k)]
        self.kmer2idx = {kmer : i for i, kmer in enumerate(self.kmers)}
        self.seq2phi = {}        # Stores the feature vector for each sequence
        self.mismatches = {kmer : self.mismatch_kmers(kmer) for kmer in self.kmers}


    def __call__(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        self.save_phi(np.concatenate((X, Y)))
        X_w = np.array([self.seq2phi[seq] for seq in X])
        Y_w = np.array([self.seq2phi[seq] for seq in Y])
        K = X_w @ Y_w.T
        return K
    
    def norm(self, X : np.ndarray) -> np.ndarray:
        self.save_phi(X)
        return np.linalg.norm([self.seq2phi[seq] for seq in X], axis = 1)
        

    def save_phi(self, X : np.ndarray):
        """ 
        Saves in self.seq2phi the weight vector for each sequence in X

        Parameters:
            X (np.ndarray) : The array of sequences
        """
        for seq in X:
            if seq not in self.seq2phi:
                w = np.zeros(len(self.kmers))
                for i in range(len(seq) - self.k + 1):
                    kmer = seq[i:i+self.k]
                    for mismatch in self.mismatches[kmer]:
                        w[self.kmer2idx[mismatch]] += 1

                self.seq2phi[seq] = w


    def mismatch_kmers(self, kmer : str) -> list:
        """ 
        Returns the set of kmers at distance at most m from kmer

        Parameters:
            kmer (str) : The kmer

        Returns:
            kmers (list) : The set of kmers at distance at most m from kmer
        """
        kmers = [kmer]
        
        for num_mismatches in range(1, self.m + 1):
            for mismatch_pos in combinations(range(self.k), num_mismatches):
                for subs in product('ACGT', repeat=num_mismatches):
                    new_kmer = list(kmer)
                    for pos, sub in zip(mismatch_pos, subs):
                        new_kmer[pos] = sub
                    kmers.append(''.join(new_kmer))

        return kmers



class SubstringKernel(Kernel):

    def __init__(self, k : int, lambd : float):
        self.k = k
        self.lambd = lambd
        self.buffer = {}

    def __call__(self, X, Y):
        K = np.zeros((len(X), len(Y)))

        if X is Y:
            for i in tqdm(range(len(X))):
                for j in range(i, len(X)):
                    K[i, j] = self.K(X[i], X[j])
                    K[j, i] = K[i, j]

        else:
            for i in range(len(X)):
                for j in range(len(Y)):
                    K[i, j] = self.K(X[i], Y[j])

        return K
    
    def B(self, x : str, y : str):
        
        dp = np.zeros((len(x) + 1, len(y) + 1, self.k + 1))
        dp[:, :, 0] = 1
        for q in range(1, self.k + 1):
            for i in range(1, len(x) + 1):
                for j in range(1, len(y) + 1):
                    dp[i, j, q] = self.lambd * dp[i-1, j, q] + self.lambd * dp[i, j-1, q] - self.lambd**2 * dp[i-1, j-1, q] + dp[i-1, j-1, q-1] * (x[i-1] == y[j-1]) * self.lambd**2
        return dp


    def K(self, x : str, y : str):
        if (x,y) in self.buffer:
            return self.buffer[(x, y)]
        elif (y,x) in self.buffer:
            return self.buffer[(y,x)]
        dp = self.B(x, y)
        res = 0
        for i in range(self.k - 2, len(x)):
            j_idx = [j for j in range(len(y)) if y[j] == x[i]]
            res += self.lambd**2 * dp[i, j_idx, self.k - 1].sum()
        self.buffer[(x, y)] = res
        return res
    
