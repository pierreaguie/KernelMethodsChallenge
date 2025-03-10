import numpy as np
from tqdm import tqdm
from typing import Tuple, List
import cvxpy as cp


try :
    import gurobipy as gp
    print("Using Gurobi")
    GUROBI_IMPORTED = True
except ImportError:
    print("Using CVXPY")
    GUROBI_IMPORTED = False


from kernel import Kernel, SumKernel


class Classifier():
    """ 
    A generic class for classifiers.
    """

    def __init__(self):
        pass

    def fit(self, X : np.ndarray, y : np.ndarray):
        """
        Fits the classifier to the data (X, y)

        Parameters:
            X (np.ndarray) : The array of sequences
            y (np.ndarray) : The label vector        
        """
        pass

    def predict(self, X_test : np.ndarray) -> np.ndarray:
        """
        Predicts the labels of the sequences in X_test

        Parameters:
            X_test (np.ndarray) : The list of sequences to predict
                
        Returns:
            y_pred (np.ndarray) : The predicted labels
        """
        pass



class SVM(Classifier):

    def __init__(self, kernel : Kernel, C : float = 1., epsilon : float = 1e-10, intercept : bool = True):
        self.kernel = kernel
        self.C = C
        self.intercept = intercept

        self.epsilon = epsilon
        self.support = None

        self.alpha = None
        self.b = None


    def fit(self, X : np.ndarray, y : np.ndarray):
        K = self.kernel(X, X) 
        Q = K * np.outer(y, y)
        
        self.beta = self.solve_qp(Q, y)
        self.support = X[self.beta > self.epsilon]
        self.alpha = self.beta[self.beta > self.epsilon] * y[self.beta > self.epsilon]

        if self.intercept:
            on_the_margin = (self.epsilon < self.beta) & (self.beta < self.C - self.epsilon)
            self.b = np.mean(y[on_the_margin] - (self.beta * y) @ K[:, on_the_margin])
        else:
            self.b = 0.


    def predict(self, X_test : np.ndarray) -> np.ndarray:
        K = self.kernel(X_test, self.support)
        f = K @ self.alpha
        return 2 * (f + self.b > 0) - 1
    

    def solve_qp(self, Q : np.ndarray, y : np.ndarray) -> np.ndarray:
        """ 
        Solves the following QP:
            min 0.5 * beta @ Q @ beta - 1 @ beta
            s.t. 0 <= beta <= C
                 y @ beta == 0
        
        Parameters:
            Q (np.ndarray) : The matrix of elements Q_ij = y_i * y_j * K(X_i, X_j)
            y (np.ndarray) : The label vector

        Returns:
            beta (np.ndarray) : The solution of the QP
        """

        if GUROBI_IMPORTED:
            m = gp.Model("qp")
            m.setParam('OutputFlag', 0)
            beta = m.addMVar(shape = len(y), name='beta')
            if self.intercept:
                m.addConstr(y @ beta == 0)
            m.addConstr(beta >= 0)
            m.addConstr(beta <= self.C)
            m.setObjective(0.5 * beta @ Q @ beta - np.ones(len(y)) @ beta, gp.GRB.MINIMIZE)
            m.optimize()

            return beta.X
    
        else:
            beta = cp.Variable(len(y))
            constraints = [0 <= beta, beta <= self.C]
            if self.intercept:
                constraints.append(y @ beta == 0)
            objective = cp.Minimize(0.5 * cp.quad_form(beta, Q) - cp.sum(beta))
            problem = cp.Problem(objective, constraints)
            problem.solve()

            return beta.value


class MultiKernelSVM(Classifier):
    """ 
    A class for Multiple Kernel Learning (MKL) with SVMs.
    """

    def __init__(self, kernels : List[Kernel], C : float = 1., c : float = 1., epsilon : float = 1e-10):
        self.kernels = kernels
        self.C = C
        self.c = c

        self.epsilon = epsilon
        self.support = None
        self.sum_kernel = None

        self.alpha = None
        self.b = None


    def fit(self, X : np.ndarray, y : np.ndarray):
        K = [kernel(X, X) for kernel in self.kernels]
        Q = [K[i] * np.outer(y, y) for i in range(len(self.kernels))]

        beta, mu = self.solve_socp(Q, y)

        self.beta = beta
        self.support = X[beta > self.epsilon]
        self.alpha = beta[beta > self.epsilon] * y[beta > self.epsilon]

        self.mu = mu
        self.sum_kernel = SumKernel(self.kernels, mu)

        on_the_margin = (self.epsilon < beta) & (beta < self.C - self.epsilon)
        self.b = np.mean(y[on_the_margin] - (beta * y) @ self.sum_kernel(X, X[on_the_margin]))


    def predict(self, X_test : np.ndarray) -> np.ndarray:
        K = self.sum_kernel(X_test, self.support)
        f = K @ self.alpha
        return 2 * (f + self.b > 0) - 1


    
    def solve_socp(self, Q : List[np.ndarray], y : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the following SOCP:
            min c/2 * t - 1 @ beta
            s.t. 0 <= beta <= C
                 y @ beta == 0
                 beta @ Q_i @ beta / r_i <= t for i = 1, ..., len(Q)
        
        Parameters:
            Q (List[np.ndarray]) : The list of matrices Q_k of elements Q_k_ij = y_i * y_j * K_k(X_i, X_j)
            y (np.ndarray) : The label vector

        Returns:
            beta (np.ndarray) : The solution of the SOCP
            mu (np.ndarray) : The dual variables associated with the SOCP constraints
        """


        r = [np.trace(Q[i]) for i in range(len(Q))]

        if GUROBI_IMPORTED:
            m = gp.Model()
            m.setParam('OutputFlag', 0)
            m.setParam('QCPDual', 1)

            beta = m.addMVar(shape = len(y), name='beta')
            t = m.addVar(name='t')

            m.addConstr(y @ beta == 0)
            m.addConstr(beta >= 0)
            m.addConstr(beta <= self.C)

            socp_constraints : List[gp.MQConstr] = []
            for i in range(len(Q)):
                socp_constraints.append(m.addConstr(beta @ Q[i] @ beta / r[i] <= t, name = f'c{i}'))

            m.setObjective(self.c / 2 * t - np.ones(len(y)) @ beta, gp.GRB.MINIMIZE)
            m.optimize()

            mu = np.array([c.QCPi for c in socp_constraints])
            
            return beta.X, mu
        
        else:
            beta = cp.Variable(len(y))
            t = cp.Variable()

            constraints = [0 <= beta, beta <= self.C]
            constraints += [beta @ Q[i] @ beta / r[i] <= t for i in range(len(Q))]
            constraints.append(y @ beta == 0)

            objective = cp.Minimize(self.c / 2 * t - cp.sum(beta))
            problem = cp.Problem(objective, constraints)
            problem.solve()

            mu = np.array([c.dual_value for c in constraints[2:-1]])

            return beta.value, mu




class LogisticRegression(Classifier):

    def __init__(self, kernel : Kernel, mu : float = 1.):
        self.kernel = kernel
        self.mu = mu

        self.X_train = None
        self.alpha = None
        self.b = None


    def fit(self, X : np.ndarray, y : np.ndarray):
        n = len(X)
        K = self.kernel(X, X)

        alpha = cp.Variable(n)
        b = cp.Variable()
        loss = cp.sum(cp.logistic(-cp.multiply(y, K @ alpha + b))) + self.mu * cp.quad_form(alpha, K)
        objective = cp.Minimize(loss)
        problem = cp.Problem(objective)
        problem.solve()

        self.alpha = alpha.value
        self.b = b.value
        self.X_train = X


    def predict(self, X_test : np.ndarray) -> np.ndarray:
        K = self.kernel(X_test, self.X_train)
        f = K @ self.alpha
        return 2 * (f + self.b > 0) - 1