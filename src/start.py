import numpy as np
import pandas as pd
import argparse

from classifier import SVM
from kernel import SpectrumKernel, MismatchKernel, NormalizedKernel
from utils import load_data

parser = argparse.ArgumentParser(description='Train a SVM model and make predictions')

parser.add_argument('--k_0', type=int, default=7, help='The length of the k-mers for dataset #0')
parser.add_argument('--m_0', type=int, default=2, help='The number of mismatches for dataset #0')
parser.add_argument('--C_0', type=float, default=1., help='The regularization parameter for dataset #0')

parser.add_argument('--k_1', type=int, default=8, help='The length of the k-mers for dataset #1')
parser.add_argument('--m_1', type=int, default=1, help='The number of mismatches for dataset #1')
parser.add_argument('--C_1', type=float, default=1., help='The regularization parameter for dataset #1')

parser.add_argument('--k_2', type=int, default=8, help='The length of the k-mers for dataset #2')
parser.add_argument('--m_2', type=int, default=0, help='The number of mismatches for dataset #2')
parser.add_argument('--C_2', type=float, default=10., help='The regularization parameter for dataset #2')

args = parser.parse_args()


if __name__ == "__main__":
    np.random.seed(42)
    
    Yte = np.zeros(3000)

    for idx in range(3):
        k, m, C = getattr(args, f"k_{idx}"), getattr(args, f"m_{idx}"), getattr(args, f"C_{idx}")
        print(f"Fitting model for dataset {idx} with k = {k}, m = {m}, C = {C}")

        X_train, y_train = load_data("train", idx)
        if m == 0:
            kernel = NormalizedKernel(SpectrumKernel(k = k))
        else:
            kernel = NormalizedKernel(MismatchKernel(k = k, m = m))
        model = SVM(kernel = kernel, intercept = True, C = C)
        model.fit(X_train, y_train)

        X_test, _ = load_data("test", idx)
        y_pred = model.predict(X_test)
        Yte[idx * 1000 : (idx + 1) * 1000] = y_pred

    Yte = pd.DataFrame({"Id" : np.arange(3000), "Bound" : (Yte + 1) // 2}, dtype = int)
    Yte.to_csv("outputs/Yte.csv", index = False)