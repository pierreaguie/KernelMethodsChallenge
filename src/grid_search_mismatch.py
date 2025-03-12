import numpy as np
from typing import List
import pandas as pd
from itertools import product
from tqdm import tqdm

from classifier import SVM
from kernel import SpectrumKernel, MismatchKernel, NormalizedKernel
from utils import cross_val_score, load_data


def grid_search(X : np.ndarray, y : np.ndarray,
                k_grid : List[int], m_grid : List[int], C_grid = List[float],
                cv : int = 5, log_path : str = "outputs/grid_search.csv") -> pd.DataFrame:
    
    results = []
    
    with open(log_path, "a") as f:
        f.write("k,m,C,score_mean,score_std\n")
    
    for k, m, C in tqdm(product(k_grid, m_grid, C_grid), desc = "Grid search", total = len(k_grid) * len(m_grid) * len(C_grid)):
        with open(log_path, "a") as f:
            if m == 0:
                kernel = NormalizedKernel(SpectrumKernel(k))
            else:
                kernel = NormalizedKernel(MismatchKernel(k, m))
            model = SVM(kernel, intercept = True, C = C)
            scores = cross_val_score(model, X, y, cv)
            results.append({"k" : k, "m" : m, "C" : C, "score_mean" : np.mean(scores), "score_std" : np.std(scores)})
            f.write(f"{k},{m},{C},{np.mean(scores):.4f},{np.std(scores):.4f}\n")

    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
    np.random.seed(42)

    k_grid = [9]
    m_grid = [0]
    C_grid = [0.1, 1., 10.]

    for idx in [0, 1, 2]:
        X, y = load_data("train", idx)
        grid_search(X, y, k_grid, m_grid, C_grid, cv = 5, log_path = f"outputs/mismatch_grid_search/grid_search_{idx}.csv")

