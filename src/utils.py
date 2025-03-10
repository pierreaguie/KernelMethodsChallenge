import numpy as np
import pandas as pd
from typing import Tuple

from classifier import Classifier


def accuracy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def f1_score(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def train_val_split(X : np.ndarray, y : np.ndarray, val_size : float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    idx = np.random.permutation(n)
    X = X[idx]
    y = y[idx]
    n_train = int(n * (1 - val_size))
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def cross_val_score(model : Classifier, X : np.ndarray, y : np.ndarray, cv : int = 5) -> np.ndarray:
    n = len(X)
    idx = np.random.permutation(n)
    X = X[idx]
    y = y[idx]
    scores = []
    for i in range(cv):
        idx_val = np.arange(i * n // cv, (i + 1) * n // cv)
        idx_train = np.setdiff1d(np.arange(n), idx_val, assume_unique=True)
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(accuracy(y_val, y_pred))
    return np.array(scores)


def load_data(split : str = "train", idx : int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if split == "train":
        split_str = "tr"
        X_path = f"data/X{split_str}{idx}.csv"
        y_path = f"data/y{split_str}{idx}.csv"
        X = pd.read_csv(X_path)['seq'].values
        y = 2 * pd.read_csv(y_path)['Bound'].values - 1
        return X, y
    
    elif split == "test":
        split_str = "te"
        X_path = f"data/X{split_str}{idx}.csv"
        X = pd.read_csv(X_path)['seq'].values
        return X, None
    
    else:
        raise ValueError("Split must be either 'train' or 'test'")