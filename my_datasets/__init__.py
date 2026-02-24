import os

import scipy.io
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

balif_small_datasets_names = [
    "wine",
    "vertebral",
    "ionosphere",
    "wbc",
    "breastw",
    "pima",
]

balif_medium_datasets_names = [
    "vowels",
    "letter",
    "cardio",
    "thyroid",
]

balif_large_datasets_names = [
    "optdigits",
    "satimage-2",
    "satellite",
    "pendigits",
    "annthyroid",
    "mammography",
]

tep_datasets_names = [
    f"TEP_{i+1}" for i in range(20)
]

csv_added_datasets_names = [
    "bisect", 
    "bisect3d",
    "bisect3d_skewed",
    "bisect6d",
    "xaxis",
    "pageblocks",
    "shuttle",
    "glass",
]

industrial_datasets_names = tep_datasets_names + ["secom"]
mat_datasets_names = balif_small_datasets_names + balif_medium_datasets_names + balif_large_datasets_names
csv_datasets_names = csv_added_datasets_names + industrial_datasets_names
datasets_names = mat_datasets_names + csv_datasets_names + industrial_datasets_names

def load(dataset_name=None, scale=None, af_num=None, rf_idx=None, eps=1.0, seed=None):

    # 1. Load data
    if dataset_name in mat_datasets_names:
        mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), f"{dataset_name}.mat"))
        data, labels = mat["X"], mat["y"][:, 0]
    elif dataset_name in csv_datasets_names:
        data_in = pd.read_csv(os.path.join(os.path.dirname(__file__), f"{dataset_name}.csv")).to_numpy()
        data = data_in[:, 1:-1]
        labels = data_in[:, -1]
    name = dataset_name

    # 2. Scale data
    if scale == 's':  # Standardized
        std = data.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        data = (data - data.mean(axis=0)) / std
        name = f'{name}_s'
    if scale == 'u':  # Unit-max
        data = (data - data.mean(axis=0)) / np.max(np.abs(data))
        name = f'{name}_u'
    X = np.array(data, dtype=np.float32)
    y = np.array(labels)

    # 3. Set seed
    if seed is not None:
        np.random.seed(seed)

    # 4. Add random features
    if af_num is not None and af_num > 0:
        added_features = np.random.uniform(size=(X.shape[0], af_num)) * eps
        X = np.hstack((X, added_features)).astype(np.float32)
        name = f'{name}_af{af_num}'

    # 5. Randomize features
    if rf_idx is not None:
        for i in rf_idx:
            X[:, i] = np.random.uniform(size=(X.shape[0])) * eps
            name = f'{name}_rf{i+1}'
    
    # 6. Return data as dict
    return {
        "X": X,
        "y": y,
        "contamination": np.mean(y),
        "name": name
    }

def dataset_dim_hist(dataset_name=None, scale=None, af_num=None, rf_idx=None, eps=1.0, seed=None, scale_hist=True, bins=10, log=False):

    # 1. Load data
    data_dict = load(dataset_name, scale, af_num, rf_idx, eps, seed)
    X = data_dict['X']
    y = data_dict['y']
    name = data_dict['name']

    if not scale_hist:
        X_min, X_max = np.min(X), np.max(X)
        bins_edges = np.linspace(X_min, X_max, bins + 1)

    for dim in range(X.shape[1]):

        if scale_hist:
            X_min, X_max = np.min(X[:, dim]), np.max(X[:, dim])
            bins_edges = np.linspace(X_min, X_max, bins + 1)

        plt.figure(figsize=(6, 6))
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.hist(X[y == 0, dim], bins=bins_edges, alpha=0.5, color="cornflowerblue", log=log, label="Inliers")
        plt.hist(X[y == 1, dim], bins=bins_edges, alpha=0.5, color="darkorange", log=log, label="Outliers")
        plt.legend()
        plt.xlabel(f"Feature {dim+1} value")
        plt.ylabel("Count (log scale)" if log else "Count")
        plt.show()
