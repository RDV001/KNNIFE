from pyod.models.dif import DIF
from pyod.utils.utility import invert_order

from sklearn.utils import check_array

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score

import numpy as np

import pandas as pd

import time

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

fit_scores = ['roc_auc', 'average_precision', 'accuracy', 'precision', 'f1', 'detected_anomalies', 'fit_time']

def fit_stats(model, y_true):

    y_score = model.decision_scores_
    y_pred = model.labels_

    df = pd.DataFrame(columns=fit_scores)

    roc_auc = roc_auc_score(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    detected_anomalies = np.sum(y_pred).astype('int64')
    fit_time = model.fit_time

    stats = list([roc_auc, average_precision, accuracy, precision, f1, detected_anomalies, fit_time])
    df.loc[len(df)] = stats

    return df

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

class MyIF(DIF):

    def __init__(
            self, 
            n_estimators=300, 
            max_samples=256, 
            contamination=None, 
            random_state=None,
        ):

        super().__init__(
            batch_size=None,
            representation_dim=None,
            hidden_neurons=None,
            hidden_activation=None,
            skip_connection=None,
            n_ensemble=1,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            device='cpu'
        )

        self.fit_time = None

    def fit(self, X):

        start_time = time.time()

        self.net_lst = []
        self.iForest_lst = []
        self.x_reduced_lst = []
        self.X = X

        self.net_lst.append(np.eye(X.shape[1]))
        self.x_reduced_lst.append(X)

        self.iForest_lst.append(
            IsolationForest(n_estimators=self.n_estimators,
                            max_samples=self.max_samples
            )
        )
        self.iForest_lst[0].fit(X)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        end_time = time.time()
        self.fit_time = end_time - start_time

        return self
    
    def decision_function(self, X):
        return invert_order(self.iForest_lst[0].decision_function(self._deep_representation(self.net_lst[0], X)))
    
    def _deep_representation(self, net, X):
        return X
    
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

class MyEIF(DIF):

    def __init__(
            self,
            n_estimators=300, 
            max_samples=256, 
            contamination=None, 
            random_state=None,
        ):

        super().__init__(
            batch_size=None,
            representation_dim=None,
            hidden_neurons=None,
            hidden_activation=None,
            skip_connection=None,
            n_ensemble=1,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            device='cpu'
        )

        # self.m = (max_samples - 1)*n_estimators
        self.m = max_samples - 1
        self.fit_time = None

    def fit(self, X):

        start_time = time.time()

        self.net_lst = []
        self.iForest_lst = []
        self.x_reduced_lst = []
        self.X = X

        A = np.random.randn(X.shape[1], self.m)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)

        self.net_lst.append(A)
        self.x_reduced_lst.append(self._deep_representation(A, X))

        self.iForest_lst.append(
            IsolationForest(n_estimators=self.n_estimators,
                            max_samples=self.max_samples,
            )
        )
        self.iForest_lst[0].fit(self.x_reduced_lst[0])
        self.reduce_representation()

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        end_time = time.time()
        self.fit_time = end_time - start_time

        return self
    
    def reduce_representation(self):

        nodes = np.sum([self.iForest_lst[0].estimators_[j].tree_.node_count for j in range(len(self.iForest_lst[0].estimators_))])
        features = -np.ones(nodes)

        k = 0
        for estimator in self.iForest_lst[0].estimators_:
            for node in range(estimator.tree_.node_count):
                features[k] = estimator.tree_.feature[node]
                k += 1

        # Keep only valid feature indices
        features = features[features > -1]

        # Get unique features and index mapping
        unique_features, indices = np.unique(features, return_inverse=True)
        unique_features = unique_features.astype(np.int64)
        self.m = len(unique_features)

        # Update trees
        k = 0
        self.iForest_lst[0].n_features_in_ = self.m
        self.iForest_lst[0]._max_features = self.m
        for estimator in self.iForest_lst[0].estimators_:
            estimator.n_features_in_ = self.m
            for node in range(estimator.tree_.node_count):
                if estimator.tree_.feature[node] > -1:
                    estimator.tree_.feature[node] = indices[k]
                    k += 1

        # Reduce
        self.net_lst[0] = self.net_lst[0][:, unique_features]
        self.x_reduced_lst[0] = self.x_reduced_lst[0][:, unique_features]

    def decision_function(self, X):
        return invert_order(self.iForest_lst[0].decision_function(self._deep_representation(self.net_lst[0], X)))
    
    def _deep_representation(self, net, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return (X @ net).reshape(X.shape[0], -1)
    
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

class MyNNIF(DIF):

    def __init__(
            self, 
            n_estimators=300, 
            max_samples=256, 
            contamination=None, 
            random_state=None,
        ):

        super().__init__(
            batch_size=None,
            representation_dim=None,
            hidden_neurons=None,
            hidden_activation=None,
            skip_connection=None,
            n_ensemble=1,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            device='cpu'
        )

        self.m = max_samples - 1
        self.fit_time = None

    def fit(self, X):

        start_time = time.time()

        self.net_lst = []
        self.iForest_lst = []
        self.x_reduced_lst = []
        self.X = X

        A = np.random.randn(X.shape[1], 500)
        B = np.random.randn(500, 100)
        C = np.random.randn(100, self.m)
        bA = np.random.randn(1, 500)
        bB = np.random.randn(1, 100)
        bC = np.random.randn(1, self.m)

        self.net_lst.append((A, B, C, bA, bB, bC))
        self.x_reduced_lst.append(self._deep_representation(self.net_lst[0], X))

        self.iForest_lst.append(
            IsolationForest(n_estimators=self.n_estimators,
                            max_samples=self.max_samples,
            )
        )
        self.iForest_lst[0].fit(self.x_reduced_lst[0])
        self.reduce_representation()

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        end_time = time.time()
        self.fit_time = end_time - start_time

        return self
    
    def reduce_representation(self):

        nodes = np.sum([self.iForest_lst[0].estimators_[j].tree_.node_count for j in range(len(self.iForest_lst[0].estimators_))])
        features = -np.ones(nodes)

        k = 0
        for estimator in self.iForest_lst[0].estimators_:
            for node in range(estimator.tree_.node_count):
                features[k] = estimator.tree_.feature[node]
                k += 1

        # Keep only valid feature indices
        features = features[features > -1]

        # Get unique features and index mapping
        unique_features, indices = np.unique(features, return_inverse=True)
        unique_features = unique_features.astype(np.int64)
        self.m = len(unique_features)

        # Update trees
        k = 0
        self.iForest_lst[0].n_features_in_ = self.m
        self.iForest_lst[0]._max_features = self.m
        for estimator in self.iForest_lst[0].estimators_:
            estimator.n_features_in_ = self.m
            for node in range(estimator.tree_.node_count):
                if estimator.tree_.feature[node] > -1:
                    estimator.tree_.feature[node] = indices[k]
                    k += 1

        # Reduce
        self.net_lst[0] = (self.net_lst[0][0], 
                           self.net_lst[0][1], 
                           self.net_lst[0][2][:, unique_features],
                           self.net_lst[0][3], 
                           self.net_lst[0][4], 
                           self.net_lst[0][5][:, unique_features]
        )
        self.x_reduced_lst[0] = self.x_reduced_lst[0][:, unique_features]
            
        
    def decision_function(self, X):
        return invert_order(self.iForest_lst[0].decision_function(self._deep_representation(self.net_lst[0], X)))

    def _deep_representation(self, net, X):

        A = net[0]
        B = net[1]
        C = net[2]
        bA = net[3]
        bB = net[4]
        bC = net[5]

        a1 = X @ A + bA
        o1 = np.where(a1 > 0, a1, 0.2*a1) 
        a2 = o1 @ B + bB
        o2 = np.where(a2 > 0, a2, 0.2*a2) 
        a3 = o2 @ C + bC
        o3 = a3.reshape(-1, self.m)
        return o3
    
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

class MyDIF(DIF):

    def __init__(
            self, 
            n_ensemble=50, 
            n_estimators=6, 
            max_samples=256, 
            contamination=None, 
            random_state=None
        ):

        super().__init__(
            n_ensemble=n_ensemble,
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            device='cpu'
        )

        self.fit_time = None

    def fit(self, X):

        self.X = X
        start_time = time.time()
        super().fit(X)
        end_time = time.time()
        self.fit_time = end_time - start_time
        return self