import numpy as np

import time

import sklearn.metrics.pairwise

import diffi_interpretability_module

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

methods = ['DIFFI', 'ExIFFI', 'kNN', 'd']

###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

class MyInterpreter:

    def __init__(self, method, ensemble):

        self.method = method
        self.ensemble = ensemble

        self.vs = None
        self.is_left = None
        self.is_right = None
        self.split_attribute = None

        start_time = time.time()
        if method == 'ExIFFI':
            self.compute_exiffi_vs_is()
        elif method == 'kNN':
            self.compute_nn_vs_is()
        elif method == 'd':
            self.compute_d_vs_is()
        end_time = time.time()
        self.fit_time = end_time - start_time

        self.process_time = None


    def LFI(self, x):

        start_time = time.time()
        FI = 0
        for h, forest in enumerate(self.ensemble.iForest_lst):
            FI += self._LFI(forest, h, x)
        end_time = time.time()
        self.process_time = end_time - start_time
        return FI
    
    def GFI(self):

        start_time = time.time()
        FI = 0
        for h, forest in enumerate(self.ensemble.iForest_lst):
            FI += self._GFI(forest, h)
        end_time = time.time()
        self.process_time = end_time - start_time
        return FI

    def _LFI(self, forest, h, x):

        y = self.ensemble._deep_representation(self.ensemble.net_lst[h], x.reshape(1, -1)).reshape(1, -1)

        if self.method == 'DIFFI':
            return diffi_interpretability_module.local_diffi(forest, y)

        # initialization 
        I = np.zeros(self.ensemble.X.shape[1])
        V = np.zeros(self.ensemble.X.shape[1])

        # for every iTree in the iForest
        for k, estimator in enumerate(forest.estimators_):
            # update cumulative i and v
            node_indicator = estimator.decision_path(y).toarray()
            path = list(np.where(node_indicator == 1)[1])
            children_left = estimator.tree_.children_left
            for i in range(len(path)-1):
                    v = self.vs[h, k, path[i]]
                    i = self.is_left[h, k, path[i]] if path[i + 1] == children_left[path[i]] else self.is_right[h, k, path[i]] 
                    V = V + v
                    I = I + i

        # compute FI
        FI = np.zeros(self.ensemble.X.shape[1])
        for j in range(I.shape[0]):
            if V[j] != 0:
                FI[j] = I[j] / V[j]
        FI = np.nan_to_num(FI)

        return FI.reshape(1, -1)
    
    def _GFI(self, forest, h):

        if self.method == 'DIFFI':
            return diffi_interpretability_module.diffi_ib(forest, self.ensemble.x_reduced_lst[h])

        # initialization
        I_O = np.zeros(self.ensemble.X.shape[1]).astype('float')
        I_I = np.zeros(self.ensemble.X.shape[1]).astype('float')
        V_O = np.zeros(self.ensemble.X.shape[1]).astype('float')
        V_I = np.zeros(self.ensemble.X.shape[1]).astype('float')

        estimators = forest.estimators_

        Y_outliers_ib = self.ensemble.x_reduced_lst[h][self.ensemble.labels_ == 1]
        Y_inliers_ib = self.ensemble.x_reduced_lst[h][self.ensemble.labels_ == 0]

        if Y_outliers_ib.shape[0] == 0 or Y_inliers_ib.shape[0] == 0:
            return np.zeros(forest.X.shape[1]).astype('float')

        for k, estimator in enumerate(estimators):
            # update I_O and V_O
            node_indicator_all_points_outliers = estimator.decision_path(Y_outliers_ib).toarray()
            # for every point judged as abnormal
            children_left = estimator.tree_.children_left
            for i in range(len(Y_outliers_ib)):
                path = list(np.where(node_indicator_all_points_outliers[i] == 1)[0])
                for i in range(len(path)-1):
                    v = self.vs[h, k, path[i]]
                    i = self.is_left[h, k, path[i]] if path[i + 1] == children_left[path[i]] else self.is_right[h, k, path[i]] 
                    V_O = V_O + v
                    I_O = I_O + i
            # update I_I and V_I
            node_indicator_all_points_inliers = estimator.decision_path(Y_inliers_ib).toarray()
            # for every point judged as normal
            for i in range(len(Y_inliers_ib)):
                path = list(np.where(node_indicator_all_points_inliers[i] == 1)[0])
                for i in range(len(path)-1):
                    v = self.vs[h, k, path[i]]
                    i = self.is_left[h, k, path[i]] if path[i + 1] == children_left[path[i]] else self.is_right[h, k, path[i]] 
                    V_I = V_I + v
                    I_I = I_I + i

        FI_outliers = np.where(V_O > 0, I_O / V_O, 0)
        FI_inliers = np.where(V_I > 0, I_I / V_I, 0)
        FI = (FI_outliers / FI_inliers).reshape(1, -1)
        FI = np.nan_to_num(FI)

        return FI
    
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    def compute_exiffi_vs_is(self):

        estimators_num = self.ensemble.n_estimators
        ensemble_num = self.ensemble.n_ensemble
        max_nodes = 2*self.ensemble.max_samples - 1
        vect_length = self.ensemble.X.shape[1]
        self.vs = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.is_left = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.is_right = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.split_attribute = []

        for h, forest in enumerate(self.ensemble.iForest_lst):

            for k, estimator in enumerate(forest.estimators_):

                for node in range(estimator.tree_.node_count):

                    if estimator.tree_.children_left[node] > -1:

                        v = np.abs(self.ensemble.net_lst[h][:, estimator.tree_.feature[node]].reshape(-1))
                        i_left = v*estimator.tree_.n_node_samples[node]/estimator.tree_.n_node_samples[estimator.tree_.children_left[node]]
                        i_right = v*estimator.tree_.n_node_samples[node]/estimator.tree_.n_node_samples[estimator.tree_.children_right[node]]
                        self.vs[h, k, node] = v
                        self.is_left[h, k, node] = i_left
                        self.is_right[h, k, node] = i_right
                        self.split_attribute.append(estimator.tree_.feature[node])

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    def compute_nn_vs_is(self):

        estimators_num = self.ensemble.n_estimators
        ensemble_num = self.ensemble.n_ensemble
        max_nodes = 2*self.ensemble.max_samples - 1
        vect_length = self.ensemble.X.shape[1]
        self.vs = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.is_left = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.is_right = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.split_attribute = []

        for h, forest in enumerate(self.ensemble.iForest_lst):

            in_bag_samples = forest.estimators_samples_

            for k, estimator in enumerate(forest.estimators_):

                node_count = estimator.tree_.node_count
                children_left = estimator.tree_.children_left
                children_right = estimator.tree_.children_right
                n_node_samples = estimator.tree_.n_node_samples
                node_indicator = estimator.decision_path(self.ensemble.x_reduced_lst[h][in_bag_samples[k]]).toarray()

                in_bag_sample = list(in_bag_samples[k])
                X_ib = self.ensemble.X[in_bag_sample, :]

                for node in range(node_count):

                    if children_left[node] > -1:

                        left_data = X_ib[np.where(node_indicator[:, children_left[node]])[0]]
                        right_data = X_ib[np.where(node_indicator[:, children_right[node]])[0]] 

                        if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                            continue

                        # Compute pairwise Euclidean distances
                        distances = sklearn.metrics.pairwise.euclidean_distances(left_data, right_data)

                        # Find nearest neighbors
                        left_nn_index = np.argmin(distances, axis=1)   # nearest right neighbor for each left point
                        right_nn_index = np.argmin(distances, axis=0)  # nearest left neighbor for each right point

                        # Compute difference vectors to nearest neighbors
                        left_d = left_data - right_data[left_nn_index]  
                        right_d = right_data - left_data[right_nn_index] 

                        # Stack all difference vectors
                        d = np.vstack((left_d, right_d))

                        # Normalize each difference vector (to unit length) and take absolute values
                        norms = np.linalg.norm(d, axis=1, keepdims=True) 
                        d_normalized = np.abs(d) / norms

                        # Aggregate normalized vectors into a single vector
                        v_sum = np.sum(d_normalized, axis=0)

                        # Normalize the final vector to unit length
                        v = v_sum / np.linalg.norm(v_sum)

                        i_left = v * n_node_samples[node] / n_node_samples[children_left[node]]
                        i_right = v * n_node_samples[node] / n_node_samples[children_right[node]]
                        
                        self.vs[h, k, node] = v
                        self.is_left[h, k, node] = i_left
                        self.is_right[h, k, node] = i_right
                        self.split_attribute.append(estimator.tree_.feature[node])

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################

    def compute_d_vs_is(self):

        estimators_num = self.ensemble.n_estimators
        ensemble_num = self.ensemble.n_ensemble
        max_nodes = 2*self.ensemble.max_samples - 1
        vect_length = self.ensemble.X.shape[1]
        self.vs = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.is_left = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.is_right = np.zeros(shape=(ensemble_num, estimators_num, max_nodes, vect_length)).astype('float')
        self.split_attribute = []

        for h, forest in enumerate(self.ensemble.iForest_lst):

            in_bag_samples = forest.estimators_samples_

            for k, estimator in enumerate(forest.estimators_):

                node_count = estimator.tree_.node_count
                children_left = estimator.tree_.children_left
                children_right = estimator.tree_.children_right
                n_node_samples = estimator.tree_.n_node_samples
                node_indicator = estimator.decision_path(self.ensemble.x_reduced_lst[h][in_bag_samples[k]]).toarray()

                in_bag_sample = list(in_bag_samples[k])
                X_ib = self.ensemble.X[in_bag_sample, :]

                for node in range(node_count):

                    if children_left[node] > -1:

                        # Get left/right subsets
                        left_data = X_ib[np.where(node_indicator[:, children_left[node]])[0]]
                        right_data = X_ib[np.where(node_indicator[:, children_right[node]])[0]]

                        if left_data.shape[0] == 0 or right_data.shape[0] == 0:
                            continue

                        # Compute all pairwise differences between left and right samples
                        diffs = left_data[:, np.newaxis, :] - right_data[np.newaxis, :, :]

                        # Compute squared differences and normalize by squared norms
                        norms = np.linalg.norm(diffs, axis=2, keepdims=True)**2
                        d = np.abs(diffs) / norms

                        # Sum over all pairs and normalize the resulting vector
                        v = np.sum(d, axis=(0, 1))
                        v /= np.linalg.norm(v)
                        
                        i_left = v * n_node_samples[node] / n_node_samples[children_left[node]]
                        i_right = v * n_node_samples[node] / n_node_samples[children_right[node]]

                        # Store results
                        self.vs[h, k, node] = v
                        self.is_left[h, k, node] = i_left
                        self.is_right[h, k, node] = i_right
                        self.split_attribute.append(estimator.tree_.feature[node])

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################