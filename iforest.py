
# Follows algorithm from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix


class Node:
    def __init__(self, idx=None, split=None, left=None, right=None, size=None, path_depth=None):
        self.idx = idx
        self.split = split
        self.left = left
        self.right = right
        self.size = size
        self.path_depth = path_depth


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.max_depth = math.ceil(math.log(sample_size, 2))
        self.trees = []

    def fit(self, X:np.ndarray):
        """
        Given a matrix of observations, build an ensemble of IsolationTrees
        and store them in self.trees. Convert dataframe to numpy ndarray.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        k = 0
        while k < self.n_trees:
            samp = X[np.random.choice(X.shape[0], self.sample_size, replace=False), :]
            tree = IsolationTree(height_limit=self.max_depth)
            tree.fit(samp)
            self.trees.append(tree)
            k += 1

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a matrix of observations, X, compute the average path length
        for each observation in X. First compute the path length for x_i using
        every tree in self.trees and then compute the average for each x_i.
        Return a numpy ndarray of average path lengths.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.sample_size > 2:
            c = 2*np.log(self.sample_size-1)+0.5772156649\
                -2*(self.sample_size-1)/self.sample_size
        elif self.sample_size == 2:
            c = 1
        else:
            c = 0
        paths = []
        for row in X:
            x_paths = []
            for tree in self.trees:
                root = tree.root
                while root.right is not None:
                    if row[root.idx] < root.split:
                        root = root.left
                    else:
                        root = root.right
                x_paths.append(root.path_depth+c)
            paths.append(np.mean(x_paths))
        return np.array(paths)


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a matrix of observations, X, compute the anomaly score for each
        x_i observation, returning an ndarray of them.
        """
        if self.sample_size > 2:
            c = 2*np.log(self.sample_size-1)+0.5772156649\
                -2*(self.sample_size-1)/self.sample_size
        elif self.sample_size == 2:
            c = 1
        else:
            c = 0
        p_lengths = self.path_length(X)
        score = lambda p: 2**(-p/c)
        return score(p_lengths)


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return scores >= threshold


    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "Easy way to call anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None
        self.n_nodes = 0

    def fit(self, X:np.ndarray, height=0):
        """
        Given a matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        """
        self.n_nodes += 1
        if len(X) <= 1 or height >= self.height_limit:
            return Node(left=None, right=None, size=len(X), path_depth=height)
        idx = np.random.randint(X.shape[1])
        max_val = max(X[:, idx])
        min_val = min(X[:, idx])
        avg_val = np.mean(X[:, idx])

        # Choose split point between wider range of (max-avg) and (avg-min) to
        # increase chances of finding an outlier.
        if max_val - avg_val > avg_val - min_val:
            split = np.random.uniform(avg_val, max_val)
        else:
            split = np.random.uniform(min_val, avg_val)

        # from Hai Lu Ve's slack post
        left_idx = X[:,idx] < split
        right_idx = np.invert(left_idx)

        X_left = X[left_idx]
        X_right = X[right_idx]
        self.root = Node(idx=idx, split=split, path_depth=height, left=self.fit(X_left, height=height+1),
                    right=self.fit(X_right, height=height+1))
        return self.root

    def sep(self, f, i):
        left = f[:i+1]
        right = f[i+1:]
        return ((np.mean(left)-np.mean(right))**2)*.5*np.var(f)/(np.var(left)+np.var(right))



def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    For each threshold, compute the TPR and FPR to see if we've reached the
    desired TPR. If so, return the score threshold and FPR.
    """
    threshold = 1
    while True:
        y_hat = scores >= threshold
        TN, FP, FN, TP = confusion_matrix(y, y_hat).ravel()
        TPR = TP/(TP+FN)
        if TPR >= desired_TPR:
            FPR = FP/(TN+FP)
            return threshold, FPR
        threshold -= 0.01
