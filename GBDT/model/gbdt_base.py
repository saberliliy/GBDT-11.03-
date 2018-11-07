# -*- coding: utf-8 -*-
"""
@Author: tushushu
@Date: 2018-08-13 10:29:29
@Last Modified by:   tushushu
@Last Modified time: 2018-08-13 10:29:29
"""
from regression_tree import RegressionTree
from random import choices
from copy import copy
from  utils import list_split
from  utils  import  *


class GradientBoostingBase(object):
    def __init__(self):
        """GBDT base class.
        http://statweb.stanford.edu/~jhf/ftp/stobst.pdf

        Attributes:
            trees {list}: 1d list with RegressionTree objects.
            lr {float}: Learning rate.
            init_val {float}: Initial value to predict.
            fn {function}: A function wrapper for prediction.
        """

        self.trees = None
        self.lr = None
        self.init_val = None
        self.fn = lambda x: x

    def _get_init_val(self, y):
        """Calculate the initial prediction of y.

        Arguments:
            y {list} -- 1D list with int or float.

        Returns:
            NotImplemented
        """
        return sum(y) / len(y)

    def _match_node(self, row, tree):
        """Find the leaf node that the sample belongs to.

        Arguments:
            row {list} -- 1D list with int or float.
            tree {RegressionTree}

        Returns:
            regression_tree.Node
        """

        nd = tree.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd

    def _get_leaves(self, tree):  #获取所有的叶子节点
        """Gets all leaf nodes of a regression tree.

        Arguments:
            tree {RegressionTree}

        Returns:
            list -- 1D list with regression_tree.Node objects.
        """

        nodes = []
        que = [tree.root]
        while que:
            node = que.pop(0)
            if node.left is None or node.right is None:
                nodes.append(node)
                continue
            left_node = node.left
            right_node = node.right
            que.append(left_node)
            que.append(right_node)
        return nodes

    def _divide_regions(self, tree, nodes, X):
        """Divide indexes of the samples into corresponding leaf nodes
        of the regression tree.

        Arguments:
            tree {RegressionTree}
            nodes {list} -- 1D list with regression_tree.Node objects.
            X {list} -- 2d list object with int or float.

        Returns:
            dict -- e.g. {node1: [1, 3, 5], node2: [2, 4, 6]...}
        """

        regions = {node: [] for node in nodes}
        for i, row in enumerate(X):      #将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            node = self._match_node(row, tree)
            regions[node].append(i)
        return regions      #avg   regions[avg]=[x[idx]]

    def _get_score(self, idxs, y_hat, residuals):
        """Calculate the regression tree leaf node value.

        Arguments:
            idxs {list} -- 1D list with int.

        Returns:
            NotImplemented
        """
        numerator = denominator = 0
        for idx in idxs:
            numerator += residuals[idx]
            #denominator += y_hat[idx] * (1 - y_hat[idx])
        return numerator / len(idxs)


    def _update_score(self, tree, X, y_hat, residuals):
        """update the score of regression tree leaf node.

        Arguments:
            tree {RegressionTree}
            X {list} -- 2d list with int or float.
            y_hat {list} -- 1d list with float.
            residuals {list} -- 1d list with float.
        """

        nodes = self._get_leaves(tree)    #obtain leaves

        regions = self._divide_regions(tree, nodes, X)           #obtain divide regions value
        for node, idxs in regions.items():
            node.score = self._get_score(idxs, y_hat, residuals)
        tree._get_rules()

    def _get_residuals(self, y, y_hat):
        """Update residuals for each iteration.

        Arguments:
            y {list} -- 1d list with int or float.
            y_hat {list} -- 1d list with float.

        Returns:
            list -- residuals
        """

        return [yi-self.fn(y_hat_i) for yi, y_hat_i in zip(y, y_hat)]

    def fit(self, X, y, n_estimators, lr, max_depth, min_samples_split,
            subsample=None):
        """Build a gradient boost decision tree.

        Arguments:
            X {list} -- 2d list with int or float.
            y {list} -- 1d list object with int or float.
            n_estimators {int} -- number of trees.
            lr {float} -- Learning rate
            max_depth {int} -- The maximum depth of the tree.
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node.


        Keyword Arguments:
            subsample {float} -- Subsample rate without replacement.
            (default: {None})
        """

        # Calculate the initial prediction of y
        self.init_val = self._get_init_val(y)
        # Initialize y_hat
        n = len(y)
        y_hat = [self.init_val] * n
        # Initialize the residuals
        residuals = self._get_residuals(y, y_hat)   #residuals
        # Train Regression Trees
        self.trees = []
        self.lr = lr
        for _ in range(n_estimators):
            # Sampling with replacement
            idx = range(n)                    #number of data
            if subsample is not None:
                k = int(subsample * n)
                idx = choices(population=idx, k=k)
            X_sub = [X[i] for i in idx]                  # X set
            residuals_sub = [residuals[i] for i in idx]   #residuals set
            y_hat_sub = [y_hat[i] for i in idx]           #y_hat    set
            # Train a Regression Tree by sub-sample of X, residuals
            tree = RegressionTree()
            tree.fit(X_sub, residuals_sub, max_depth, min_samples_split)
            # Update scores of tree leaf nodes
            self._update_score(tree, X_sub, y_hat_sub, residuals_sub)
            # Update y_hat
            y_hat = [y_hat_i + lr * res_hat_i for y_hat_i,
                     res_hat_i in zip(y_hat, tree.predict(X))]
            # Update residuals
            residuals = self._get_residuals(y, y_hat)
            self.trees.append(tree)

    def _predict(self, Xi):
        """Auxiliary function of predict.

        Arguments:
            Xi {list} -- 1D list with int or float.

        Returns:
            int or float -- prediction of yi.
        """

        # Sum y_hat with residuals of each tree and then calulate sigmoid value
        return self.fn(self.init_val +
                       sum(self.lr * tree._predict(Xi) for tree in self.trees))

    def predict(self, X):
        """Get the prediction of y.

        Arguments:
            X {list} -- 2d list object with int or float.

        Returns:
            NotImplemented
        """
        return [self._predict(Xi) for Xi in X]
def main():
    print("Tesing the accuracy of GBDT regressor...")
    X, y = load_boston_house_prices()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    reg = GradientBoostingBase()
    reg.fit(X=X_train, y=y_train, n_estimators=8,
            lr=0.5, max_depth=2, min_samples_split=2,subsample=0.8)
    get_r2(reg, X_test, y_test)
if __name__ == '__main__':
  main()