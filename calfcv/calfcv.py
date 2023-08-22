"""
The Calf and CalfCV classifiers.

===============================================================
Author: Rolf Carlson, Carlson Research, LLC <hrolfrc@gmail.com>
License: 3-clause BSD
===============================================================

Calf implements a Coarse Approximation Linear Function. [1]
CalfCV optimizes weight selection through cross validation.

References
========================
[1] Jeffries, C.D., Ford, J.R., Tilson, J.L. et al.
A greedy regression algorithm with coarse weights offers novel advantages.
Sci Rep 12, 5440 (2022). https://doi.org/10.1038/s41598-022-09415-2

"""
import time

import numpy as np
from scipy.sparse import issparse, csr_array
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from multiprocessing import Pool, TimeoutError

X_s = None
""" Global matrix X, as shared memory during multiprocessing """

y_s = None
""" Global ground truth y, as shared memory during multiprocessing """

grid_s = None
""" Global grid with default [-1, 1], as shared memory during multiprocessing """


def predict(X, w):
    """ Predict the classes from the weights and features

    Parameters
    ----------
        X : array-like, shape (n_samples, n_features)
            The training input features and samples
        w : weights

    Returns
    -------
        y_pred : the prediction of the ground truth, y

    """
    if issparse(X):
        Z = X.multiply(csr_array(w))
        y_pred = Z.sum(1).flatten().tolist()[0]
    else:
        y_pred = np.sum(X * w, 1)
    return y_pred


def init_worker(X_shared, y_shared, grid_shared):
    """ worker process that shares sparse matrix X, ground truth y, and grid """
    global X_s, y_s, grid_s
    X_s = X_shared
    y_s = y_shared
    grid_s = grid_shared


def column_task(i):
    """ Get the auc from predicting column i

    Parameters
    ----------
        i : the column to evaluate for prediction of the ground truth

    Returns
    -------
        auc : prediction auc for column i
        w : weight at column i that gives the highest auc
        i : the column index

    """
    global X_s, y_s, grid_s

    V = X_s.getcol(i)  # sparse
    result = []
    for w in grid_s:
        Z = V * w
        y_score = np.nan_to_num(Z.toarray(), copy=False).ravel()
        result.append(
            (
                roc_auc_score(
                    y_true=y_s,
                    y_score=y_score
                ),
                time.time(),  # auc tie-breaker for max
                w
            )
        )

    auc, _, w = max(result)
    return auc, w, i


def fit_columns(X, y, grid):
    """ fit multiprocess

    """
    candidates = []
    with Pool(
            initializer=init_worker,
            initargs=(X, y, grid,),
            processes=40,
            maxtasksperchild=1000
    ) as pool:
        iter_obj = pool.imap_unordered(column_task, list(range(X.shape[1])))
        while True:
            try:
                # get the next result and abort if there is a timeout.
                # "Also if chunksize is 1 then the next() method of the iterator returned by the
                # imap() method has an optional timeout parameter: next(timeout) will raise
                # multiprocessing.TimeoutError if the result cannot be returned within timeout seconds."
                result = iter_obj.next(timeout=5)
            except StopIteration:
                break
            except TimeoutError:
                print("Timeout exceeding 5 seconds.  Skipping fit...")
            else:
                if result:
                    candidates.append(result)
    return sorted(candidates, reverse=True)


def fit_hv_sparse(X, y, grid, auc_tol=1e-6, order_col=False, verbose=False):
    """ Find the weights that best fit sparse X using points from grid

    Parameters
    ----------
        X : sparse array, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        grid : a list or array of candidate weights
        auc_tol : tolerance above max auc for inclusion of a feature index
        order_col : whether to order the columns by individual auc
        verbose : Boolean, print status

    Returns
    -------
        auc, w : the weights that maximize auc, and the list of feature auc

    """
    # preprocessing step
    if order_col:
        tups = fit_columns(X, y, grid)
        # skip individual column auc and expected weight, for now
        col_order = [i for _, _, i in tups]
    else:
        col_order = range(X.shape[1])

    # find the features and weights that maximize auc over the grid
    count = 0
    weights = []
    auc = []
    index = []
    U = X.getcol(0) * 0  # zero column with the correct shape
    for i in col_order:
        V = X.getcol(i)
        candidates = []
        for w in grid:
            Z = U + V * w
            y_score = np.nan_to_num(Z.toarray().ravel())
            candidates.append(
                (
                    roc_auc_score(y_true=y, y_score=y_score),
                    time.time(),  # sorted tie-breaker
                    Z,  # maintain sparse representation
                    w  # candidate weight
                )
            )
        max_auc, _, U, w_c = max(candidates)

        if not auc or max_auc > max(auc) + auc_tol:  # contribution needs to be non-random
            weights = weights + [w_c]
            index = index + [i]

            if auc and verbose:
                print(
                    'Count ', count, ' of ', X.shape[1],
                    ' fit feature ', i,
                    ' feature auc: ', round(max_auc, 4), ' > max auc: ', round(max(auc), 4),
                    ' weight: ', w_c,
                    ' selected features: ', len(index),
                    ' auc tol: ', auc_tol
                )
        else:
            if count % 100 == 0 and verbose:
                print(
                    'Count ', count, ' of ', X.shape[1],
                    '  max auc: ', round(max(auc), 4),
                    ' number of contributing features ', len(index)
                )

        count = count + 1
        auc = auc + [max_auc]

        # if the auc has exceeded 0.999 then stop.
        if max(auc) >= 0.999:
            if verbose:
                print('found ', len(index), 'features that contribute positive auc.')
                print('auc threshold reached, breaking ...')
            break

    return auc, weights, index


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
def fit_hv(X, y, grid, verbose=False):
    """ Find the weights that best fit X using points from grid

    Parameters
    ----------
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        grid : a list or array of candidate weights
        verbose : Boolean, print status

    Returns
    -------
        auc, w : the weights that maximize auc, and the list of feature auc

    """
    weights = []
    auc = []
    index = []
    U = np.empty((X.shape[0]))
    for i in range(X.shape[1]):
        V = X[:, i]
        candidates = []
        for w in grid:
            y_score = np.nan_to_num(U + V * w)
            candidates.append(
                (
                    roc_auc_score(y_true=y, y_score=y_score),
                    time.time(),  # sorted tie-breaker
                    y_score,
                    w
                )
            )
        max_auc, _, U, w_c = sorted(candidates, reverse=True)[0]

        if not auc or max_auc > max(auc):
            weights = weights + [w_c]
            index = index + [i]

        # the auc measures the best contribution from each feature
        auc = auc + [max_auc]

        if verbose:
            if max_auc > max(auc):
                print(
                    'fit feature ', i, ' of ', X.shape[1],
                    ' feature auc: ', round(max_auc, 4), ' > max auc: ', round(max(auc), 4),
                    ' weight: ', w_c,
                    'number of contributing features ', len(index)
                )
            else:
                print(
                    'fit feature ', i, ' of ', X.shape[1],
                    ' feature auc: ', round(max_auc, 4), ' <= max auc: ', round(max(auc), 4),
                    ' weight:', 0,
                    'number of contributing features ', len(index)
                )

        # if the auc has exceeded 0.999 then stop.
        if max(auc) >= 0.999:
            if verbose:
                print('found ', len(index), 'features that contribute positive auc.')
                print('auc threshold reached, breaking ...')
            break

    return auc, weights, index


# noinspection PyAttributeOutsideInit
class Calf(ClassifierMixin, BaseEstimator):
    """ Course approximation linear function

    CalfCV fits a linear model with coefficients  w = (w1, ..., wp)
    to maximize the AUC of the targets predicted by the linear function.

    Parameters
    ----------
        grid : the search grid.  Default is (-1, 1).

        verbose : 0 is silent.  1-3 are increasingly verbose

    Attributes
    ----------
        coef_ : array of shape (n_features, )
            Estimated coefficients for the linear fit problem.  Only
            one target should be passed, and this is a 1D array of length
            n_features.

        auc_ : array of shape (n_features, )
            The cumulative auc up to each feature.

        n_features_in_ : int
            Number of features seen during :term:`fit`.

        classes_ : list
            The unique class labels

        fit_time_ : float
            The number of seconds to fit X to y

    Notes
    -----
        The feature matrix must be centered at 0.  This can be accomplished with
        sklearn.preprocessing.StandardScaler, or similar.  No intercept is calculated.

    Examples
    --------
        >>> import numpy
        >>> from calfcv import Calf
        >>> from sklearn.datasets import make_classification as mc
        >>> X, y = mc(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
        >>> numpy.round(X[0:3, :], 2)
        array([[ 1.23, -0.76],
               [ 0.7 , -1.38],
               [ 2.55,  2.5 ]])

        >>> y[0:3]
        array([0, 0, 1])

        >>> cls = CalfCV().fit(X, y)
        >>> cls.score(X, y)
        0.7

        >>> cls.best_coef_
        [1, 1]

        >>> numpy.round(cls.best_score_, 2)
        0.82

        >>> cls.fit_time_ > 0
        True

        >>> cls.predict(np.array([[3, 5]]))
        array([0])

        >>> np.round(cls.predict_proba(np.array([[3, 5]])), 2)
        array([[0.73, 0.27]])

        """

    def __init__(
            self,
            grid=(-1, 1),
            auc_tol=1e-6,
            order_col=False,
            verbose=False
    ):
        """ Initialize Calf"""
        self.grid = [grid] if isinstance(grid, int) else grid
        self.auc_tol = auc_tol
        self.order_col = order_col
        self.verbose = verbose

    def fit(self, X, y):
        """ Fit the model according to the given training data.

        Parameters
        ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

            y : array-like of shape (n_samples,)
                Target vector relative to X.

        Returns
        -------
            self
                Fitted estimator.

        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        if self.verbose:
            print('fitting ', X.shape[1], ' features.')

        # fit and time the fit
        start = time.time()
        # the feature_index is the index of features with non-zero weights
        # the weights is the set of non-zero weights for features indexed by the feature_index
        # the feature index is used for slicing feature columns to achieve dimensionality
        # reduction when multiplying against the weights.
        if issparse(X):
            self.auc_, self.weights_, self.feature_index_ = fit_hv_sparse(
                X, y,
                grid=self.grid,
                auc_tol=self.auc_tol,
                order_col=self.order_col,
                verbose=self.verbose
            )
        else:
            self.auc_, self.weights_, self.feature_index_ = fit_hv(
                X, y,
                grid=self.grid,
                verbose=self.verbose
            )
        self.fit_time_ = time.time() - start

        # expand to get coefficients
        self.coef_ = [0] * X.shape[1]
        for i, w in zip(self.feature_index_, self.weights_):
            self.coef_[i] = w
        self.is_fitted_ = True

        if self.verbose:
            print('=======================================')
            print('First several coefficients ', self.weights_[0:min(len(self.weights_), 10)])
            print('Max AUC', max(self.auc_))
            print('Objective score', self.score(X, y))
            print('Fit time', self.fit_time_)

        return self

    def decision_function(self, X):
        """ Identify confidence scores for the samples

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns
        -------
            y_d : the decision vector (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])

        X = self._validate_data(X, accept_sparse="csr", reset=False)

        scores = np.array(
            minmax_scale(
                predict(X[:, self.feature_index_], self.weights_),
                feature_range=(-1, 1)
            )
        )
        return scores

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix for which we want to get the predictions.

        Returns
        -------
            y_pred : ndarray of shape (n_samples,)
                Vector containing the class labels for each sample.

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        if len(self.classes_) < 2:
            y_class = self.y_
        else:
            # convert to [0, 1] classes.
            y_class = np.heaviside(self.decision_function(X), 0).astype(int)
            # get the class labels
            y_class = [self.classes_[x] for x in y_class]
        return np.array(y_class)

    def predict_proba(self, X):
        """Probability estimates for samples in X.

        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where n_samples is the number of samples and
                n_features is the number of features.

        Returns
        -------
            T : array-like of shape (n_samples, n_classes)
                Returns the probability of the sample for each class in the model,
                where classes are ordered as they are in `self.classes_`.
                To create the probabilities, calf uses the same expit (sigmoid)
                function used by LogisticRegression in the binary case.
                https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression:~:text=1.1.11.1.%20Binary%20Case

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X, accept_sparse=True)
        y_proba = expit(self.decision_function(X))
        class_prob = np.column_stack((1 - y_proba, y_proba))
        return class_prob

    def transform(self, X):
        """ Reduce X to the features that contribute positive AUC.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns
        -------
            X_r : array of shape [n_samples, n_selected_features]
                The input samples with only the selected features.

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X, accept_sparse=True)

        return X[:, self.feature_index_]

    def fit_transform(self, X, y):
        """ Fit to the data, then reduce X to the features that contribute positive AUC.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

            y : array-like of shape (n_samples,)
                Target vector relative to X.

        Returns
        -------
            X_r : array of shape [n_samples, n_selected_features]
                The input samples with only the selected features.

        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }


# noinspection PyAttributeOutsideInit, PyUnresolvedReferences
class CalfCV(ClassifierMixin, BaseEstimator):
    """ Course approximation linear function with cross validation

    CalfCV fits a linear model with coefficients  w = (w1, ..., wp)
    to maximize the AUC of the targets predicted by the linear function.

    Parameters
    ----------
        grid : the search grid.  Default is (-1, 1).

        verbose : 0 is silent.  1-3 are increasingly verbose

    Attributes
    ----------
        best_coef_ : array of shape (n_features, )
            Estimated coefficients for the linear fit problem.  Only
            one target should be passed, and this is a 1D array of length
            n_features.

        best_score_ : float
            The best auc score over the cross validation

        best_auc_ : array of shape (n_features, )
            The cumulative auc by feature.

        n_features_in_ : int
            Number of features seen during :term:`fit`.

        classes_ : list
            The unique class labels

        fit_time_ : float
            The number of seconds to fit X to y

    Notes
    -----
        Only one processor is used due to a bug caused by "Python’s multiprocessing that
        does fork without exec". See, https://scikit-learn.org/stable/faq.html#id27

    Examples
    --------
        >>> import numpy
        >>> from calfcv import CalfCV
        >>> from sklearn.datasets import make_classification as mc
        >>> X, y = mc(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
        >>> numpy.round(X[0:3, :], 2)
        array([[ 1.23, -0.76],
               [ 0.7 , -1.38],
               [ 2.55,  2.5 ]])

        >>> y[0:3]
        array([0, 0, 1])

        >>> cls = CalfCV().fit(X, y)
        >>> cls.score(X, y)
        0.7

        >>> numpy.round(cls.best_score_, 2)
        0.82

        >>> numpy.round(cls.best_auc_, 2)
        array([0.53, 0.8 ])

        >>> cls.best_coef_
        [1, 1]

        >>> numpy.round(cls.best_score_, 2)
        0.82

        >>> cls.fit_time_ > 0
        True

        >>> cls.predict(np.array([[3, 5]]))
        array([0])

        >>> np.round(cls.predict_proba(np.array([[3, 5]])), 2)
        array([[0.73, 0.27]])

    """

    def __init__(
            self,
            grid=(-1, 1),
            auc_tol=1e-6,
            order_col=False,
            verbose=False
    ):
        """ Initialize Calf"""
        self.grid = [grid] if isinstance(grid, int) else grid
        self.auc_tol = auc_tol
        self.order_col = order_col
        self.verbose = verbose

    def fit(self, X, y):
        """ Fit the model according to the given training data.

        Parameters
        ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

            y : array-like of shape (n_samples,)
                Target vector relative to X.

        Returns
        -------
            self
                Fitted estimator.

        """

        X, y = check_X_y(X, y, accept_sparse=True)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)

        parameter_grid = {
            'classifier__grid': [self.grid],
            'classifier__auc_tol': [self.auc_tol],
            'classifier__order_col': [self.order_col],
            'classifier__verbose': [self.verbose]
        }

        if issparse(X):
            # scaling the data makes the matrix dense
            self.model_ = GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ('classifier', Calf())
                    ]
                ),
                param_grid=parameter_grid,
                scoring="roc_auc",
                verbose=self.verbose
            )
        else:
            self.model_ = GridSearchCV(
                estimator=Pipeline(
                    steps=[
                        ('scaler', StandardScaler()),
                        ('classifier', Calf())
                    ]
                ),
                param_grid=parameter_grid,
                scoring="roc_auc",
                verbose=self.verbose
            )

        # fit and time
        start = time.time()
        self.model_.fit(X, y)
        self.fit_time_ = time.time() - start

        self.is_fitted_ = True

        # "best_score_: Mean cross-validated score of the best_estimator"
        # "https://stackoverflow.com/a/50233868/12865125"
        self.best_score_ = self.model_.best_score_
        self.best_coef_ = self.model_.best_estimator_['classifier'].coef_
        self.best_auc_ = self.model_.best_estimator_['classifier'].auc_

        if self.verbose:
            print()
            print('=======================================')
            print('Objective best score', self.best_score_)
            print('Best coef_ ', self.best_coef_)
            print('Objective best params', self.model_.best_params_)

        return self

    def decision_function(self, X):
        """ Identify confidence scores for the samples

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns
        -------
            y_d : the decision vector (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.decision_function(X)

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix for which we want to get the predictions.

        Returns
        -------
            y_pred : ndarray of shape (n_samples,)
                Vector containing the class labels for each sample.

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Probability estimates for samples in X.

        Parameters
        ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where n_samples is the number of samples and
                n_features is the number of features.

        Returns
        -------
            T : array-like of shape (n_samples, n_classes)
                Returns the probability of the sample for each class in the model,
                where classes are ordered as they are in `self.classes_`.

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.predict_proba(X)

    def transform(self, X):
        """ Reduce X to the features that contribute positive AUC.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns
        -------
            X_r : array of shape [n_samples, n_selected_features]
                The input samples with only the selected features.

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.transform(X)

    def fit_transform(self, X, y):
        """ Fit to the data, then reduce X to the features that contribute positive AUC.

        Parameters
        ----------
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

            y : array-like of shape (n_samples,)
                Target vector relative to X.

        Returns
        -------
            X_r : array of shape [n_samples, n_selected_features]
                The input samples with only the selected features.

        """
        return self.fit(X, y).model_.transform(X)

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }
