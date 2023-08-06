"""
The Calf and CalfCV classifiers.

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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def predict(X, w):
    """ Predict the classes from the weights and features

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples
        w : weights

    Returns:
        y_pred : the prediction of the ground truth, y

    """
    return np.sum(np.multiply(X, w), 1)


def hv_candidate(X, y, w, c):
    """ Find the auc of the weights augmented by the candidate vertex.

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples
        y : ground truth vector
        w : vetted weights
        c : scalar candidate weight

    Returns:
        auc : the prediction AUC

    """
    assert X.shape[1] - 1 == len(w), "X or w have the wrong shape"
    y_p = predict(X, w + [c])
    try:
        auc = roc_auc_score(y_true=y, y_score=y_p)
    except ValueError:
        auc = 0
    return auc


def hv_max(X, y, w, c):
    """Find the weight in c that augments w to maximize auc

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples
        y : ground truth vector
        w : vetted weights
        c : list of candidate weights

    Returns:
        (auc, v) : a tuple of auc at the best grid point v

    """
    res = [(hv_candidate(X, y, w, v), v) for v in c]
    return sorted(res, reverse=True)[0]


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
def fit_hv(X, y, grid):
    """ Find the weights that best fit X using points from grid

    Arguments:
        X : array-like, shape (n_samples, n_features)
            The training input features and samples.
        y : ground truth vector
        grid : a list or array of candidate weights

    Returns:
        auc, w : the weights that maximize auc, and the list of feature auc

    """
    feature_range = range(X.shape[1])
    w = []
    auc = []
    for i in feature_range:
        X_c = X[:, 0:i + 1]

        # granular approximation
        max_auc, w_c = hv_max(X_c, y, w, grid)

        # if the auc goes down then we skip the feature by weighting it at 0
        if auc and max_auc <= max(auc):
            w = w + [0]
        else:
            w = w + [w_c]
        auc = auc + [max_auc]
    return auc, w


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

            >>> cls.predict_proba(np.array([[3, 5]]))
            array([[1., 0.]])

        """
    def __init__(self, grid=(-1, 1), verbose=0):
        self.grid = [grid] if isinstance(grid, int) else grid
        self.verbose = verbose if isinstance(verbose, int) and verbose in [0, 1, 2, 3] else 0

    def fit(self, X, y):
        """ fit Calf to X and y

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples
            y : ground truth vector

        Returns:
            self

        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # fit and time the fit
        start = time.time()
        self.auc_, self.w_ = fit_hv(X, y, grid=self.grid)
        self.fit_time_ = time.time() - start

        self.coef_ = self.w_
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        """ Identify confidence scores for the samples

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the decision vector (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = np.array(
            minmax_scale(
                predict(X, self.w_),
                feature_range=(-1, 1)
            )
        )
        return scores

    def predict(self, X):
        """ Predict class labels for each sample

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the class prediction of X (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        if len(self.classes_) < 2:
            y_class = self.y_
        else:
            # and convert to [0, 1] classes.
            y_class = np.heaviside(self.decision_function(X), 0).astype(int)
            # get the class labels
            y_class = [self.classes_[x] for x in y_class]
        return np.array(y_class)

    def predict_proba(self, X):
        """ Identify the probability of each sample class label

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the class probabilities of X (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'X_', 'y_'])
        X = check_array(X)

        y_proba = np.array(
            minmax_scale(
                self.decision_function(X),
                feature_range=(0, 1)
            )
        )
        class_prob = np.column_stack((1 - y_proba, y_proba))
        return class_prob

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

        >>> cls.predict_proba(np.array([[3, 5]]))
        array([[1., 0.]])

    """

    def __init__(self, grid=(-1, 1), verbose=0):
        self.grid = [grid] if isinstance(grid, int) else grid
        self.verbose = verbose if isinstance(verbose, int) and verbose in [0, 1, 2, 3] else 0

    def fit(self, X, y):
        """ fit X and y

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples.
            y : ground truth vector

        Returns:
            self

        """

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)

        self.model_ = GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ('scaler', StandardScaler()),
                    ('classifier', Calf())
                ]
            ),
            param_grid={'classifier__grid': [self.grid]},
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

        if self.verbose > 0:
            print()
            print('=======================================')
            print('Objective best score', self.best_score_)
            print('Best coef_ ', self.best_coef_)
            print('Objective best params', self.model_.best_params_)

        return self

    def decision_function(self, X):
        """ Identify confidence scores for the samples

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the decision vector (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.decision_function(X)

    def predict(self, X):
        """ Predict class labels for each sample

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the class prediction of X (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.predict(X)

    def predict_proba(self, X):
        """ Identify the probability of each sample class label

        Arguments:
            X : array-like, shape (n_samples, n_features)
                The training input features and samples

        Returns:
            the class probabilities of X (n_samples)

        """
        check_is_fitted(self, ['is_fitted_', 'model_'])
        return self.model_.predict_proba(X)

    def _more_tags(self):
        return {
            'poor_score': True,
            'non_deterministic': True,
            'binary_only': True
        }
