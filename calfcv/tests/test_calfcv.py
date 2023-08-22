# ===============================================================
# Author: Rolf Carlson, Carlson Research, LLC <hrolfrc@gmail.com>
# License: 3-clause BSD
# ===============================================================

import numpy
import pytest
from scipy.sparse import issparse
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from calfcv.calfcv import Calf, CalfCV


@pytest.fixture
def sparse_data():
    """ Make a sparse classification problem for visual inspection. """

    text = [
        "It was the best of times",
        "it was the worst of times",
        "it was the age of wisdom",
        "it was the age of foolishness"
    ]

    X = TfidfVectorizer().fit_transform(text)
    # samples 0 and 2 are positive
    # samples 1 and 3 are negative
    y = [1, 0, 1, 0]

    return X, y


def test_calf_sparse(sparse_data):
    X, y = sparse_data

    assert issparse(X) == True
    assert X.shape == (4, 10)

    # the sparse representation
    # (0, 4)	0.3169454420370736
    # (0, 1)	0.6073596130854014
    # (0, 5)	0.3169454420370736
    # (0, 7)	0.3169454420370736
    # (0, 3)	0.3169454420370736
    # (1, 9)	0.6073596130854014
    # (1, 6)	0.4788492951654494
    # (1, 4)	0.3169454420370736
    # (1, 5)	0.3169454420370736
    # (1, 7)	0.3169454420370736
    # (1, 3)	0.3169454420370736
    # (2, 8)	0.6073596130854014
    # (2, 0)	0.4788492951654494
    # (2, 4)	0.3169454420370736
    # (2, 5)	0.3169454420370736
    # (2, 7)	0.3169454420370736
    # (2, 3)	0.3169454420370736
    # (3, 2)	0.6073596130854014
    # (3, 0)	0.4788492951654494
    # (3, 4)	0.3169454420370736
    # (3, 5)	0.3169454420370736
    # (3, 7)	0.3169454420370736
    # (3, 3)	0.3169454420370736

    # the dense representation with 4 rows and 10 columns
    # assert repr(np.round(X.toarray(), 2)) == [
    #     [0., 0.61, 0., 0.32, 0.32, 0.32, 0.48, 0.32, 0., 0.],
    #     [0., 0., 0., 0.32, 0.32, 0.32, 0.48, 0.32, 0., 0.61],
    #     [0.48, 0., 0., 0.32, 0.32, 0.32, 0., 0.32, 0.61, 0.],
    #     [0.48, 0., 0.61, 0.32, 0.32, 0.32, 0., 0.32, 0., 0.]
    # ]

    clf = Calf()
    assert clf.grid == (-1, 1)

    clf.fit(X, y)
    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    assert clf.auc_ == [0.5, 0.875, 1.0]
    assert clf.coef_ == [1, 1, -1, 0, 0, 0, 0, 0, 0, 0]

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


@pytest.fixture
def data():
    """ Make a classification problem for visual inspection. """
    X, y = make_classification(
        n_samples=10,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_classes=2,
        hypercube=True,
        random_state=8
    )
    return X, y


# noinspection DuplicatedCode
def test_calf(data):
    X, y = data
    clf = Calf()
    assert clf.grid == (-1, 1)

    clf.fit(X, y)
    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)

    # check the first several entries of y
    y = data[1]
    assert all(y == [0, 0, 1, 1, 0, 1, 1, 0, 0, 1])

    # Get the prediction
    y_score = numpy.round(clf.fit(X, y).predict(X), 2)
    assert all(y_score == [0, 0, 1, 1, 0, 1, 1, 1, 0, 1])

    # check shape
    assert len(y_score) == len(y) == X.shape[0]

    auc_p = roc_auc_score(y_true=y, y_score=y_score)
    assert numpy.round(auc_p, 2) == 0.9

    # expect 1-2 informative features to be found
    X_r = clf.transform(X)
    assert X_r.shape[1] == 3
    assert X_r.shape[0] == len(y)

    X_r = clf.fit_transform(X, y)
    assert X_r.shape[1] == 3
    assert X_r.shape[0] == len(y)


# noinspection DuplicatedCode
def test_calfcv(data):
    X, y = data
    clf = CalfCV()
    assert clf.grid == (-1, 1)

    clf.fit(X, y)
    assert hasattr(clf, 'is_fitted_')
    assert hasattr(clf, 'classes_')
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)

    # expect 1-2 informative features to be found
    X_r = clf.transform(X)
    assert X_r.shape[1] == 3
    assert X_r.shape[0] == len(y)

    X_r = clf.fit_transform(X, y)
    assert X_r.shape[1] == 3
    assert X_r.shape[0] == len(y)
