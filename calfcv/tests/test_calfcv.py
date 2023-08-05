import numpy
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from calfcv.calfcv import Calf, CalfCV


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
