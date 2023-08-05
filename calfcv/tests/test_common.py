import pytest
from sklearn.utils.estimator_checks import check_estimator

from calfcv.calfcv import Calf, CalfCV


@pytest.mark.parametrize(
    "estimator",
    [Calf(), CalfCV()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
