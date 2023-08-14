.. -*- mode: rst -*-

|Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Codecov| image:: https://codecov.io/gh/hrolfrc/calfcv/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/hrolfrc/calfcv

.. |CircleCI| image:: https://circleci.com/gh/hrolfrc/calfcv.svg?style=shield
.. _CircleCI: https://circleci.com/gh/hrolfrc/calfcv

.. |ReadTheDocs| image:: https://readthedocs.org/projects/calfcv/badge/?version=latest
.. _ReadTheDocs: https://calfcv.readthedocs.io/en/latest/?badge=latest


Calf, CalfCV
#####################################

A binomial classifier that implements the Coarse Approximation Linear Function (CALF).

Contact
------------------
Rolf Carlson hrolfrc@gmail.com

Install
------------------
Use pip to install calfcv.

``pip install calfcv``

Introduction
------------------
This is a python implementation of the Coarse Approximation Linear Function (CALF). The implementation is based on the greedy forward selection algorithm described in the paper referenced below.

Two classes are provided: Calf, and CalfCV.  Calf provides classification and prediction for two classes, the binomial case. Multinomial classification with more than two cases is not implemented. Calf provides a transform method that can be used for feature selection and dimensionality reduction of data sets.  Calf requires that the feature matrix be scaled to have zero mean and unit variance.  CalfCV provides the same functionality as Calf, but CalfCV includes built in data scaling and cross-validation.  Choose Calf over CalfCV if you are optimizing hyperparameters over a grid using cross-validation.

Both Calf and CalfCV are designed for use with scikit-learn_ pipelines and composite estimators.

.. _scikit-learn: https://scikit-learn.org

Example
===========

.. code:: ipython2

    from calfcv import CalfCV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

Make a classification problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    seed = 42
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

Train the classifier
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls = CalfCV().fit(X_train, y_train)

Get the score on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls.score(X_test, y_test)




.. parsed-literal::

    0.875


Authors
------------------
The CALF algorithm was designed by Clark D. Jeffries, John R. Ford, Jeffrey L. Tilson, Diana O. Perkins, Darius M. Bost, Dayne L. Filer and Kirk C. Wilhelmsen. This python implementation was written by Rolf Carlson.

References
------------------
Jeffries, C.D., Ford, J.R., Tilson, J.L. et al. A greedy regression algorithm with coarse weights offers novel advantages. Sci Rep 12, 5440 (2022). https://doi.org/10.1038/s41598-022-09415-2



