.. -*- mode: rst -*-

|ReadTheDocs|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/calfcv/badge/?version=latest
.. _ReadTheDocs: https://calfcv.readthedocs.io/en/latest/?badge=latest

CalfCV
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

Currently, CalfCV provides classification and prediction for two classes, the binomial case. Multinomial classification with more than two cases is not implemented.

The feature matrix is scaled to have zero mean and unit variance. Cross-validation is implemented to identify optimal score and coefficients. CalfCV is designed for use with scikit-learn_ pipelines and composite estimators.

.. _scikit-learn: https://scikit-learn.org

Example
------------------
.. code:: ipython2

    from calfcv import CalfCV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import numpy as np

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

The best score is the best average auc.

.. code:: ipython2

    cls = CalfCV().fit(X_train, y_train)
    cls.best_score_




.. parsed-literal::

    0.95


The coefficients for the best score are in ``[-1, 0, 1]``.


.. code:: ipython2

    cls.best_coef_




.. parsed-literal::

    [-1, 1, 0, 1, 1]



The probabilities of class 1 are in the last row
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We vertically stack the ground truth on the top with the probabilities
of class 1 on the bottom. We show the first 5 entries.



.. code:: ipython2

    np.round(np.vstack((y_train, cls.predict_proba(X_train).T))[:, 0:5], 2)




.. parsed-literal::

    array([[0.  , 1.  , 1.  , 0.  , 0.  ],
           [0.71, 0.05, 0.19, 0.34, 0.54],
           [0.29, 0.95, 0.81, 0.66, 0.46]])



Predicting the training data should give a slightly higher score than the best_score\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

That is what we see here. The reason is that best_score\_ is a mean of
auc over the cross validation.

.. code:: ipython

    roc_auc_score(y_true=y_train, y_score=cls.predict_proba(X_train)[:, 1])




.. parsed-literal::

    0.9750000000000001



The classifier will likely produce a lower score on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often we get a lower score on the unseen data, but in this case we
get a higher score.

.. code:: ipython2

    roc_auc_score(y_true=y_test, y_score=cls.predict_proba(X_test)[:, 1])




.. parsed-literal::

    1.0



Score using classes is lower than score using probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ground truth is on the top and the predicted class is on the bottom. Sample 6 of y_test is predicted incorrectly but the others are correct.

.. code:: ipython2

    y_pred = cls.predict(X_test)
    np.vstack((y_test, y_pred))




.. parsed-literal::

    array([[0, 1, 1, 0, 1, 0, 0, 0],
           [0, 1, 1, 0, 1, 0, 1, 0]])




.. code:: ipython2

    roc_auc_score(y_true=y_test, y_score=y_pred)




.. parsed-literal::

    0.9




Authors
------------------
The CALF algorithm was designed by Clark D. Jeffries, John R. Ford, Jeffrey L. Tilson, Diana O. Perkins, Darius M. Bost, Dayne L. Filer and Kirk C. Wilhelmsen. This python implementation was written by Rolf Carlson.

References
------------------
Jeffries, C.D., Ford, J.R., Tilson, J.L. et al. A greedy regression algorithm with coarse weights offers novel advantages. Sci Rep 12, 5440 (2022). https://doi.org/10.1038/s41598-022-09415-2



