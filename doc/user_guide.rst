.. title:: User guide : contents

.. _user_guide:

User Guide
==========

Make a classification problem
-----------------------------

.. code:: ipython2

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from calfcv import CalfCV

.. code:: ipython2

    seed = 45
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_redundant=2,
        n_classes=2,
        random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

Train the classifier
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    cls = CalfCV().fit(X_train, y_train)

Get the score for class prediction on unseen data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython2

    cls.score(X_test, y_test)




.. parsed-literal::

    0.92



Class probabilities
^^^^^^^^^^^^^^^^^^^

We vertically stack the ground truth on the top with the probabilities
of class 1 on the bottom. The first five entries are shown.

.. code:: ipython2

    np.round(np.vstack((y_train, cls.predict_proba(X_train).T))[:, 0:5], 2)




.. parsed-literal::

    array([[1.  , 1.  , 0.  , 0.  , 0.  ],
           [0.21, 0.46, 1.  , 0.8 , 0.71],
           [0.79, 0.54, 0.  , 0.2 , 0.29]])



.. code:: ipython2

    roc_auc_score(y_true=y_train, y_score=cls.predict_proba(X_train)[:, 1])




.. parsed-literal::

    0.9722617354196301



Predict the classes
^^^^^^^^^^^^^^^^^^^

The ground truth is on the top and the predicted classes are on the
bottom. The first five entries are shown.

.. code:: ipython2

    y_pred = cls.predict(X_test)
    np.vstack((y_test, y_pred))[:, 0:5]




.. parsed-literal::

    array([[0, 0, 0, 1, 0],
           [0, 0, 0, 1, 0]])



The class prediction is expected to be lower than the auc prediction.

.. code:: ipython2

    roc_auc_score(y_true=y_test, y_score=y_pred)




.. parsed-literal::

    0.9198717948717948



Reproduce the AUC from example of the Calf paper [1]
----------------------------------------------------

While calfpy yields an auc of 0.875 in example 1 from the paper, calfcv
produces an auc of 0.9796875.

.. code:: ipython2

    input_file = "../../data/n2.csv"
    df = pd.read_csv(input_file, header=0, sep=",")

    # The input data is everything except the first column
    X = df.loc[:, df.columns != 'ctrl/case']
    # The outcome or diagnoses are in the first ctrl/case column
    Y = df['ctrl/case']

    # The header row is the feature set
    features = list(X.columns)

    # label the outcomes
    Y_names = Y.replace({0: 'non_psychotic', 1: 'pre_psychotic'})

    # glmnet requires float64
    x = X.to_numpy(dtype='float64')
    y = Y.to_numpy(dtype='float64')


Features
~~~~~~~~

Here we look at the feature names, number of features, shape, category
balance, and probability of choosing the positive category by chance.

.. code:: ipython2

    features[0:5]




.. parsed-literal::

    ['ADIPOQ', 'SERPINA3', 'AMBP', 'A2M', 'ACE']



.. code:: ipython2

    x.size




.. parsed-literal::

    9720



.. code:: ipython2

    x.shape




.. parsed-literal::

    (72, 135)



Category Balance
~~~~~~~~~~~~~~~~

.. code:: ipython2

    print(list(Y).count(1), list(Y).count(0))


.. parsed-literal::

    32 40


.. code:: ipython2

    len(y)




.. parsed-literal::

    72



AUC improvement
~~~~~~~~~~~~~~~

CalfCV improves on the calfpy auc of 0.875 from example 1 of the paper.

.. code:: ipython2

    y_pred = CalfCV().fit(x, y).predict_proba(x)
    roc_auc_score(y, y_pred[:, 1])




.. parsed-literal::

    0.9796875



References
----------

[1] Jeffries, C.D., Ford, J.R., Tilson, J.L. et al. A greedy regression
algorithm with coarse weights offers novel advantages. Sci Rep 12, 5440
(2022). https://doi.org/10.1038/s41598-022-09415-2
