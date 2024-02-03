import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def fit_lr(features, y):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(
                             random_state=0, max_iter=1000000, multi_class='ovr')
                         )
    pipe.fit(features, y)
    return pipe


def fit_svm(features, y):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    svm = SVC(C=np.inf, gamma='scale')
    if train_size // nb_classes < 5 or train_size < 50:
        return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            svm, {
                'C': [0.001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )

    grid_search.fit(features, y)
    return grid_search.best_estimator_
