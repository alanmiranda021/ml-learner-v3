"""Catálogo de algoritmos e grades de hiperparâmetros.

Equivalente aos Regression/Classification Learner do MATLAB.
"""
from __future__ import annotations

from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression, Ridge)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

REGRESSION = {
    "Regressão Linear": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "Lasso": (Lasso(max_iter=10000), {"alpha": [1e-4, 1e-3, 1e-2, 0.1, 1]}),
    "Árvore de Decisão": (DecisionTreeRegressor(random_state=0),
                          {"max_depth": [3, 5, 8, 12, None]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 9, 15]}),
    "SVR (RBF)": (SVR(), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}),
    "Random Forest": (RandomForestRegressor(random_state=0, n_jobs=1),
                      {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Extra Trees": (ExtraTreesRegressor(random_state=0, n_jobs=1),
                    {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=0),
                          {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
    "HistGradient Boosting": (HistGradientBoostingRegressor(random_state=0),
                              {"max_iter": [100, 300], "learning_rate": [0.05, 0.1]}),
    "Rede Neural (MLP)": (MLPRegressor(max_iter=1000, random_state=0),
                          {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}),
    "Gaussian Process (Kriging)": (GaussianProcessRegressor(kernel=Matern(nu=1.5), normalize_y=True, random_state=0),
                                   {"kernel": [Matern(nu=1.5), RBF()]}),
    "XGBoost Regressor": (XGBRegressor(random_state=0, n_jobs=1),
                          {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]}),
    "LightGBM Regressor": (LGBMRegressor(random_state=0, n_jobs=1, verbose=-1),
                           {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
}

CLASSIFICATION = {
    "Regressão Logística": (LogisticRegression(max_iter=2000, n_jobs=1),
                            {"C": [0.1, 1, 10]}),
    "Naive Bayes": (GaussianNB(), {}),
    "Árvore de Decisão": (DecisionTreeClassifier(random_state=0),
                          {"max_depth": [3, 5, 8, 12, None]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 9, 15]}),
    "SVM (RBF)": (SVC(probability=True), {"C": [0.1, 1, 10, 100],
                                          "gamma": ["scale", 0.01, 0.1]}),
    "Random Forest": (RandomForestClassifier(random_state=0, n_jobs=1),
                      {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Extra Trees": (ExtraTreesClassifier(random_state=0, n_jobs=1),
                    {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=0),
                          {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1]}),
    "HistGradient Boosting": (HistGradientBoostingClassifier(random_state=0),
                              {"max_iter": [100, 300], "learning_rate": [0.05, 0.1]}),
    "Rede Neural (MLP)": (MLPClassifier(max_iter=1000, random_state=0),
                          {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2]}),
}
