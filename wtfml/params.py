from dataclasses import dataclass
from typing import Union

import xgboost as xgb
from optuna import trial
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .enums import ProblemType


@dataclass
class Params:
    model: Union[
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        LinearSVC,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        KNeighborsClassifier,
        KNeighborsRegressor,
        LinearSVR,
        Lasso,
        Ridge,
        GaussianNB,
        MultinomialNB,
        LinearRegression,
        LogisticRegression,
        SGDClassifier,
        SGDRegressor,
        xgb.XGBClassifier,
        xgb.XGBRegressor,
    ]
    problem_type: ProblemType

    def _xgb_base_params(self):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 500),
            "n_estimators": trial.suggest_categorical("n_estimators", [7000, 15000, 20000]),
        }
        return params

    def _xgb_cpu_params(self):
        params = self._xgb_base_params()
        params["tree_method"] = trial.suggest_categorical("tree_method", ["exact", "approx", "hist"])
        params["booster"] = trial.suggest_categorical("booster", ["gbtree", "gblinear"])
        if params["booster"] == "gbtree":
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        return params

    def _xgb_gpu_params(self):
        params = self._xgb_base_params()
        params["tree_method"] = "gpu_hist"
        params["gpu_id"] = 0
        params["predictor"] = "gpu_predictor"
        return params

    def _rf_base_params(self):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 10000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        return params

    def _rf_classifier_params(self):
        params = self._rf_base_params()
        params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        return params

    def _rf_regressor_params(self):
        params = self._rf_base_params()
        params["criterion"] = trial.suggest_categorical(
            "criterion",
            [
                "squared_error",
                "absolute_error",
                "poisson",
            ],
        )
        return params

    def _et_base_params(self):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 10000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        return params

    def _et_classifier_params(self):
        params = self._et_base_params()
        params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        return params

    def _et_regressor_params(self):
        params = self._et_base_params()
        params["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
        return params

    def _dt_base_params(self):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2", None]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        }
        return params

    def _dt_classifier_params(self):
        params = self._dt_base_params()
        params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        return params

    def _dt_regressor_params(self):
        params = self._dt_base_params()
        params["criterion"] = trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"]
        )
        return params

    def _linear_regression_params(self):
        params = {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        return params

    def _logistic_regression_params(self):
        params = {
            "C": trial.suggest_float("C", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "solver": trial.suggest_categorical("solver", ["liblinear", "saga", "newton-cg", "lbfgs"]),
        }
        if params["solver"] == "liblinear":
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        elif params["solver"] == "saga":
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
        elif params["solver"] == "newton-cg":
            params["penalty"] = trial.suggest_categorical("penalty", ["l2", "none"])
        elif params["solver"] == "lbfgs":
            params["penalty"] = trial.suggest_categorical("penalty", ["l2", "none"])
        else:
            raise ValueError("Unknown solver: {}".format(params["solver"]))
        return params

    def _linear_svc_params(self):
        params = {
            "C": trial.suggest_float("C", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        }
        return params

    def _linear_svr_params(self):
        params = {
            "C": trial.suggest_float("C", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "loss": trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
            "epsilon": trial.suggest_float("epsilon", 1e-8, 1e-1, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        }
        return params

    def _lasso_params(self):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        }
        return params

    def _ridge_params(self):
        params = {
            "alpha": trial.suggest_float("alpha", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        }
        return params

    def _gaussian_nb_params(self):
        params = {}
        return params

    def _multinomial_nb_params(self):
        params = {}
        return params

    def _sgd_base_params(self):
        params = {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e3, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000),
            "tol": trial.suggest_float("tol", 1e-8, 1e-1, log=True),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]
            ),
            "eta0": trial.suggest_float("eta0", 1e-8, 1e3, log=True),
            "power_t": trial.suggest_float("power_t", 1e-8, 1e3, log=True),
            "early_stopping": trial.suggest_categorical("early_stopping", [True, False]),
            "validation_fraction": trial.suggest_float("validation_fraction", 1e-8, 1e-1, log=True),
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 1, 100),
        }
        return params

    def _sgd_classifier_params(self):
        params = self._sgd_base_params()
        params["loss"] = trial.suggest_categorical(
            "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
        )
        return params

    def _sgd_regressor_params(self):
        params = self._sgd_base_params()
        params["loss"] = trial.suggest_categorical("loss", ["squared_loss", "huber", "epsilon_insensitive"])
        return params

    def _knn_base_params(self):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 1, 100),
            "p": trial.suggest_categorical("p", [1, 2]),
            "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
        }
        return params

    def _knn_classifier_params(self):
        params = self._knn_base_params()
        return params

    def _knn_regressor_params(self):
        params = self._knn_base_params()
        return params

    def fetch_params(self):
        if isinstance(self.model, RandomForestClassifier):
            params = self._rf_classifier_params()
        elif isinstance(self.model, RandomForestRegressor):
            params = self._rf_regressor_params()
        elif isinstance(self.model, ExtraTreesClassifier):
            params = self._et_classifier_params()
        elif isinstance(self.model, ExtraTreesRegressor):
            params = self._et_regressor_params()
        elif isinstance(self.model, DecisionTreeClassifier):
            params = self._dt_classifier_params()
        elif isinstance(self.model, DecisionTreeRegressor):
            params = self._dt_regressor_params()
        elif isinstance(self.model, LinearRegression):
            params = self._linear_regression_params()
        elif isinstance(self.model, LogisticRegression):
            params = self._logistic_regression_params()
        elif isinstance(self.model, LinearSVC):
            params = self._linear_svc_params()
        elif isinstance(self.model, LinearSVR):
            params = self._linear_svr_params()
        elif isinstance(self.model, Lasso):
            params = self._lasso_params()
        elif isinstance(self.model, Ridge):
            params = self._ridge_params()
        elif isinstance(self.model, GaussianNB):
            params = self._gaussian_nb_params()
        elif isinstance(self.model, MultinomialNB):
            params = self._multinomial_nb_params()
        elif isinstance(self.model, SGDClassifier):
            params = self._sgd_classifier_params()
        elif isinstance(self.model, SGDRegressor):
            params = self._sgd_regressor_params()
        elif isinstance(self.model, KNeighborsClassifier):
            params = self._knn_classifier_params()
        elif isinstance(self.model, KNeighborsRegressor):
            params = self._knn_regressor_params()
        else:
            raise ValueError("Unknown model: {}".format(self.model))

        return params
