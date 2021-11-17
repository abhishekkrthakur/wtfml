from dataclasses import dataclass

import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .enums import DataType, TaskType


class WTFModel:
    def __init__(self, x, normalize, one_hot_encode):
        self.x = x
        self.normalize = normalize
        self.one_hot_encode = one_hot_encode


@dataclass
class WTFModels:
    data_type: DataType
    task: TaskType

    def fetch_models(self):
        if self.data_type == DataType.tabular:
            if self.task == TaskType.classification:
                return self._tabular_classification_models()
            elif self.task == TaskType.regression:
                return self._tabular_regression_models()
        else:
            raise NotImplementedError(f"Data type {self.data_type} is not yet implemented")

    def fetch_normalizers(self):
        if self.data_type == DataType.tabular:
            return self._tabular_normalizers()
        else:
            raise NotImplementedError(f"Data type {self.data_type} is not yet implemented")

    def _tabular_normalizers(self):
        normalizers = [
            StandardScaler,
            MinMaxScaler,
            RobustScaler,
        ]
        return normalizers

    def _tabular_classification_models(self):
        models = [
            WTFModel(LogisticRegression, normalize=True, one_hot_encode=True),
            WTFModel(LinearSVC, normalize=True, one_hot_encode=True),
            WTFModel(RandomForestClassifier, normalize=False, one_hot_encode=False),
            WTFModel(ExtraTreesClassifier, normalize=False, one_hot_encode=False),
            WTFModel(SGDClassifier, normalize=True, one_hot_encode=True),
            WTFModel(KNeighborsClassifier, normalize=True, one_hot_encode=True),
            WTFModel(DecisionTreeClassifier, normalize=False, one_hot_encode=False),
            WTFModel(GaussianNB, normalize=True, one_hot_encode=True),
            WTFModel(MultinomialNB, normalize=True, one_hot_encode=True),
            WTFModel(xgb.XGBClassifier, normalize=False, one_hot_encode=False),
        ]
        return models

    def _tabular_regression_models(self):
        models = [
            WTFModel(LinearRegression, normalize=True, one_hot_encode=True),
            WTFModel(LinearSVR, normalize=True, one_hot_encode=True),
            WTFModel(Ridge, normalize=True, one_hot_encode=True),
            WTFModel(Lasso, normalize=True, one_hot_encode=True),
            WTFModel(SGDRegressor, normalize=True, one_hot_encode=True),
            WTFModel(KNeighborsRegressor, normalize=True, one_hot_encode=True),
            WTFModel(DecisionTreeRegressor, normalize=False, one_hot_encode=False),
            WTFModel(RandomForestRegressor, normalize=False, one_hot_encode=False),
            WTFModel(ExtraTreesRegressor, normalize=False, one_hot_encode=False),
            WTFModel(xgb.XGBRegressor, normalize=False, one_hot_encode=False),
        ]
        return models
