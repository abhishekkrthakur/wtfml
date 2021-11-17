"""
__author__: Abhishek Thakur
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn import model_selection

from .enums import ProblemType
from .logger import logger


@dataclass
class FoldGenerator:
    train_df: pd.DataFrame
    targets: List[str]
    problem_type: ProblemType
    num_folds: int = 5
    shuffle: bool = True
    seed: int = 42

    def _create_folds(self, train_df, problem_type):
        if "kfold" in train_df.columns:
            self.num_folds = len(np.unique(train_df["kfold"]))
            logger.info("Using `kfold` for folds from training data")
            return train_df

        logger.info("Creating folds")
        train_df["kfold"] = -1
        if problem_type in (ProblemType.binary_classification, ProblemType.multi_class_classification):
            y = train_df[self.targets].values
            kf = model_selection.StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed,
            )
            for fold, (_, valid_indicies) in enumerate(kf.split(X=train_df, y=y)):
                train_df.loc[valid_indicies, "kfold"] = fold

        elif problem_type == ProblemType.single_column_regression:
            y = train_df[self.targets].values
            num_bins = int(np.floor(1 + np.log2(len(train_df))))
            if num_bins > 10:
                num_bins = 10
            kf = model_selection.StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed,
            )
            train_df["bins"] = pd.cut(
                train_df[self.targets].values.ravel(),
                bins=num_bins,
                labels=False,
            )
            for fold, (_, valid_indicies) in enumerate(kf.split(X=train_df, y=train_df.bins.values)):
                train_df.loc[valid_indicies, "kfold"] = fold
            train_df = train_df.drop("bins", axis=1)

        elif problem_type == ProblemType.multi_column_regression:
            y = train_df[self.targets].values
            kf = model_selection.KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed,
            )
            for fold, (_, valid_indicies) in enumerate(kf.split(X=train_df, y=y)):
                train_df.loc[valid_indicies, "kfold"] = fold
        # TODO: use iterstrat
        elif problem_type == ProblemType.multi_label_classification:
            y = train_df[self.targets].values
            kf = model_selection.KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed,
            )
            for fold, (_, valid_indicies) in enumerate(kf.split(X=train_df, y=y)):
                train_df.loc[valid_indicies, "kfold"] = fold
        else:
            raise Exception("Problem type not supported")
        return train_df

    def generate(self):
        train_df = self._create_folds(self.train_df, self.problem_type)
        return train_df

    def get_fold(self, fold):
        return self.train_df[self.train_df["kfold"] == fold]
