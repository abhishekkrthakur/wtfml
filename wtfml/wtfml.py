import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.utils.multiclass import type_of_target

from .enums import DataType, ProblemType
from .fold_generator import FoldGenerator
from .logger import logger
from .models import WTFModels
from .optimize import optimize
from .schema import WTFArgs
from .utils import get_problem_helpers, reduce_memory_usage


@dataclass
class WTFML:
    args: WTFArgs

    def __post_init__(self):
        self.task = self.args.task
        self.targets = self.args.targets
        self.features = self.args.features
        self.categorical_features = self.args.categorical_features
        self.drop_cols = self.args.drop_cols
        self.train = self.args.train
        self.test = self.args.test
        self.idx = self.args.idx
        self.data_type = self.args.data_type
        self.seed = self.args.seed
        self.num_folds = self.args.num_folds
        self.output = self.args.output
        self.use_gpu = self.args.use_gpu
        self.ignore_columns = [self.idx, "kfold"] + self.targets

        self.test_df = None
        self.problem_type = None

        if self.data_type != DataType.tabular:
            raise Exception("Only tabular data is supported at the moment")

        if isinstance(self.train, str):
            self.train_df = pd.read_csv(self.train)
        else:
            self.train_df = self.train

        self.train_df = reduce_memory_usage(self.train_df)

        if self.test is not None:
            if isinstance(self.test, str):
                self.test_df = pd.read_csv(self.test)
            else:
                self.test_df = self.test
            self.test_df = reduce_memory_usage(self.test_df)

        self._process_tabular_data()

    def _inject_idx_column(self, df):
        if self.idx not in df.columns:
            df[self.idx] = np.arange(len(df))
        return df

    def _determine_problem_type(self):
        if self.task is not None:
            if self.task == "classification":
                if len(self.targets) == 1:
                    if len(np.unique(self.train_df[self.targets].values)) == 2:
                        problem_type = ProblemType.binary_classification
                    else:
                        problem_type = ProblemType.multi_class_classification
                else:
                    problem_type = ProblemType.multi_label_classification

            elif self.task == "regression":
                if len(self.targets) == 1:
                    problem_type = ProblemType.single_column_regression
                else:
                    problem_type = ProblemType.multi_column_regression
            else:
                raise Exception("Problem type not understood")

        else:
            target_type = type_of_target(self.train_df[self.targets].values)
            # target type is one of the following using scikit-learn's type_of_target
            # * 'continuous': `y` is an array-like of floats that are not all
            #   integers, and is 1d or a column vector.
            # * 'continuous-multioutput': `y` is a 2d array of floats that are
            #   not all integers, and both dimensions are of size > 1.
            # * 'binary': `y` contains <= 2 discrete values and is 1d or a column
            #   vector.
            # * 'multiclass': `y` contains more than two discrete values, is not a
            #   sequence of sequences, and is 1d or a column vector.
            # * 'multiclass-multioutput': `y` is a 2d array that contains more
            #   than two discrete values, is not a sequence of sequences, and both
            #   dimensions are of size > 1.
            # * 'multilabel-indicator': `y` is a label indicator matrix, an array
            #   of two dimensions with at least two columns, and at most 2 unique
            #   values.
            # * 'unknown': `y` is array-like but none of the above, such as a 3d
            #   array, sequence of sequences, or an array of non-sequence objects.
            if target_type == "continuous":
                problem_type = ProblemType.single_column_regression
            elif target_type == "continuous-multioutput":
                problem_type = ProblemType.multi_column_regression
            elif target_type == "binary":
                problem_type = ProblemType.binary_classification
            elif target_type == "multiclass":
                problem_type = ProblemType.multi_class_classification
            elif target_type == "multilabel-indicator":
                problem_type = ProblemType.multi_label_classification
            else:
                raise Exception("Unable to infer `problem_type`. Please provide `classification` or `regression`")
        logger.info(f"Problem type: {problem_type.name}")
        return problem_type

    def _process_tabular_data(self):
        logger.info("Reading training data")
        self.problem_type = self._determine_problem_type()
        self.train_df = self._inject_idx_column(self.train_df)

        if self.test_df is not None:
            self.test_df = self._inject_idx_column(self.test_df)

        fg = FoldGenerator(
            self.train_df,
            self.targets,
            self.features,
            self.categorical_features,
            self.drop_cols,
            self.seed,
            self.num_folds,
        )
        self.train_df = fg.generate()

        if self.features is None:
            self.features = list(self.train_df.columns)
            self.features = [x for x in self.features if x not in self.ignore_columns]

        # encode target(s)
        if self.problem_type in [ProblemType.binary_classification, ProblemType.multi_class_classification]:
            logger.info("Encoding target(s)")
            target_encoder = LabelEncoder()
            target_encoder.fit(self.train_df[self.targets].values.reshape(-1))
            self.train_df.loc[:, self.targets] = target_encoder.transform(
                self.train_df[self.targets].values.reshape(-1)
            )
        else:
            target_encoder = None

        if self.categorical_features is None:
            # find categorical features
            categorical_features = []
            for col in self.features:
                if self.train_df[col].dtype == "object":
                    categorical_features.append(col)
        else:
            categorical_features = self.categorical_features

        logger.info(f"Found {len(categorical_features)} categorical features.")

        if len(categorical_features) > 0:
            logger.info("Encoding categorical features")

        categorical_encoders = {}
        one_hot_encoders = {}
        for fold in range(self.num_folds):
            fold_train = self.train_df[self.train_df.kfold != fold].reset_index(drop=True)
            fold_valid = self.train_df[self.train_df.kfold == fold].reset_index(drop=True)
            if self.test_df is not None:
                test_fold = self.test_df.copy(deep=True)
            if len(categorical_features) > 0:
                ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                oh_encoder = OneHotEncoder(handle_unknown="ignore")
                fold_train[categorical_features] = ord_encoder.fit_transform(fold_train[categorical_features].values)
                fold_valid[categorical_features] = ord_encoder.transform(fold_valid[categorical_features].values)
                oh_encoder.fit(fold_train[categorical_features].values)
                if self.test_df is not None:
                    test_fold[categorical_features] = ord_encoder.transform(test_fold[categorical_features].values)
                categorical_encoders[fold] = ord_encoder
                one_hot_encoders[fold] = oh_encoder
            fold_train.to_feather(os.path.join(self.output, f"train_fold_{fold}.feather"))
            fold_valid.to_feather(os.path.join(self.output, f"valid_fold_{fold}.feather"))
            if self.test_df is not None:
                test_fold.to_feather(os.path.join(self.output, f"test_fold_{fold}.feather"))

        logger.info("Saving encoders")
        joblib.dump(categorical_encoders, f"{self.output}/wtfml.categorical_encoders")
        joblib.dump(target_encoder, f"{self.output}/wtfml.target_encoder")
        joblib.dump(one_hot_encoders, f"{self.output}/wtfml.one_hot_encoders")

    def run(
        self,
        trials_per_model: Optional[int] = 10,
        time_limit_per_model: Optional[int] = None,
        num_models: Optional[int] = 10,
    ):
        wtm = WTFModels(data_type=self.data_type, task=self.task)
        models = wtm.get_models()
        normalizers = wtm.get_normalizers()
        use_predict_proba, eval_metric, direction = get_problem_helpers(problem_type=self.problem_type)
        for model in models:
            optimize_func = partial(
                optimize,
                model,
                use_predict_proba,
                eval_metric,
                problem_type=self.problem_type,
                num_folds=self.num_folds,
                features=self.features,
                targets=self.targets,
                output=self.output,
                fast=True,
                seed=self.seed,
            )
            # db_path = os.path.join(model_config.output, "params.db")
            study = optuna.create_study(
                direction=direction,
                study_name="wtfml",
                # storage=f"sqlite:///{db_path}",
                # load_if_exists=True,
            )
            study.optimize(optimize_func, n_trials=trials_per_model, timeout=time_limit_per_model)
