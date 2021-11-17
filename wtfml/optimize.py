import os

import numpy as np
import pandas as pd
from metrics import Metrics
from xgboost.sklearn import XGBClassifier, XGBRegressor

from .enums import ProblemType
from .logger import logger
from .params import Params
from .utils import dict_mean


def optimize(
    trial,
    model,
    use_predict_proba,
    eval_metric,
    problem_type,
    num_folds,
    features,
    targets,
    output,
    fast,
    seed,
):

    one_hot_encode = model.one_hot_encode
    normalize = model.normalize
    mx = model.x

    params = Params(model=mx, problem_type=problem_type)
    training_params = params.fetch_params()

    if isinstance(mx, (XGBClassifier, XGBRegressor)):
        early_stopping_rounds = training_params["early_stopping_rounds"]
        del training_params["early_stopping_rounds"]

    metrics = Metrics(problem_type)

    scores = []

    for fold in range(num_folds):
        train_feather = pd.read_feather(os.path.join(output, f"train_fold_{fold}.feather"))
        valid_feather = pd.read_feather(os.path.join(output, f"valid_fold_{fold}.feather"))
        xtrain = train_feather[features]
        xvalid = valid_feather[features]

        ytrain = train_feather[targets].values
        yvalid = valid_feather[targets].values

        if isinstance(mx, (XGBClassifier, XGBRegressor)):
            mx = mx(
                random_state=seed,
                eval_metric=eval_metric,
                use_label_encoder=False,
                **training_params,
            )
        else:
            mx = mx(**training_params)

        if problem_type in (ProblemType.multi_column_regression, ProblemType.multi_label_classification):
            ypred = []
            models = [mx] * len(targets)
            for idx, _m in enumerate(models):
                if isinstance(_m, (XGBClassifier, XGBRegressor)):
                    _m.fit(
                        xtrain,
                        ytrain[:, idx],
                        early_stopping_rounds=early_stopping_rounds,
                        eval_set=[(xvalid, yvalid[:, idx])],
                        verbose=False,
                    )
                else:
                    _m.fit(xtrain, ytrain[:, idx])
                if problem_type == ProblemType.multi_column_regression:
                    ypred_temp = _m.predict(xvalid)
                else:
                    ypred_temp = _m.predict_proba(xvalid)[:, 1]
                ypred.append(ypred_temp)
            ypred = np.column_stack(ypred)

        else:
            if isinstance(mx, (XGBClassifier, XGBRegressor)):
                mx.fit(
                    xtrain,
                    ytrain,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(xvalid, yvalid)],
                    verbose=False,
                )
            else:
                mx.fit(xtrain, ytrain)

            if use_predict_proba:
                ypred = mx.predict_proba(xvalid)
            else:
                ypred = mx.predict(xvalid)

        # calculate metric
        metric_dict = metrics.calculate(yvalid, ypred)
        scores.append(metric_dict)
        if fast is True:
            break

    mean_metrics = dict_mean(scores)
    logger.info(f"Metrics: {mean_metrics}")
    return mean_metrics[eval_metric]
