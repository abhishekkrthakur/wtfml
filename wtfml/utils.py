import numpy as np
import optuna

from .enums import ProblemType
from .logger import logger


optuna.logging.set_verbosity(optuna.logging.INFO)


def reduce_memory_usage(df, verbose=True):
    # NOTE: Original author of this function is unknown
    # if you know the *original author*, please let me know.
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def get_problem_helpers(problem_type: ProblemType):
    if problem_type == ProblemType.binary_classification:
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "logloss"
    elif problem_type == ProblemType.multi_class_classification:
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "mlogloss"
    elif problem_type == ProblemType.multi_label_classification:
        use_predict_proba = True
        direction = "minimize"
        eval_metric = "logloss"
    elif problem_type == ProblemType.single_column_regression:
        use_predict_proba = False
        direction = "minimize"
        eval_metric = "rmse"
    elif problem_type == ProblemType.multi_column_regression:
        use_predict_proba = False
        direction = "minimize"
        eval_metric = "rmse"
    else:
        raise NotImplementedError

    return use_predict_proba, eval_metric, direction
