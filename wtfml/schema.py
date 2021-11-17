from typing import List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from .enums import DataType, ProblemType, TaskType


class WTFConfig(BaseModel):
    train_filename: str
    test_filename: Optional[str] = None
    idx: str
    targets: List[str]
    problem_type: ProblemType
    output: str
    features: List[str]
    num_folds: int
    use_gpu: bool
    seed: int
    categorical_features: List[str]


class WTFArgs(BaseModel):
    train: Union[pd.DataFrame, str]
    # TODO: add support for custom validation data
    # valid: Union[pd.DataFrame, str]
    test: Union[pd.DataFrame, str]
    task: TaskType

    output: str

    use_gpu: Optional[bool] = False

    features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None

    targets: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None

    idx: Optional[str] = None
    data_type: Optional[DataType] = DataType.tabular
    seed: Optional[int] = 42
    num_folds: Optional[int] = 5
