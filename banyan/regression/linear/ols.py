from collections.abc import Iterable
from typing import Union

from numpy import ndarray, dot
from numpy.linalg import pinv
from pandas import DataFrame, Series

from ...decorators import InputTypeAdapter


class OLS:
    def __init__(self) -> None:
        pass
    
    @InputTypeAdapter
    def fit(self, X: Union[Iterable, ndarray, DataFrame], y: Union[Iterable, ndarray, Series]) -> None:
        self.b = pinv(X) @ y
        self.y_avg = y.mean()
        self.r2 = 1.0 - ((y - X @ self.b) ** 2).sum() / ((y - self.y_avg) ** 2).sum()

    @InputTypeAdapter
    def predict(self, X: Union[Iterable, ndarray, DataFrame]) -> ndarray:        
        y = X @ self.b
        return y