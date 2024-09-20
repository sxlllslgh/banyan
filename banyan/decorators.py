from collections.abc import Iterable
from functools import wraps
from inspect import getfullargspec
from types import MethodType

from numpy import array
from pandas import DataFrame, Series


class InputTypeAdapter:
    def __init__(self, func) -> None:
        wraps(func)(self)
    
    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, arg_name in enumerate(getfullargspec(self.__wrapped__)[0]):
            if arg_name == 'X':
                if isinstance(args[i], Iterable):
                    args[i] = array(args[i])
                elif isinstance(args[i], DataFrame):
                    args[i] = args[i].values

            if arg_name == 'y':
                if isinstance(args[i], Iterable):
                    args[i] = array(args[i])
                elif isinstance(args[i], Series):
                    args[i] = args[i].values
        return self.__wrapped__(*args, **kwargs)
    
    def __get__(self, instance, _):
        if instance is None:
            return self
        else:
            return MethodType(self, instance)