"""
Interface for realizing the data algebra as a sequence of steps over an object.
"""


import abc
import re
from typing import Any, Dict, List


class DataModel(abc.ABC):
    """
    Interface for realizing the data algebra as a sequence of steps over Pandas like objects.
    """

    presentation_model_name: str

    def __init__(self, presentation_model_name: str):
        assert isinstance(presentation_model_name, str)
        self.presentation_model_name = presentation_model_name

    # data frame helpers

    @abc.abstractmethod
    def data_frame(self, arg=None):
        """
        Build a new data frame.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """

    @abc.abstractmethod
    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """

    # evaluate

    @abc.abstractmethod
    def eval(self, op, *, data_map: Dict[str, Any]):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :return: data frame result
        """
    
    # expression helpers

    @abc.abstractmethod
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
    
    @abc.abstractmethod
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: converted result
        """
    
    @abc.abstractmethod
    def act_on_expression(self, *, arg, values: List, op: str):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param values: op arguments already converted
        :param op: operator to apply
        :return: converted result
        """


# map type name strings to data models
data_model_type_map = dict()

def default_data_model() -> DataModel:
    """Get the default (Pandas) data model"""
    return data_model_type_map["default_data_model"]

polars_regexp = re.compile(r".*[^a-zA-Z]polars[^a-zA-Z].*[^a-zA-z]dataframe[^a-zA-Z].*")
assert polars_regexp.match("<class 'polars.internals.dataframe.frame.DataFrame'>") is not None

def lookup_data_model_for_key(key: str) -> DataModel:
    assert isinstance(key, str)
    if polars_regexp.match(key.lower()) is not None:
        import data_algebra.polars_model
        data_algebra.polars_model.register_polars_model(key)
    return data_model_type_map[key]

def lookup_data_model_for_dataframe(d) -> DataModel:
    key = str(type(d))
    res = lookup_data_model_for_key(key)
    assert res.is_appropriate_data_instance(d)
    return res
