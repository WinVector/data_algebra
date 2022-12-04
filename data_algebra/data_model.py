"""
Interface for realizing the data algebra as a sequence of steps over an object.
"""


import abc
from typing import Any, Dict, List


# map type name strings to data models
data_model_type_map = dict()


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
    def eval(self, op, *, data_map: Dict[str, Any], narrow: bool = False):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :param narrow: if True narrow results to only columns anticipated
        :return: data frame result
        """
    
    # expression helpers

    @abc.abstractmethod
    def act_on_literal(self, *, arg, value):
        """
        Action for a literal/constant in an expression.

        :param arg: item we are acting on
        :param value: literal value being supplied
        :return: arg acted on
        """
    
    @abc.abstractmethod
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: arg acted on
        """
    
    @abc.abstractmethod
    def act_on_expression(self, *, arg, values: List, op: str):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param values: list of values to work on
        :param op: name of operator to apply
        :return: arg acted on
        """
