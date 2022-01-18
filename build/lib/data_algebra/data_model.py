"""
Interface for realizing the data algebra as a sequence of steps over an object.
"""


import abc
from typing import Dict, Optional


class DataModel(abc.ABC):
    """
    Interface for realizing the data algebra as a sequence of steps over a Pandas like object.
    """

    presentation_model_name: str

    def __init__(self, presentation_model_name: str):
        assert isinstance(presentation_model_name, str)
        self.presentation_model_name = presentation_model_name

    # helper functions

    @abc.abstractmethod
    def data_frame(self, arg=None):
        """
        Build a new emtpy data frame.

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
    def eval(self, op, *, data_map: Optional[Dict] = None, narrow: bool = False):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :param narrow: if True narrow results to only columns anticipated
        :return: data frame result
        """
