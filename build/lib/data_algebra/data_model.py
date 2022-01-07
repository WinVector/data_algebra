"""
Interface for realizing the data algebra as a sequence of steps over an object.
"""


from abc import ABC
from typing import Dict, Optional


class DataModel(ABC):
    """
    Interface for realizing the data algebra as a sequence of steps over a Pandas like object.
    """

    presentation_model_name: str

    def __init__(self, presentation_model_name: str):
        assert isinstance(presentation_model_name, str)
        self.presentation_model_name = presentation_model_name

    # helper functions

    def data_frame(self, arg=None):
        """
        Build a new emtpy data frame.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """
        raise NotImplementedError("base method called")

    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """
        raise NotImplementedError("base method called")

    # evaluate

    def eval(self, op, *, data_map: Optional[Dict] = None, narrow: bool = False):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :param narrow: if True narrow results to only columns anticipated
        :return: data frame result
        """
        raise NotImplementedError("base method called")
