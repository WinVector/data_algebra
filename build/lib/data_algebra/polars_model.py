
"""
Adapter to use Polars ( https://www.pola.rs ) in the data algebra.

Note: not implemented yet.
"""

from typing import Any, Callable, Dict, List, Optional
import datetime
import types
import numbers
import warnings

import numpy

import data_algebra.util
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components

import polars as pl


class PolarsModel(data_algebra.data_model.DataModel):
    """
    Interface for realizing the data algebra as a sequence of steps over Polars https://www.pola.rs .
    """

    presentation_model_name: str

    def __init__(self):
        data_algebra.data_model.DataModel.__init__(
            self, presentation_model_name="Polars"
        )

    def data_frame(self, arg=None):
        """
        Build a new data frame.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """
        if arg is None:
            return pl.DataFrame()
        return pl.DataFrame(arg)

    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """
        return isinstance(d, pl.DataFrame)

    # evaluate

    def eval(self, op, *, data_map: Optional[Dict], narrow: bool = False):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :param narrow: if True narrow results to only columns anticipated
        :return: data frame result
        """
        assert isinstance(data_map, Dict)
        assert isinstance(narrow, bool)
        assert isinstance(op, data_algebra.data_ops_types.OperatorPlatform)
        raise ValueError("not implemented, yet")  # TODO: implement
