"""
Interface for realizing the data algebra as a sequence of steps over an object.
"""


from abc import ABC
from typing import Callable, Dict


class DataModel(ABC):
    """
    Interface for realizing the data algebra as a sequence of steps over a Pandas like object.
    """

    presentation_model_name: str
    user_fun_map: Dict[str, Callable]

    def __init__(self, presentation_model_name: str):
        assert isinstance(presentation_model_name, str)
        self.presentation_model_name = presentation_model_name
        self.user_fun_map = dict()

    # helper functions

    def data_frame(self, arg=None):
        """
        Build a new emtpy data frame.

        :param arg" optional argument passed to constructor.
        :return: data frame
        """
        raise NotImplementedError("base method called")

    def is_appropriate_data_instance(self, df):
        """
        Check if df is our type of data frame.
        """
        raise NotImplementedError("base method called")

    def can_convert_col_to_numeric(self, x):
        """
        Return True if column or value can be converted to numeric type.
        """
        raise NotImplementedError("base method called")

    def to_numeric(self, x, *, errors="coerce"):
        """
        Convert column to numeric.
        """
        raise NotImplementedError("base method called")

    def isnull(self, x):
        """
        Return vector indicating which entries are null (vectorized).
        """
        raise NotImplementedError("base method called")

    def bad_column_positions(self, x):
        """
        Return vector indicating which entries are bad (null or nan) (vectorized).
        """
        raise NotImplementedError("base method called")

    # operation implementations

    def table_step(self, op, *, data_map, narrow):
        """
        Return data frame from table description and data_map.

        :param op: operation
        :param data_map: map from tables to values
        :param narrow: optional columns to narrow to.
        :return: transformed table
        """
        raise NotImplementedError("base method called")

    def extend_step(self, op, *, data_map, narrow):
        """
        Execute an extend step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def project_step(self, op, *, data_map, narrow):
        """
        Execute a project step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def select_rows_step(self, op, *, data_map, narrow):
        """
        Execute a select rows step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def select_columns_step(self, op, *, data_map, narrow):
        """
        Execute a select columns step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def drop_columns_step(self, op, *, data_map, narrow):
        """
        Execute a drop columns step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def order_rows_step(self, op, *, data_map, narrow):
        """
        Execute an order rows step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def rename_columns_step(self, op, *, data_map, narrow):
        """
        Execute a rename columns step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def natural_join_step(self, op, *, data_map, narrow):
        """
        Execute a natural join step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def concat_rows_step(self, op, *, data_map, narrow):
        """
        Execute a concat rows step, returning a data frame.
        """
        raise NotImplementedError("base method called")

    def convert_records_step(self, op, *, data_map, narrow):
        """
        Execute record conversion step, returning a data frame.
        """
        raise NotImplementedError("base class called")
