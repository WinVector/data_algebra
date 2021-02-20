from abc import ABC


class DataModel(ABC):
    def __init__(self, presentation_model_name):
        self.presentation_model_name = presentation_model_name

    # helper functions

    def data_frame(self, arg=None):
        raise NotImplementedError("base method called")

    def is_appropriate_data_instance(self, df):
        raise NotImplementedError("base method called")

    def can_convert_col_to_numeric(self, x):
        """check if non-empty vector can convert to numeric"""
        raise NotImplementedError("base method called")

    def to_numeric(self, x, *, errors="coerce"):
        raise NotImplementedError("base method called")

    def isnull(self, x):
        raise NotImplementedError("base method called")

    def bad_column_positions(self, x):
        """ for numeric vector x, return logical vector of positions that are null, NaN, infinite"""
        raise NotImplementedError("base method called")

    # operation implementations

    def table_step(self, op, *, data_map, eval_env, narrow):
        """
        Represents a data input.

        :param op:
        :param data_map:
        :param eval_env:
        :param narrow:
        :return:
        """
        raise NotImplementedError("base method called")

    def extend_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        raise NotImplementedError("base method called")

    def project_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def select_rows_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def select_columns_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def drop_columns_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def order_rows_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def rename_columns_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def natural_join_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def concat_rows_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base method called")

    def convert_records_step(self, op, *, data_map, eval_env, narrow):
        raise NotImplementedError("base class called")
