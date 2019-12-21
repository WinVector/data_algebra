class DataModel:
    def __init__(self):
        pass

    # helper functions

    def assert_is_appropriate_data_instance(self, df, arg_name=""):
        raise NotImplementedError("base method called")

    # operation implementations

    def table_step(self, op, *, data_map, eval_env):
        """
        Represents a data input.

        :param op:
        :param data_map:
        :param eval_env:
        :return:
        """
        raise NotImplementedError("base method called")

    def extend_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        raise NotImplementedError("base method called")

    def project_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def select_rows_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def select_columns_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def drop_columns_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def order_rows_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def rename_columns_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def natural_join_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def concat_rows_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    def convert_records_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base class called")
