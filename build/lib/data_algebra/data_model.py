import data_algebra
import data_algebra.expr_rep
import data_algebra.data_ops


class DataModel:
    def __init__(self):
        pass

    def assert_is_appropriate_data_instance(self, df, arg_name=""):
        raise NotImplementedError("base method called")

    def table_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base method called")

    # noinspection PyMethodMayBeStatic
    def check_extend_window_fns(self, op):
        if not isinstance(op, data_algebra.data_ops.ExtendNode):
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        window_situation = (len(op.partition_by) > 0) or (len(op.order_by) > 0)
        if window_situation:
            # check these are forms we are prepared to work with
            for (k, opk) in op.ops.items():
                if len(opk.args) > 1:
                    raise ValueError(
                        "non-trivial windows expression: " + str(k) + ": " + str(opk)
                    )
                if len(opk.args) == 1:
                    if not isinstance(
                        opk.args[0], data_algebra.expr_rep.ColumnReference
                    ):
                        raise ValueError(
                            "windows expression argument must be a column: "
                            + str(k)
                            + ": "
                            + str(opk)
                        )

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

    def convert_records_step(self, op, *, data_map, eval_env):
        raise NotImplementedError("base class called")
