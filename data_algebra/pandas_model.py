import types

import numpy

import data_algebra
import data_algebra.util
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components


class PandasModel(data_algebra.data_model.DataModel):
    def __init__(self, *, pd=None):
        data_algebra.data_model.DataModel.__init__(self)
        if pd is None:
            pd = data_algebra.pd
        if not isinstance(pd, types.ModuleType):
            raise TypeError("Expected pd to be a module")
        self.pd = pd
        # alter here, expr_rep pd_formatters, expr_rep @-defs, and env populate_specidls in parallel
        # to extend functionality
        self.pandas_eval_env = {
            "is_null": lambda x: self.pd.isnull(x),
            "is_bad": lambda x: data_algebra.util.is_bad(x, pd=self.pd),
            "if_else": lambda c, x, y: numpy.where(c, x, y),
            "co_equalizer": lambda f, g: data_algebra.connected_components.connected_components(
                f, g
            ),
            "connected_components": lambda f, g: data_algebra.connected_components.connected_components(
                f, g
            ),
            "partitioned_eval": lambda fn, arg_columns, partition_columns: (
                data_algebra.connected_components.partitioned_eval(
                    fn, arg_columns, partition_columns
                )
            ),
            "max": lambda x: [numpy.max(x)] * len(x),
            "min": lambda x: [numpy.min(x)] * len(x),
        }

    def assert_is_appropriate_data_instance(self, df, arg_name=""):
        # noinspection PyUnresolvedReferences
        if not isinstance(df, self.pd.DataFrame):
            raise TypeError(arg_name + " was supposed to be a self.pd.DataFrame")

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def table_step(self, op, *, data_map, eval_env):
        if op.node_name != "TableDescription":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.TableDescription"
            )
        if len(op.qualifiers) > 0:
            raise ValueError(
                "table descriptions used with eval_implementation() must not have qualifiers"
            )
        df = data_map[op.table_name]
        self.assert_is_appropriate_data_instance(df, "data_map[" + op.table_name + "]")
        # check all columns we expect are present
        columns_using = op.column_names
        if op.columns_currently_used is not None and len(op.columns_currently_used) > 0:
            columns_using = [c for c in columns_using if c in op.columns_currently_used]
        missing = set(columns_using) - set([c for c in df.columns])
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        # make an index-free copy of the data to isolate side-effects and not deal with indices
        res = df.loc[:, columns_using]
        res = res.reset_index(drop=True)
        return res

    def extend_step(self, op, *, data_map, eval_env):
        if op.node_name != "ExtendNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ExtendNode")
        window_situation = (
            op.windowed_situation
            or (len(op.partition_by) > 0)
            or (len(op.order_by) > 0)
        )
        if window_situation:
            op.check_extend_window_fns()
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        standin_name = "_data_algebra_temp_g"  # name of an arbitrary input variable
        if not window_situation:
            for (k, opk) in op.ops.items():
                op_src = opk.to_pandas()
                res[k] = res.eval(
                    op_src, local_dict=self.pandas_eval_env, global_dict=eval_env
                )
        else:
            for (k, opk) in op.ops.items():
                # work on a slice of the data frame
                col_list = [c for c in set(op.partition_by)]
                for c in op.order_by:
                    if c not in col_list:
                        col_list = col_list + [c]
                value_name = None
                # assumes all args are column names, enforce this earlier
                if len(opk.args) > 0:
                    value_name = opk.args[0].to_pandas()
                    if value_name not in set(col_list):
                        col_list = col_list + [value_name]
                ascending = [c not in set(op.reverse) for c in col_list]
                subframe = res[col_list].reset_index(drop=True)
                subframe["_data_algebra_orig_index"] = subframe.index
                subframe = subframe.sort_values(
                    by=col_list, ascending=ascending
                ).reset_index(drop=True)
                subframe[standin_name] = 1
                if len(op.partition_by) > 0:
                    opframe = subframe.groupby(op.partition_by)
                    #  Groupby preserves the order of rows within each group.
                    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
                else:
                    opframe = subframe.groupby([standin_name])
                # TODO: document exactly which of these are available
                if len(opk.args) == 0:
                    if opk.op == "row_number":
                        subframe[k] = opframe.cumcount() + 1
                    elif opk.op == "ngroup":
                        subframe[k] = opframe.ngroup()
                    elif opk.op == "size":
                        subframe[k] = opframe[standin_name].transform(
                            opk.op
                        )  # Pandas transform, not data_algegra
                    else:
                        raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                else:
                    # len(opk.args) == 1
                    subframe[k] = opframe[value_name].transform(
                        opk.op
                    )  # Pandas transform, not data_algegra
                subframe = subframe.reset_index(drop=True)
                subframe = subframe.sort_values(by=["_data_algebra_orig_index"])
                subframe = subframe.reset_index(drop=True)
                res[k] = subframe[k]
        return res

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        # noinspection PyUnresolvedReferences
        return self.pd.DataFrame(cols)

    def project_step(self, op, *, data_map, eval_env):
        if op.node_name != "ProjectNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        # check these are forms we are prepared to work with, and build an aggregation dictionary
        # build an agg list: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        # https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with
        # try the following tutorial:
        # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        for (k, opk) in op.ops.items():
            if len(opk.args) > 1:
                raise ValueError(
                    "non-trivial aggregation expression: " + str(k) + ": " + str(opk)
                )
            if len(opk.args) > 0:
                if not isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                    raise ValueError(
                        "windows expression argument must be a column: "
                        + str(k)
                        + ": "
                        + str(opk)
                    )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        res["_data_table_temp_col"] = 1
        if len(op.group_by) > 0:
            res = res.groupby(op.group_by)
        if len(op.ops) > 0:
            cols = {
                k: (
                    res[str(opk.args[0])].agg(opk.op)
                    if len(opk.args) > 0
                    else res["_data_table_temp_col"].agg(opk.op)
                )
                for (k, opk) in op.ops.items()
            }
        else:
            cols = {"_data_table_temp_col": res["_data_table_temp_col"].agg("sum")}

        # agg can return scalars, which then can't be made into a self.pd.DataFrame
        def promote_scalar(v):
            # noinspection PyBroadException
            try:
                len(v)
            except Exception:
                return [v]
            return v

        cols = {k: promote_scalar(v) for (k, v) in cols.items()}
        res = self.columns_to_frame(cols).reset_index(
            drop=len(op.group_by) < 1
        )  # grouping variables in the index
        if "_data_table_temp_col" in res.columns:
            res = res.drop("_data_table_temp_col", 1)
        # double check shape is what we expect
        if not data_algebra.util.table_is_keyed_by_columns(res, op.group_by):
            raise ValueError("result wasn't keyed by group_by columns")
        return res

    def select_rows_step(self, op, *, data_map, eval_env):
        if op.node_name != "SelectRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectRowsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        q = op.expr.to_pandas()
        res = res.query(q).reset_index(drop=True)
        return res

    def select_columns_step(self, op, *, data_map, eval_env):
        if op.node_name != "SelectColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        return res[op.column_selection]

    def drop_columns_step(self, op, *, data_map, eval_env):
        if op.node_name != "DropColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.DropColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        column_selection = [c for c in res.columns if c not in op.column_deletions]
        return res[column_selection]

    def order_rows_step(self, op, *, data_map, eval_env):
        if op.node_name != "OrderRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        ascending = [
            False if ci in set(op.reverse) else True for ci in op.order_columns
        ]
        res = res.sort_values(by=op.order_columns, ascending=ascending).reset_index(
            drop=True
        )
        return res

    def rename_columns_step(self, op, *, data_map, eval_env):
        if op.node_name != "RenameColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        return res.rename(columns=op.reverse_mapping)

    # noinspection PyMethodMayBeStatic
    def standardize_join_code(self, jointype):
        if not isinstance(jointype, str):
            raise TypeError("expected jointype to be a string")
        jointype = jointype.lower()
        mp = {"full": "outer"}
        try:
            return mp[jointype]
        except KeyError:
            pass
        return jointype

    def natural_join_step(self, op, *, data_map, eval_env):
        if op.node_name != "NaturalJoinNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.NaturalJoinNode"
            )
        left = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        right = op.sources[1].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        common_cols = set([c for c in left.columns]).intersection(
            [c for c in right.columns]
        )
        # noinspection PyUnresolvedReferences
        res = self.pd.merge(
            left=left,
            right=right,
            how=self.standardize_join_code(op.jointype),
            on=op.by,
            sort=False,
            suffixes=("", "_tmp_right_col"),
        )
        res = res.reset_index(drop=True)
        for c in common_cols:
            if c not in op.by:
                is_null = res[c].isnull()
                res.loc[is_null, c] = res.loc[is_null, c + "_tmp_right_col"]
                res = res.drop(c + "_tmp_right_col", axis=1)
        res = res.reset_index(drop=True)
        return res

    def concat_rows_step(self, op, *, data_map, eval_env):
        if op.node_name != "ConcatRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConcatRowsNode"
            )
        left = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        right = op.sources[1].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        if op.id_column is not None:
            left[op.id_column] = op.a_name
            right[op.id_column] = op.b_name
        # noinspection PyUnresolvedReferences
        res = self.pd.concat([left, right], axis=0, ignore_index=True)
        res = res.reset_index(drop=True)
        return res

    def convert_records_step(self, op, *, data_map, eval_env):
        if op.node_name != "ConvertRecordsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConvertRecordsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=self
        )
        return op.record_map.transform(res)
