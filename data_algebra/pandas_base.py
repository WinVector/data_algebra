from abc import ABC
import types
import numbers

import numpy

import data_algebra.util
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops_types
import data_algebra.connected_components


# TODO: possibly import dask, Nvidia Rapids, or modin instead

# base class for Pandas-like API realization
class PandasModelBase(data_algebra.data_model.DataModel, ABC):
    def __init__(self, *, pd, presentation_model_name):
        data_algebra.data_model.DataModel.__init__(
            self, presentation_model_name=presentation_model_name
        )
        if not isinstance(pd, types.ModuleType):
            raise TypeError("Expected pd to be a module")
        self.pd = pd

    # utils

    def data_frame(self, arg=None):
        if arg is None:
            # noinspection PyUnresolvedReferences
            return self.pd.DataFrame()
        # noinspection PyUnresolvedReferences
        return self.pd.DataFrame(arg)

    def is_appropriate_data_instance(self, df):
        # noinspection PyUnresolvedReferences
        return isinstance(df, self.pd.DataFrame)

    def can_convert_col_to_numeric(self, x):
        if isinstance(x, numbers.Number):
            return True
        # noinspection PyUnresolvedReferences
        return self.pd.api.types.is_numeric_dtype(x)

    def to_numeric(self, x, *, errors="coerce"):
        # noinspection PyUnresolvedReferences
        return self.pd.to_numeric(x, errors="coerce")

    def isnull(self, x):
        return self.pd.isnull(x)

    def bad_column_positions(self, x):
        if self.can_convert_col_to_numeric(x):
            x = numpy.asarray(x + 0, dtype=float)
            return numpy.logical_or(
                self.pd.isnull(x), numpy.logical_or(numpy.isnan(x), numpy.isinf(x))
            )
        return self.pd.isnull(x)

    # bigger stuff

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def table_step(self, op, *, data_map, narrow):
        if op.node_name != "TableDescription":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.TableDescription"
            )
        df = data_map[op.table_name]
        if not self.is_appropriate_data_instance(df):
            raise ValueError("data_map[" + op.table_name + "] was not the right type")
        # check all columns we expect are present
        columns_using = op.column_names
        missing = set(columns_using) - set([c for c in df.columns])
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        # make an index-free copy of the data to isolate side-effects and not deal with indices
        if not narrow:
            columns_using = [c for c in df.columns]
        res = df.loc[:, columns_using]
        res = res.reset_index(drop=True)
        return res

    def extend_step(self, op, *, data_map, narrow):
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
            data_map=data_map, data_model=self, narrow=narrow
        )
        data_algebra_temp_cols = {}
        if not window_situation:
            for (k, opk) in op.ops.items():
                res_k = opk.evaluate(res)  # makes debugging easier
                res[k] = res_k
        else:
            standin_name = "_data_algebra_temp_g"  # name of an arbitrary input variable
            # build up a sub-frame to work on
            col_list = [c for c in set(op.partition_by)]
            col_set = set(col_list)
            for c in op.order_by:
                if c not in col_set:
                    col_list = col_list + [c]
                    col_set.add(c)
            order_cols = [c for c in col_list]  # must be partition by followed by order

            for (k, opk) in op.ops.items():
                # assumes all args are column names or values, enforce this earlier
                if (len(opk.args) > 0):
                    assert len(opk.args) == 1
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in col_set:
                            col_list = col_list + [value_name]
                            col_set.add(value_name)
                    elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        key = str(opk.args[0].value)
                        if not key in data_algebra_temp_cols.keys():
                            value_name = 'data_algebra_extend_temp_col_' + str(len(data_algebra_temp_cols))
                            data_algebra_temp_cols[key] = value_name
                            col_list.append(value_name)
                            res[value_name] = opk.args[0].value
                    else:
                        raise ValueError("opk must be a ColumnReference or Value")
            ascending = [c not in set(op.reverse) for c in col_list]
            subframe = res[col_list].reset_index(drop=True)
            subframe["_data_algebra_orig_index"] = subframe.index
            if len(order_cols) > 0:
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
            # perform calculations
            for (k, opk) in op.ops.items():
                # work on a slice of the data frame
                # TODO: document exactly which of these are available
                if len(opk.args) <= 0:
                    if opk.op in ["row_number", "count"]:
                        subframe[k] = opframe.cumcount() + 1
                    elif opk.op == "ngroup":
                        subframe[k] = opframe.ngroup()
                    elif opk.op == "size":
                        subframe[k] = opframe[standin_name].transform(
                            opk.op
                        )  # Pandas transform, not data_algebra
                    else:
                        raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                else:
                    # len(opk.args) == 1
                    assert len(opk.args) == 1
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in set(col_list):
                            col_list = col_list + [value_name]
                        subframe[k] = opframe[value_name].transform(
                            opk.op
                        )  # Pandas transform, not data_algegra
                    elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        value_name = data_algebra_temp_cols[str(opk.args[0].value)]
                        subframe[k] = opframe[value_name].transform(
                            opk.op
                        )  # Pandas transform, not data_algegra
                    else:
                        raise ValueError("opk must be a ColumnReference or Value")
            # clear some temps
            for value_name in data_algebra_temp_cols.values():
                del res[value_name]
            # copy out results
            subframe = subframe.reset_index(drop=True)
            subframe = subframe.sort_values(by=["_data_algebra_orig_index"])
            subframe = subframe.reset_index(drop=True)
            for (k, opk) in op.ops.items():
                res[k] = subframe[k]
        return res

    def columns_to_frame(self, cols):
        """

        :param cols: dictionary mapping column names to columns
        :return:
        """
        # noinspection PyUnresolvedReferences
        return self.pd.DataFrame(cols)

    def project_step(self, op, *, data_map, narrow):
        if op.node_name != "ProjectNode":
            raise TypeError("op was supposed to be a data_algebra.data_ops.ProjectNode")
        # check these are forms we are prepared to work with, and build an aggregation dictionary
        # build an agg list: https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        # https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with
        # try the following tutorial:
        # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
        data_algebra_temp_cols = {}
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        for (k, opk) in op.ops.items():
            if len(opk.args) > 1:
                raise ValueError(
                    "non-trivial aggregation expression: " + str(k) + ": " + str(opk)
                )
            if len(opk.args) > 0:
                if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                    pass
                elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                    key = str(opk.args[0].value)
                    if not key in data_algebra_temp_cols.keys():
                        value_name = 'data_algebra_project_temp_col_' + str(len(data_algebra_temp_cols))
                        data_algebra_temp_cols[key] = value_name
                        res[value_name] = opk.args[0].value
                else:
                    raise ValueError(
                        "windows expression argument must be a column or value: "
                        + str(k)
                        + ": "
                        + str(opk)
                    )
        res["_data_table_temp_col"] = 1
        if len(op.group_by) > 0:
            res = res.groupby(op.group_by)
        if len(op.ops) > 0:
            cols = {}
            for k, opk in op.ops.items():
                value_name = None
                if len(opk.args) > 0:
                    value_name = str(opk.args[0])
                    if isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        value_name = data_algebra_temp_cols[value_name]
                if len(opk.args) > 0:
                    vk = res[value_name].agg(opk.op)
                else:
                    vk = res["_data_table_temp_col"].agg(opk.op)
                cols[k] = vk
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

    def select_rows_step(self, op, *, data_map, narrow):
        if op.node_name != "SelectRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectRowsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        selection = op.expr.evaluate(res)
        res = res.loc[selection, :].reset_index(drop=True, inplace=False)
        return res

    def select_columns_step(self, op, *, data_map, narrow):
        if op.node_name != "SelectColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.SelectColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        return res[op.column_selection]

    def drop_columns_step(self, op, *, data_map, narrow):
        if op.node_name != "DropColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.DropColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        column_selection = [c for c in res.columns if c not in op.column_deletions]
        return res[column_selection]

    def order_rows_step(self, op, *, data_map, narrow):
        if op.node_name != "OrderRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.OrderRowsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        ascending = [
            False if ci in set(op.reverse) else True for ci in op.order_columns
        ]
        res = res.sort_values(by=op.order_columns, ascending=ascending).reset_index(
            drop=True
        )
        return res

    def rename_columns_step(self, op, *, data_map, narrow):
        if op.node_name != "RenameColumnsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.RenameColumnsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        return res.rename(columns=op.reverse_mapping)

    # noinspection PyMethodMayBeStatic
    def standardize_join_code(self, jointype):
        if not isinstance(jointype, str):
            raise TypeError("expected jointype to be a string")
        jointype = jointype.lower()
        mp = {
            "full": "outer",
            "cross": "outer"  # cross new to Pandas 1.2.0 December 2020
        }
        try:
            return mp[jointype]
        except KeyError:
            pass
        return jointype

    def natural_join_step(self, op, *, data_map, narrow):
        if op.node_name != "NaturalJoinNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.NaturalJoinNode"
            )
        left = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        right = op.sources[1].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        common_cols = set([c for c in left.columns]).intersection(
            [c for c in right.columns]
        )
        by = op.by
        if by is None:
            by = []
        scratch_col = None  # extra column to prevent empty-by issues
        if len(by) <= 0:
            scratch_col = 'data_algebra_temp_merge_col'
            by = [scratch_col]
            left[scratch_col] = 1
            right[scratch_col] = 1
        # noinspection PyUnresolvedReferences
        res = self.pd.merge(
            left=left,
            right=right,
            how=self.standardize_join_code(op.jointype),
            on=by,
            sort=False,
            suffixes=("", "_tmp_right_col"),
        )
        res = res.reset_index(drop=True)
        if scratch_col is not None:
            del res[scratch_col]
        for c in common_cols:
            if c not in op.by:
                is_null = res[c].isnull()
                res.loc[is_null, c] = res.loc[is_null, c + "_tmp_right_col"]
                res = res.drop(c + "_tmp_right_col", axis=1)
        res = res.reset_index(drop=True)
        return res

    def concat_rows_step(self, op, *, data_map, narrow):
        if op.node_name != "ConcatRowsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConcatRowsNode"
            )
        left = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        right = op.sources[1].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        if op.id_column is not None:
            left[op.id_column] = op.a_name
            right[op.id_column] = op.b_name
        # noinspection PyUnresolvedReferences
        res = self.pd.concat([left, right], axis=0, ignore_index=True)
        res = res.reset_index(drop=True)
        return res

    def convert_records_step(self, op, *, data_map, narrow):
        if op.node_name != "ConvertRecordsNode":
            raise TypeError(
                "op was supposed to be a data_algebra.data_ops.ConvertRecordsNode"
            )
        res = op.sources[0].eval_implementation(
            data_map=data_map, data_model=self, narrow=narrow
        )
        return op.record_map.transform(res, local_data_model=self)
