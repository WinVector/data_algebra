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
        assert isinstance(pd, types.ModuleType)
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
        if not narrow:
            columns_using = [c for c in df.columns]
        missing = set(columns_using) - set([c for c in df.columns])
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        # make an index-free copy of the data to isolate side-effects and not deal with indices
        res = df.loc[:, columns_using]
        res = res.reset_index(drop=True)
        # check type compatibility
        if (op.column_types is not None) and (len(op.column_types) > 0):
            types_seen = data_algebra.util.guess_column_types(res)
            for c in columns_using:
                assert data_algebra.util.compatible_types({types_seen[c]}.union({op.column_types[c]}))
        return res

    def columns_to_frame_(self, cols, *, target_rows=0):
        """
        Convert a dictionary of column names to series-like objects and scalars into a Pandas data frame.

        :param cols: dictionary mapping column names to columns
        :return: Pandas data frame.
        """
        # noinspection PyUnresolvedReferences
        assert isinstance(cols, dict)
        if len(cols) < 1:
            return self.pd.DataFrame(cols)
        for k, v in cols.items():
            try:
                target_rows = max(target_rows, len(v))
            except TypeError:
                target_rows = max(target_rows, 1)  # scalar
        if target_rows < 1:
            return self.pd.DataFrame(cols)

        # agg can return scalars, which then can't be made into a self.pd.DataFrame
        def promote_scalar(v, *, target_len):
            # noinspection PyBroadException
            try:
                len_v = len(v)
                if len_v != target_len:
                    if len_v == 0:
                        return [None] * target_len
                    elif len_v == 1:
                        return [v[0]] * target_len
                    else:
                        raise ValueError("incompatible column lengths")
            except Exception:
                return [v] * target_len  # scalar
            return v

        cols = {k: promote_scalar(v, target_len=target_rows) for (k, v) in cols.items()}
        return self.pd.DataFrame(cols)

    def add_data_frame_columns_to_data_frame_(self, res, transient_new_frame):
        if (res.shape[0] == 0) and (transient_new_frame.shape[0] > 0):
            # scalars get interpreted as single row items, instead of zero row items
            # growing the extension frame
            transient_new_frame = transient_new_frame.iloc[range(0), :].reset_index(drop=True)
        if res.shape[0] == transient_new_frame.shape[0]:
            if res.shape[1] < 1:
                return transient_new_frame
            if transient_new_frame.shape[1] < 1:
                return res
        if (2 * transient_new_frame.shape[1]) > res.shape[1]:
            # lots of columns path
            # https://win-vector.com/2021/08/03/i-think-pandas-may-have-lost-the-plot/
            for c in set(res.columns).intersection(set(transient_new_frame.columns)):
                del res[c]
            return self.pd.concat([res, transient_new_frame], axis=1)
        # normal path
        for c in transient_new_frame.columns:
           res[c] = transient_new_frame[c]
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
        if not window_situation:
            new_cols = {k: opk.evaluate(res) for k, opk in op.ops.items()}
            # for k, v in new_cols.items():
            #     res[k] = v
            new_frame = self.columns_to_frame_(new_cols, target_rows=res.shape[0])
            res = self.add_data_frame_columns_to_data_frame_(res, new_frame)
        else:
            data_algebra_temp_cols = {}
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
                if len(opk.args) > 0:
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in col_set:
                            col_list = col_list + [value_name]
                            col_set.add(value_name)
                    elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        key = str(opk.args[0].value)
                        if key not in data_algebra_temp_cols.keys():
                            value_name = "data_algebra_extend_temp_col_" + str(
                                len(data_algebra_temp_cols)
                            )
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
                # TODO: check if need any of these in project
                if len(opk.args) <= 0:
                    if opk.op in {"row_number", "_row_number", "count", "_count"}:
                        subframe[k] = opframe.cumcount() + 1
                    elif opk.op in {"ngroup", "_ngroup"}:
                        subframe[k] = opframe.ngroup()
                    elif opk.op in {"size", "_size"}:
                        subframe[k] = opframe[standin_name].transform(
                            opk.op
                        )  # Pandas transform, not data_algebra
                    else:
                        raise KeyError("not implemented: " + str(k) + ": " + str(opk))
                else:
                    transform_args = []
                    if len(opk.args) > 1:
                        for i in range(1, len(opk.args)):
                            assert isinstance(opk.args[i], data_algebra.expr_rep.Value)
                        transform_args = [opk.args[i].value for i in range(1, len(opk.args))]
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in set(col_list):
                            col_list = col_list + [value_name]
                        subframe[k] = opframe[value_name].transform(
                            opk.op, *transform_args
                        )  # Pandas transform, not data_algegra
                    elif isinstance(opk.args[0], data_algebra.expr_rep.Value):
                        value_name = data_algebra_temp_cols[str(opk.args[0].value)]
                        subframe[k] = opframe[value_name].transform(
                            opk.op, *transform_args
                        )  # Pandas transform, not data_algegra
                    else:
                        raise ValueError("opk must be a ColumnReference or Value")
            # clear some temps
            for value_name in data_algebra_temp_cols.values():
                del res[value_name]
            # copy out results
            subframe = subframe.sort_values(by=["_data_algebra_orig_index"])
            subframe = subframe.loc[:, list(op.ops.keys())]
            subframe = subframe.reset_index(drop=True)
            res = self.add_data_frame_columns_to_data_frame_(res, subframe)
        return res

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
                    if key not in data_algebra_temp_cols.keys():
                        value_name = "data_algebra_project_temp_col_" + str(
                            len(data_algebra_temp_cols)
                        )
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
        res = self.columns_to_frame_(cols)
        res = res.reset_index(drop=len(op.group_by) < 1)  # grouping variables in the index
        missing_group_cols = set(op.group_by) - set(res.columns)
        assert len(missing_group_cols) <= 0
        if "_data_table_temp_col" in res.columns:
            res = res.drop("_data_table_temp_col", axis=1, inplace=False)
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
        if res.shape[0] < 1:
            return res
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
        if res.shape[0] > 1:
            ascending = [
                False if ci in set(op.reverse) else True for ci in op.order_columns
            ]
            res = res.sort_values(by=op.order_columns, ascending=ascending).reset_index(
                drop=True
            )
        if (op.limit is not None) and (res.shape[0] > op.limit):
            res = res.iloc[range(op.limit), :].reset_index(drop=True)
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
        assert isinstance(jointype, str)
        jointype = jointype.lower()
        mp = {
            "full": "outer",
            "cross": "outer",  # cross new to Pandas 1.2.0 December 2020
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
        type_checks = data_algebra.util.check_columns_appear_compatible(left, right, columns=common_cols)
        if type_checks is not None:
            raise ValueError(f"join: incompatible column types: {type_checks}")
        by = op.by
        if by is None:
            by = []
        scratch_col = None  # extra column to prevent empty-by issues
        if len(by) <= 0:
            scratch_col = "data_algebra_temp_merge_col"
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
                res = res.drop(c + "_tmp_right_col", axis=1, inplace=False)
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
        type_checks = data_algebra.util.check_columns_appear_compatible(left, right)
        if type_checks is not None:
            raise ValueError(f"concat: incompatible column types: {type_checks}")
        if left.shape[0] < 1:
            return right
        if right.shape[0] < 1:
            return left
        # noinspection PyUnresolvedReferences
        res = self.pd.concat([left, right], axis=0, ignore_index=True, sort=False)
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
