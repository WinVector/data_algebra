from typing import Dict, List

import pandas

import data_algebra.pipe
import data_algebra.data_ops
import data_algebra.expr_rep


class Extend(data_algebra.pipe.PipeStep):
    """Class to specify adding or altering columns.

       If outer namespace is set user values are visible and
       _-side effects can be written back.

       Example:
           from data_algebra.data_ops import *
           import data_algebra.env
           with data_algebra.env.Env(locals()) as env:
               q = 4
               x = 2
               var_name = 'y'

               print("first example")
               ops = (
                  TableDescription('d', ['x', 'y']) .
                     extend({'z':_.x + _[var_name]/q + _get('x')})
                )
                print(ops)
    """

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops, *, partition_by=None, order_by=None, reverse=None):
        data_algebra.pipe.PipeStep.__init__(self, name="Extend")
        self._ops = ops
        self.partition_by = partition_by
        self.order_by = order_by
        self.reverse = reverse

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.extend(
            ops=self._ops,
            partition_by=self.partition_by,
            order_by=self.order_by,
            reverse=self.reverse,
        )


class Project(data_algebra.pipe.PipeStep):
    """Class to specify aggregating or summarizing columns."""

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops, *, group_by=None, order_by=None, reverse=None):
        data_algebra.pipe.PipeStep.__init__(self, name="Project")
        self._ops = ops
        self.group_by = group_by
        self.order_by = order_by
        self.reverse = reverse

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.project(
            ops=self._ops,
            group_by=self.group_by,
            order_by=self.order_by,
            reverse=self.reverse,
        )


class SelectRows(data_algebra.pipe.PipeStep):
    """Class to specify a choice of rows.
    """

    expr: data_algebra.expr_rep.Expression

    def __init__(self, expr):
        data_algebra.pipe.PipeStep.__init__(self, name="SelectRows")
        self.expr = expr

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.select_rows(expr=self.expr)


class SelectColumns(data_algebra.pipe.PipeStep):
    """Class to specify a choice of columns.
    """

    column_selection: List[str]

    def __init__(self, columns):
        column_selection = [c for c in columns]
        self.column_selection = column_selection
        data_algebra.pipe.PipeStep.__init__(self, name="SelectColumns")

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.select_columns(self.column_selection)


class DropColumns(data_algebra.pipe.PipeStep):
    """Class to specify removal of columns.
    """

    column_deletions: List[str]

    def __init__(self, column_deletions):
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions
        data_algebra.pipe.PipeStep.__init__(self, name="DropColumns")

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.drop_columns(self.column_deletions)


class OrderRows(data_algebra.pipe.PipeStep):
    """Class to specify a columns to determine row order.
    """

    order_columns: List[str]
    reverse: List[str]

    def __init__(self, columns, *, reverse=None, limit=None):
        self.order_columns = [c for c in columns]
        if reverse is None:
            reverse = []
        self.reverse = [c for c in reverse]
        self.limit = limit
        data_algebra.pipe.PipeStep.__init__(self, name="OrderRows")

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.order_rows(
            columns=self.order_columns, reverse=self.reverse, limit=self.limit
        )


class RenameColumns(data_algebra.pipe.PipeStep):
    """Class to rename columns.
    """

    column_remapping: Dict[str, str]

    def __init__(self, column_remapping):
        self.column_remapping = column_remapping.copy()
        data_algebra.pipe.PipeStep.__init__(self, name="RenameColumns")

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.rename_columns(column_remapping=self.column_remapping)


class NaturalJoin(data_algebra.pipe.PipeStep):
    _by: List[str]
    _jointype: str
    _b: data_algebra.data_ops.OperatorPlatform

    def __init__(self, *, b=None, by=None, jointype="INNER"):
        if not isinstance(b, data_algebra.data_ops.OperatorPlatform):
            raise Exception("b should be a data_algebra.data_ops.OperatorPlatform")
        self._by = by
        self._jointype = jointype
        self._b = b
        data_algebra.pipe.PipeStep.__init__(self, name="NaturalJoin")

    def apply(self, other):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise Exception(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.natural_join(b=self._b, by=self._by, jointype=self._jointype)


class Locum(data_algebra.data_ops.OperatorPlatform):
    """Class to represent future opertions."""

    def __init__(self, name=None):
        data_algebra.data_ops.OperatorPlatform.__init__(self)
        self.ops = []
        self.name = name

    # noinspection PyPep8Naming
    def realize(self, X):
        if isinstance(X, pandas.DataFrame):
            pipeline = data_algebra.data_ops.describe_pandas_table(X, table_name="X")
            for s in self.ops:
                # pipeline = pipeline >> s
                pipeline = s.apply(pipeline)
            return pipeline
        raise Exception("can not apply realize() to type " + str(type(X)))

    def eval_pandas(self, data_map):
        if len(data_map) < 1:
            raise Exception("eval_pandas can not be applited to empty table map")
        if len(data_map) != 1:
            raise Exception("eval_pandas not yet implemented for more than one table")
        k = [k for k in data_map.keys()][0]
        return self.transform(data_map[k])

    # noinspection PyPep8Naming
    def transform(self, X):
        if isinstance(X, pandas.DataFrame):
            pipeline = self.realize(X)
            return pipeline.transform(X)
        raise Exception("can not apply transform() to type " + str(type(X)))

    def __rrshift__(self, other):  # override other >> self
        if not isinstance(other, pandas.DataFrame):
            raise Exception("other should be a pandas.DataFrame")
        return self.transform(other)

    # implement method chaining collection of pending operations

    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        op = Extend(
                ops=ops,
                partition_by=partition_by,
                order_by=order_by,
                reverse=reverse,
            )
        self.ops.append(op)
        return self

    def project(self, ops, *, group_by=None, order_by=None, reverse=None):
        op = Project(
                ops=ops,
                group_by=group_by,
                order_by=order_by,
                reverse=reverse,
            )
        self.ops.append(op)
        return self

    def natural_join(self, b, *, by=None, jointype="INNER"):
        op = NaturalJoin(
                by=by,
                jointype=jointype,
                b=b,
        )
        self.ops.append(op)
        return self

    def select_rows(self, expr):
        op = SelectRows(expr=expr)
        self.ops.append(op)
        return self

    def drop_columns(self, column_deletions):
        op = DropColumns(
                column_deletions=column_deletions
            )
        self.ops.append(op)
        return self

    def select_columns(self, columns):
        op = SelectColumns(columns=columns)
        self.ops.append(op)
        return self

    def rename_columns(self, column_remapping):
        op = RenameColumns(
                column_remapping=column_remapping
            )
        self.ops.append(op)
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        op = OrderRows(
                columns=columns, reverse=reverse, limit=limit
            )
        self.ops.append(op)
        return self
