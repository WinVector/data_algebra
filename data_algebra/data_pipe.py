from typing import Dict, List

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
        if isinstance(partition_by, str):
            partition_by = [partition_by]
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(reverse, str):
            reverse = [reverse]
        if reverse is not None and len(reverse) > 0:
            if order_by is None:
                raise ValueError("set is None when order_by is not None")
            unknown = set(reverse) - set(order_by)
            if len(unknown) > 0:
                raise ValueError("columns in reverse that are not in order_by: "  + str(unknown))
        data_algebra.pipe.PipeStep.__init__(self, name="Extend")
        self._ops = ops
        self.partition_by = partition_by
        self.order_by = order_by
        self.reverse = reverse

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.extend(
            ops=self._ops,
            partition_by=self.partition_by,
            order_by=self.order_by,
            reverse=self.reverse,
            parse_env=parse_env,
        )

    def __repr__(self):
        return ("Extend(" + self._ops.__repr__()
                + ", partition_by=" + self.partition_by.__repr__()
                + ", order_by=" + self.order_by.__repr__()
                + ", reverse=" + self.reverse.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class Project(data_algebra.pipe.PipeStep):
    """Class to specify aggregating or summarizing columns."""

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops, *, group_by=None):
        if isinstance(group_by, str):
            group_by = [group_by]
        data_algebra.pipe.PipeStep.__init__(self, name="Project")
        self._ops = ops
        self.group_by = group_by

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.project(ops=self._ops, group_by=self.group_by, parse_env=parse_env)

    def __repr__(self):
        return ("Project(" + self._ops.__repr__()
                + ", group_by=" + self.group_by.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class SelectRows(data_algebra.pipe.PipeStep):
    """Class to specify a choice of rows.
    """

    expr: data_algebra.expr_rep.Expression

    def __init__(self, expr):
        data_algebra.pipe.PipeStep.__init__(self, name="SelectRows")
        self.expr = expr

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.select_rows(expr=self.expr, parse_env=parse_env)

    def __repr__(self):
        return ("SelectRows(" + self.expr.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class SelectColumns(data_algebra.pipe.PipeStep):
    """Class to specify a choice of columns.
    """

    column_selection: List[str]

    def __init__(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        column_selection = [c for c in columns]
        self.column_selection = column_selection
        data_algebra.pipe.PipeStep.__init__(self, name="SelectColumns")

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.select_columns(self.column_selection)

    def __repr__(self):
        return ("SelectColumns(" + self.column_selection.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class DropColumns(data_algebra.pipe.PipeStep):
    """Class to specify removal of columns.
    """

    column_deletions: List[str]

    def __init__(self, column_deletions):
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions
        data_algebra.pipe.PipeStep.__init__(self, name="DropColumns")

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.drop_columns(self.column_deletions)

    def __repr__(self):
        return ("DropColumns(" + self.column_deletions.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class OrderRows(data_algebra.pipe.PipeStep):
    """Class to specify a columns to determine row order.
    """

    order_columns: List[str]
    reverse: List[str]

    def __init__(self, columns, *, reverse=None, limit=None):
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(reverse, str):
            reverse = [reverse]
        if reverse is not None and len(reverse) > 0:
            if columns is None:
                raise ValueError("set is None when order_by is not None")
            unknown = set(reverse) - set(columns)
            if len(unknown) > 0:
                raise ValueError("columns in reverse that are not in order_by: "  + str(unknown))
        self.order_columns = [c for c in columns]
        if reverse is None:
            reverse = []
        self.reverse = [c for c in reverse]
        self.limit = limit
        data_algebra.pipe.PipeStep.__init__(self, name="OrderRows")

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.order_rows(
            columns=self.order_columns, reverse=self.reverse, limit=self.limit
        )

    def __repr__(self):
        return ("OrderRows(" + self.order_columns.__repr__()
                + ", reverse=" + self.reverse.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class RenameColumns(data_algebra.pipe.PipeStep):
    """Class to rename columns.
    """

    column_remapping: Dict[str, str]

    def __init__(self, column_remapping):
        self.column_remapping = column_remapping.copy()
        data_algebra.pipe.PipeStep.__init__(self, name="RenameColumns")

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.rename_columns(column_remapping=self.column_remapping)

    def __repr__(self):
        return ("RenameColumns(" + self.column_remapping.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class NaturalJoin(data_algebra.pipe.PipeStep):
    _by: List[str]
    _jointype: str
    _b: data_algebra.data_ops.OperatorPlatform

    def __init__(self, *, b=None, by=None, jointype="INNER"):
        if not isinstance(b, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        if isinstance(by, str):
            by = [by]
        self._by = by
        self._jointype = jointype
        self._b = b
        data_algebra.pipe.PipeStep.__init__(self, name="NaturalJoin")

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.natural_join(b=self._b, by=self._by, jointype=self._jointype)

    def __repr__(self):
        return ("NaturalJoin("
                + ", b=" + self._b.__repr__()
                + ", by=" + self._by.__repr__()
                + ", jointype=" + self._jointype.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class ConvertRecords(data_algebra.pipe.PipeStep):
    def __init__(self, record_map, *, blocks_out_table=None):
        self.record_map = record_map
        self.blocks_out_table = blocks_out_table
        data_algebra.pipe.PipeStep.__init__(self, name="ConvertRecords")

    def apply(self, other, **kwargs):
        if not isinstance(other, data_algebra.data_ops.OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.convert_records(record_map=self.record_map,
                                     blocks_out_table=self.blocks_out_table)

    def __repr__(self):
        return ("ConvertRecords(" + self.record_map.__repr__()
                + ", record_map=" + self.record_map.__repr__()
                + ", blocks_out_table=" + self.blocks_out_table.__repr__()
                + ")"
                )

    def __str__(self):
        return self.__repr__()


class Locum(data_algebra.data_ops.OperatorPlatform):
    """Class to represent future opertions."""

    def __init__(self):
        data_algebra.data_ops.OperatorPlatform.__init__(self)
        self.ops = []

    def apply_to(self, pipeline):
        if not isinstance(pipeline, data_algebra.data_ops.OperatorPlatform):
            raise TypeError("Expected othter to be a data_algebra.data_ops.OperatorPlatform")
        for s in self.ops:
            # pipeline = pipeline >> s
            pipeline = s.apply(pipeline)
        return pipeline

    def append(self, other):
        if isinstance(other, Locum):
            for o in other.ops:
                self.ops.append(o)
        elif isinstance(other, data_algebra.pipe.PipeStep):
            self.ops.append(other)
        else:
            raise TypeError("unexpeted type for Locum + " + str(type(other)))
        return self

    def realize(self, x):
        pipeline = data_algebra.data_ops.describe_table(x, table_name="x")
        return self.apply_to(pipeline)

    # noinspection PyPep8Naming
    def transform(self, X):
        if isinstance(X, data_algebra.data_ops.OperatorPlatform):
            return self.apply_to(X)
        pipeline = self.realize(X)
        return pipeline.transform(X)

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __add__(self, other):  # override self + other
        res = Locum()
        res.append(self)
        res.append(other)
        return res

    def __radd__(self, other):  # override other + self
        return self.apply_to(other)

    # print

    def __repr__(self):
        return '[\n    ' + \
               '\n    '.join([str(o) + ',' for o in self.ops]) + \
               '\n]'

    def __str__(self):
        return '[\n    ' + \
               '\n    '.join([str(o) + ',' for o in self.ops]) + \
               '\n]'

    # implement method chaining collection of pending operations

    def extend(
            self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        if parse_env is not None:
            raise ValueError("Expected parse_env to be None")
        op = Extend(
            ops=ops, partition_by=partition_by, order_by=order_by, reverse=reverse
        )
        self.ops.append(op)
        return self

    def project(self, ops, *, group_by=None, parse_env=None):
        if parse_env is not None:
            raise ValueError("Expected parse_env to be None")
        op = Project(ops=ops, group_by=group_by)
        self.ops.append(op)
        return self

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        op = NaturalJoin(by=by, jointype=jointype, b=b)
        self.ops.append(op)
        return self

    def select_rows(self, expr, parse_env=None):
        if parse_env is not None:
            raise ValueError("Expected parse_env to be None")
        op = SelectRows(expr=expr)
        self.ops.append(op)
        return self

    def drop_columns(self, column_deletions):
        op = DropColumns(column_deletions=column_deletions)
        self.ops.append(op)
        return self

    def select_columns(self, columns):
        op = SelectColumns(columns=columns)
        self.ops.append(op)
        return self

    def rename_columns(self, column_remapping):
        op = RenameColumns(column_remapping=column_remapping)
        self.ops.append(op)
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        op = OrderRows(columns=columns, reverse=reverse, limit=limit)
        self.ops.append(op)
        return self

    def convert_records(
            self, record_map, *, blocks_out_table=None
    ):
        op = ConvertRecords(record_map=record_map, blocks_out_table=blocks_out_table)
        self.ops.append(op)
        return self


def wrap_pipeline(ops):
    return ops.apply(Locum())


def wrap_ops(*args):
    r = Locum()
    for s in args:
        s.apply(r)
    return r
