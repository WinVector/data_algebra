import data_algebra.expr_rep
from data_algebra.data_ops import *


class Extend(PipeStep):
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
        PipeStep.__init__(self)
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
                raise ValueError(
                    "columns in reverse that are not in order_by: " + str(unknown)
                )
        self._ops = ops
        self.partition_by = partition_by
        self.order_by = order_by
        self.reverse = reverse

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
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
        return (
            "Extend("
            + self._ops.__repr__()
            + ", partition_by="
            + self.partition_by.__repr__()
            + ", order_by="
            + self.order_by.__repr__()
            + ", reverse="
            + self.reverse.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class Project(PipeStep):
    """Class to specify aggregating or summarizing columns."""

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops=None, *, group_by=None):
        PipeStep.__init__(self)
        if isinstance(group_by, str):
            group_by = [group_by]
        if ops is None:
            ops = {}
        self._ops = ops
        self.group_by = group_by

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.project(ops=self._ops, group_by=self.group_by, parse_env=parse_env)

    def __repr__(self):
        return (
            "Project("
            + self._ops.__repr__()
            + ", group_by="
            + self.group_by.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class SelectRows(PipeStep):
    """Class to specify a choice of rows.
    """

    expr: data_algebra.expr_rep.Expression

    def __init__(self, expr):
        PipeStep.__init__(self)
        self.expr = expr

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.select_rows(expr=self.expr, parse_env=parse_env)

    def __repr__(self):
        return "SelectRows(" + self.expr.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class SelectColumns(PipeStep):
    """Class to specify a choice of columns.
    """

    column_selection: List[str]

    def __init__(self, columns):
        PipeStep.__init__(self)
        if isinstance(columns, str):
            columns = [columns]
        column_selection = [c for c in columns]
        self.column_selection = column_selection

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.select_columns(self.column_selection)

    def __repr__(self):
        return "SelectColumns(" + self.column_selection.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class DropColumns(PipeStep):
    """Class to specify removal of columns.
    """

    column_deletions: List[str]

    def __init__(self, column_deletions):
        PipeStep.__init__(self)
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.drop_columns(self.column_deletions)

    def __repr__(self):
        return "DropColumns(" + self.column_deletions.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class OrderRows(PipeStep):
    """Class to specify a columns to determine row order.
    """

    order_columns: List[str]
    reverse: List[str]

    def __init__(self, columns, *, reverse=None, limit=None):
        PipeStep.__init__(self)
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(reverse, str):
            reverse = [reverse]
        if reverse is not None and len(reverse) > 0:
            if columns is None:
                raise ValueError("set is None when order_by is not None")
            unknown = set(reverse) - set(columns)
            if len(unknown) > 0:
                raise ValueError(
                    "columns in reverse that are not in order_by: " + str(unknown)
                )
        self.order_columns = [c for c in columns]
        if reverse is None:
            reverse = []
        self.reverse = [c for c in reverse]
        self.limit = limit

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.order_rows(
            columns=self.order_columns, reverse=self.reverse, limit=self.limit
        )

    def __repr__(self):
        return (
            "OrderRows("
            + self.order_columns.__repr__()
            + ", reverse="
            + self.reverse.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class RenameColumns(PipeStep):
    """Class to rename columns.
    """

    column_remapping: Dict[str, str]

    def __init__(self, column_remapping):
        PipeStep.__init__(self)
        self.column_remapping = column_remapping.copy()

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.rename_columns(column_remapping=self.column_remapping)

    def __repr__(self):
        return "RenameColumns(" + self.column_remapping.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class NaturalJoin(PipeStep):
    _by: List[str]
    _jointype: str
    _b: OperatorPlatform

    def __init__(self, *, b=None, by=None, jointype="INNER"):
        PipeStep.__init__(self)
        if not isinstance(b, ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        if isinstance(by, str):
            by = [by]
        self._by = by
        self._jointype = data_algebra.expr_rep.standardize_join_type(jointype)
        self._b = b

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.natural_join(b=self._b, by=self._by, jointype=self._jointype)

    def __repr__(self):
        return (
            "NaturalJoin("
            + "b="
            + self._b.__repr__()
            + ", by="
            + self._by.__repr__()
            + ", jointype="
            + self._jointype.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class ConcatRows(PipeStep):
    _id_column: str
    _b: OperatorPlatform

    def __init__(self, *, b=None, id_column="table_name", a_name="a", b_name="b"):
        PipeStep.__init__(self)
        if not isinstance(b, ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        self._id_column = id_column
        self.a_name = a_name
        self.b_name = b_name
        self._b = b

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.concat_rows(
            b=self._b, id_column=self._id_column, a_name=self.a_name, b_name=self.b_name
        )

    def __repr__(self):
        return (
            "ConcatRows("
            + "b="
            + self._b.__repr__()
            + ", id_column="
            + self._id_column.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class ConvertRecords(PipeStep):
    def __init__(self, record_map):
        PipeStep.__init__(self)
        self.record_map = record_map

    def apply_to(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.convert_records(record_map=self.record_map)

    def __repr__(self):
        return (
            "ConvertRecords("
            + self.record_map.__repr__()
            + ", record_map="
            + self.record_map.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()
