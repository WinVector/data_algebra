import types

import data_algebra.table_rep
import data_algebra.pipe


class ViewRepresentation(data_algebra.pipe.PipeValue):
    """Structure to represent the columns of a query or a table"""

    def __init__(self, table_name, column_names, *, qualifiers=None, view_name=None):
        if (table_name is not None) and (not isinstance(table_name, str)):
            raise Exception("table_name must be a string")
        if view_name is None:
            view_name = "ViewRepresentation"
        self._view_name = view_name
        self.table_name = table_name
        self.column_names = [c for c in column_names]
        if qualifiers is None:
            qualifiers = {}
        if not isinstance(qualifiers, dict):
            raise Exception("qualifiers must be a dictionary")
        self.qualifiers = qualifiers.copy()
        for ci in self.column_names:
            if not isinstance(ci, str):
                raise Exception("non-string column name(s)")
        if len(self.column_names) < 1:
            raise Exception("no column names")
        self._column_set = set(self.column_names)
        if not len(self.column_names) == len(self._column_set):
            raise Exception("duplicate column name(s)")
        column_dict = {
            ci: data_algebra.table_rep.ColumnReference(self, ci)
            for ci in self.column_names
        }
        self.column_map = types.SimpleNamespace(**column_dict)
        data_algebra.pipe.PipeValue.__init__(self)

    def __repr__(self):
        return (
            self._view_name
            + "("
            + self.table_name.__repr__()
            + ", "
            + self.column_names.__repr__()
            + ", "
            + self.qualifiers.__repr__()
            + ")"
        )

    def __str__(self):
        if len(self.qualifiers) <= 0:
            return self.table_name
        return str(self.qualifiers) + "." + self.table_name

    # define builders for all node types on base class

    def extend(self, ops):
        return ExtendNode(self, ops)


def _maybe_set_underbar(mp, mp1=None):
    # make last result referable by names _ and _0
    ns = data_algebra.outer_namespace()
    if ns is not None:
        ns["_"] = mp
        ns["_0"] = mp
        ns["_1"] = mp1


def mk_td(table_name, column_names, *, qualifiers=None):
    """Make a table representation object.

       If outer namespace is set user values are visible and
       _-side effects can be written back.

       Example:
           from data_algebra.data_ops import *
           import data_algebra
           with data_algebra.Env(globals()) as env:
               d = mk_td('d', ['x', 'y'])
           print(_) # should be a namespace, not d
           print(d)
    """
    vr = ViewRepresentation(
        table_name=table_name, column_names=column_names, qualifiers=qualifiers
    )
    # make last result referable by names _ and _0
    _maybe_set_underbar(vr.column_map)
    return vr


class ExtendNode(ViewRepresentation):
    def __init__(self, source, ops):
        if not isinstance(source, ViewRepresentation):
            raise Exception("source must be a ViewRepresentation")
        ops = data_algebra.table_rep.check_convert_op_dictionary(
            ops, source.column_map.__dict__
        )
        column_names = source.column_names.copy()
        known_cols = set(column_names)
        for ci in ops.keys():
            if ci not in known_cols:
                column_names = column_names + [ci]
        ViewRepresentation.__init__(self, table_name=None, column_names=column_names)
        self._source = source
        self._ops = ops
        _maybe_set_underbar(self.column_map)

    def __repr__(self):
        return str(self._source) + " >>\n    " + "Extend(" + str(self._ops) + ")"

    def __str__(self):
        return str(self._source) + " >>\n    " + "Extend(" + str(self._ops) + ")"


class Extend(data_algebra.pipe.PipeStep):
    """Class to specify adding or altering columns.

       If outer namespace is set user values are visible and
       _-side effects can be written back.

       Examples:
           from data_algebra.data_ops import *
           import data_algebra
           with data_algebra.Env(globals()) as env:
               q = 4

               print("first example")
               ops = (
                  mk_td('d', ['x', 'y']) >>
                     Extend({'z':_.x + _.y/q})
                )
                print(ops)

                print("ex 2")
                ops2 = (
                    mk_td('d', ['x', 'y']) .
                        extend({'z':'1/q + x'})
                )
                print(ops2)

                print("ex 3")
                ops3 = (
                    mk_td('d', ['x', 'y']) .
                        extend({'z':'1/q + _.x', 'f':1, 'g':'"2"'})
                )
                print(ops3)

                print("ex 4")
                import data_algebra.pipe

                ops4 = data_algebra.pipe.build_pipeline(
                    mk_td('d', ['x', 'y']),
                    Extend({'z':'1/_.y + 1/q', 'x':'x+1'})
                )
                print(ops4)
    """

    def __init__(self, ops):
        data_algebra.pipe.PipeStep.__init__(self, name="Extend")
        self._ops = ops

    def apply(self, other):
        return other.extend(self._ops)
