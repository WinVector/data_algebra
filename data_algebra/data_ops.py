
import types

import data_algebra.table_rep
import data_algebra.pipe


class ViewRepresentation(data_algebra.pipe.PipeValue):
    """Structure to represent the columns of a query or a table"""

    def __init__(self, table_name, column_names,
                 *,
                 qualifiers = None,
                 view_name = None):
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
        if len(self.column_names)<1:
            raise Exception("no column names")
        self._column_set = set(self.column_names)
        if not len(self.column_names) == len(self._column_set):
            raise Exception("duplicate column name(s)")
        column_dict = {ci:data_algebra.table_rep.ColumnReference(self, ci) for ci in self.column_names}
        self.column_map = types.SimpleNamespace(**column_dict)
        data_algebra.pipe.PipeValue.__init__(self)

    def __repr__(self):
        return (self._view_name + '(' + self.table_name.__repr__() +
                    ", " + self.column_names.__repr__() +
                    ", " + self.qualifiers.__repr__() +
                    ")")

    def __str__(self):
        if len(self.qualifiers) <= 0:
            return self.table_name
        return str(self.qualifiers) + '.' + self.table_name

    # define builders for all node types on base class

    def extend(self, ops):
        ops = data_algebra.table_rep.check_convert_op_dictionary(ops, self.column_map.__dict__)
        return ExtendNode(self, ops)


def _maybe_set_underbar(mp, mp1=None):
    # make last result referable by names _ and _0
    if data_algebra._ref_to_global_namespace is not None:
        data_algebra._ref_to_global_namespace['_'] = mp
        data_algebra._ref_to_global_namespace['_0'] = mp
        data_algebra._ref_to_global_namespace['_1'] = mp1


def mk_td(table_name, column_names,
          *,
          qualifiers = None):
    """Make a table representation object.

       If data_algebra._ref_to_global_namespace = globals() then
       _ and _0 are set to column name maps as a side-effect.

       Example:
           from data_algebra.data_ops import *
           data_algebra._ref_to_global_namespace = globals()
           d = mk_td('d', ['x', 'y'])
           print(_)
    """
    vr = ViewRepresentation(
        table_name = table_name,
        column_names = column_names,
        qualifiers = qualifiers)
    # make last result referable by names _ and _0
    _maybe_set_underbar(vr.column_map)
    return vr


class ExtendNode(ViewRepresentation):

    def __init__(self, source, ops):
        if not isinstance(source, ViewRepresentation):
            raise Exception("source must be a ViewRepresentation")
        column_names = source.column_names.copy()
        known_cols = set(column_names)
        for ci in ops.keys():
            if ci not in known_cols:
                column_names = column_names + [ci]
        ViewRepresentation.__init__(
            self,
            table_name=None,
            column_names=column_names
        )
        self._source = source
        self._ops = ops
        _maybe_set_underbar(self.column_map)

    def __repr__(self):
        return str(self._source) + " >>\n    " + "Extend(" + str(self._ops) + ")"

    def __str__(self):
        return str(self._source) + " >>\n    " + "Extend(" + str(self._ops) + ")"


class Extend(data_algebra.pipe.PipeStep):
    """Class to specify adding or altering columns.

       If data_algebra._ref_to_global_namespace = globals() then
       _ and _0 are set to column name maps as a side-effect.

       Example:
           from data_algebra.data_ops import *
           data_algebra._ref_to_global_namespace = globals() # needed to define _
           ops = (
              mk_td('d', ['x', 'y']) >>
                 Extend({'z':_.x + _.y})
            )
            print(ops)
            ops2 = (
                mk_td('d', ['x', 'y']) .
                    extend({'z':'1 + x'})
            )
            print(ops2.column_map)
            ops3 = (
                mk_td('d', ['x', 'y']) .
                    extend({'z':'1 + _.x'})
            )
            print(ops3.column_map)
    """

    def __init__(self, ops):
        data_algebra.pipe.PipeStep.__init__(
            self,
            name="Extend")
        self._ops = ops

    def apply(self, other):
        return other.extend(self._ops)

