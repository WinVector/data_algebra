
import types

import data_algebra.pipe


# for some ideas in capturing expressions in Python see:
#  scipy
# pipe-like idea
#  http://code.activestate.com/recipes/384122-infix-operators/
#  http://tomerfiliba.com/blog/Infix-Operators/


class Term:
    """Inherit from this class to capture expressions.
    Abstract class, should be extended for use.-"""

    def __init__(self, ):
        pass

    def __op_expr__(self, op, other):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        if not isinstance(other, Term):
            other = Value(other)
        return Expression(op, (self, other))

    def __rop_expr__(self, op, other):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        if not isinstance(other, Term):
            other = Value(other)
        return Expression(op, (other, self))

    def __add__(self, other):
        return self.__op_expr__('+', other)

    def __radd__(self, other):
        return self.__rop_expr__('+', other)

    def __sub__(self, other):
        return self.__op_expr__('-', other)

    def __rsub__(self, other):
        return self.__rop_expr__('-', other)

    def __mul__(self, other):
        return self.__op_expr__('*', other)

    def __rmul__(self, other):
        return self.__rop_expr__('*', other)

    def __matmul__(self, other):
        return self.__op_expr__('@', other)

    def __rmatmul__(self, other):
        return self.__rop_expr__('@', other)

    def __truediv__(self, other):
        return self.__op_expr__('/', other)

    def __rtruediv__(self, other):
        return self.__rop_expr__('/', other)

    def __floordiv__(self, other):
        return self.__op_expr__('//', other)

    def __rfloordiv__(self, other):
        return self.__rop_expr__('//', other)

    def __mod__(self, other):
        return self.__op_expr__('%', other)

    def __rmod__(self, other):
        return self.__rop_expr__('%', other)

    def __divmod__(self, other):
        return self.__op_expr__('divmod', other)

    def __rdivmod__(self, other):
        return self.__rop_expr__('divmod', other)

    def __pow__(self, other, modulo=None):
        return self.__op_expr__('pow', other)

    def __rpow__(self, other):
        return self.__rop_expr__('pow', other)

    def __lshift__(self, other):
        return self.__op_expr__('lshift', other)

    def __rlshift__(self, other):
        return self.__rop_expr__('lshift', other)

    def __rshift__(self, other):
        return self.__op_expr__('rshift', other)

    def __rrshift__(self, other):
        return self.__rop_expr__('rshift', other)

    def __and__(self, other):
        return self.__op_expr__('and', other)

    def __rand__(self, other):
        return self.__rop_expr__('and', other)

    def __xor__(self, other):
        return self.__op_expr__('xor', other)

    def __rxor__(self, other):
        return self.__rop_expr__('xor', other)

    def __or__(self, other):
        return self.__op_expr__('or', other)

    def __ror__(self, other):
        return self.__rop_expr__('or', other)


class Value(Term):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return str(self.value)


class Expression(Term):
    def __init__(self, op, args):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        if len(args)<1:
            raise Exception("args is not supposed to be empty")
        self.op = op
        self.args = args

    def __repr__(self):
        # not a full repr
        return self.op + str(self.args)

    def __str__(self):
        return self.op + str(self.args)


class ColumnReference(Term):
    """class to represent referring to a column"""

    def __init__(self, table, column_name):
        self.table = table
        self.column_name = column_name
        if not isinstance(column_name, str):
            raise Exception("column_name must be a string")
        if column_name not in table._column_set:
            raise Exception("column_name must be a column of the given table")

    def __repr__(self):
        # not a full repr
        return str(self.table) + "." + self.column_name

    def __str__(self):
        return str(self.table) + "." + self.column_name



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
        column_dict = {ci:ColumnReference(self, ci) for ci in self.column_names}
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


