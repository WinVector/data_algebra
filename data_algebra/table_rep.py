
import types

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

    def get_column_names(self, columns_seen):
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

    def __uop_expr__(self, op, *, params=None):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        return Expression(op, (self), params=params)

    # override most of https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

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

    def __iadd__(self, other):
        raise Exception("assignment operator called")

    def __isub__(self, other):
        raise Exception("assignment operator called")

    def __imul__(self, other):
        raise Exception("assignment operator called")

    def __imatmul__(self, other):
        raise Exception("assignment operator called")

    def __itruediv__(self, other):
        raise Exception("assignment operator called")

    def __ifloordiv__(self, other):
        raise Exception("assignment operator called")

    def __imod__(self, other):
        raise Exception("assignment operator called")

    def __ipow__(self, other, modulo=None):
        raise Exception("assignment operator called")

    def __ilshift__(self, other):
        raise Exception("assignment operator called")

    def __irshift__(self, other):
        raise Exception("assignment operator called")

    def __iand__(self, other):
        raise Exception("assignment operator called")

    def __ixor__(self, other):
        raise Exception("assignment operator called")

    def __ior__(self, other):
        raise Exception("assignment operator called")

    def __neg__(self):
        return self.__uop_expr__('neg')

    def __pos__(self):
        return self.__uop_expr__('pos')

    def __abs__(self):
        return self.__uop_expr__('abs')

    def __invert__(self):
        return self.__uop_expr__('invert')

    def __complex__(self):
        raise Exception("cast called")

    def __int__(self):
        raise Exception("cast called")

    def __float__(self):
        raise Exception("cast called")

    def __index__(self):
        raise Exception("cast called")

    def __round__(self, ndigits=None):
        return self.__uop_expr__('neg',
                                 params={'ndigits':ndigits})

    def __trunc__(self):
        return self.__uop_expr__('trunc')

    def __floor__(self):
        return self.__uop_expr__('floor')

    def __ceil__(self):
        return self.__uop_expr__('ceil')



class Value(Term):
    def __init__(self, value):
        allowed = [int, float, str, bool]
        if not any([isinstance(value, tp) for tp in allowed]):
            raise Exception("value type must be one of: " + str(allowed))
        self.value = value

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return str(self.value)


class Expression(Term):
    def __init__(self, op, args, *, params=None):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        if len(args)<1:
            raise Exception("args is not supposed to be empty")
        self.op = op
        self.args = args
        self.params = params

    def get_column_names(self, columns_seen):
        for a in self.args:
            a.get_column_names(columns_seen)

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

    def get_column_names(self, columns_seen):
        columns_seen.add(self.column_name)

    def __repr__(self):
        # not a full repr
        if self.table.table_name is None:
            return self.column_name
        return self.table.table_name  + "." + self.column_name

    def __str__(self):
        if self.table.table_name is None:
            return self.column_name
        return self.table.table_name + "." + self.column_name


def check_convert_op_dictionary(ops, column_defs):
    if not isinstance(ops, dict):
        raise Exception("ops should be a dictionary")
    if not isinstance(column_defs, dict):
        raise Exception("column_defs should be a dictionary")
    # first: make sure all entries are parsed
    columns_used = set()
    newops = {}
    mp = column_defs.copy()
    ns = types.SimpleNamespace(**column_defs.copy())
    mp["_"] = ns
    mp["_0"] = ns
    mp["_1"] = ns
    for k in ops.keys():
        if not isinstance(k, str):
            raise Exception("ops keys should be strings")
        v = ops[k]
        if not isinstance(v, Term):
            if not isinstance(v, str):
                raise Exception("ops values must be Terms or strings")
            v = eval(v, mp)
        newops[k] = v
        v.get_column_names(columns_used)
    intersect = set(ops.keys()).intersection(columns_used)
    if len(intersect)>0:
        raise Exception("columns both produced and used in same expression set: " + str(intersect))
    return newops

