from typing import Union
import collections

import data_algebra.env

# for some ideas in capturing expressions in Python see:
#  scipy
# pipe-like idea
#  http://code.activestate.com/recipes/384122-infix-operators/
#  http://tomerfiliba.com/blog/Infix-Operators/


class Term:
    """Inherit from this class to capture expressions.
    Abstract class, should be extended for use.-"""

    source_string: Union[str, None]

    def __init__(self,):
        self.source_string = None

    def get_column_names(self, columns_seen):
        pass

    def __op_expr__(self, op, other):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        if not isinstance(other, Term):
            other = Value(other)
        return Expression(op, (self, other), inline=True)

    def __rop_expr__(self, op, other):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        if not isinstance(other, Term):
            other = Value(other)
        return Expression(op, (other, self), inline=True)

    def __uop_expr__(self, op, *, params=None):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        return Expression(op, (self,), params=params)

    def to_python(self, *, want_inline_parens=False):
        raise Exception("base class called")

    def to_pandas(self, *, want_inline_parens=False):
        return self.to_python(want_inline_parens=want_inline_parens)

    # noinspection PyPep8Naming
    def to_R(self, *, want_inline_parens=False):
        return self.to_pandas(want_inline_parens=want_inline_parens)

    def to_source(self, *, want_inline_parens=False, dialect="Python"):
        if dialect == "Python":
            return self.to_python(want_inline_parens=want_inline_parens)
        elif dialect == "Pandas":
            return self.to_pandas(want_inline_parens=want_inline_parens)
        elif dialect == "R":
            return self.to_R(want_inline_parens=want_inline_parens)
        else:
            raise Exception("unexpected dialect string: " + str(dialect))

    def __repr__(self):
        return self.to_python(want_inline_parens=False)

    def __str__(self):
        return self.to_python(want_inline_parens=False)

    # try to get at == and other comparision operators

    def __eq__(self, other):
        return self.__op_expr__("==", other)

    def __ne__(self, other):
        return self.__op_expr__("!=", other)

    def __lt__(self, other):
        return self.__op_expr__("<", other)

    def __le__(self, other):
        return self.__op_expr__("<=", other)

    def __gt__(self, other):
        return self.__op_expr__(">", other)

    def __ge__(self, other):
        return self.__op_expr__(">=", other)

    # override most of https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types

    def __add__(self, other):
        return self.__op_expr__("+", other)

    def __radd__(self, other):
        return self.__rop_expr__("+", other)

    def __sub__(self, other):
        return self.__op_expr__("-", other)

    def __rsub__(self, other):
        return self.__rop_expr__("-", other)

    def __mul__(self, other):
        return self.__op_expr__("*", other)

    def __rmul__(self, other):
        return self.__rop_expr__("*", other)

    def __matmul__(self, other):
        return self.__op_expr__("@", other)

    def __rmatmul__(self, other):
        return self.__rop_expr__("@", other)

    def __truediv__(self, other):
        return self.__op_expr__("/", other)

    def __rtruediv__(self, other):
        return self.__rop_expr__("/", other)

    def __floordiv__(self, other):
        return self.__op_expr__("//", other)

    def __rfloordiv__(self, other):
        return self.__rop_expr__("//", other)

    def __mod__(self, other):
        return self.__op_expr__("%", other)

    def __rmod__(self, other):
        return self.__rop_expr__("%", other)

    def __divmod__(self, other):
        return self.__op_expr__("divmod", other)

    def __rdivmod__(self, other):
        return self.__rop_expr__("divmod", other)

    def __pow__(self, other, modulo=None):
        return self.__op_expr__("pow", other)

    def __rpow__(self, other):
        return self.__rop_expr__("pow", other)

    def __lshift__(self, other):
        return self.__op_expr__("lshift", other)

    def __rlshift__(self, other):
        return self.__rop_expr__("lshift", other)

    def __rshift__(self, other):
        return self.__op_expr__("rshift", other)

    def __rrshift__(self, other):
        return self.__rop_expr__("rshift", other)

    def __and__(self, other):
        return self.__op_expr__("and", other)

    def __rand__(self, other):
        return self.__rop_expr__("and", other)

    def __xor__(self, other):
        return self.__op_expr__("xor", other)

    def __rxor__(self, other):
        return self.__rop_expr__("xor", other)

    def __or__(self, other):
        return self.__op_expr__("or", other)

    def __ror__(self, other):
        return self.__rop_expr__("or", other)

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
        return self.__uop_expr__("neg")

    def __pos__(self):
        return self.__uop_expr__("pos")

    def __abs__(self):
        return self.__uop_expr__("abs")

    def __invert__(self):
        return self.__uop_expr__("invert")

    def __complex__(self):
        raise Exception("cast called")

    def __int__(self):
        raise Exception("cast called")

    def __float__(self):
        raise Exception("cast called")

    def __index__(self):
        raise Exception("cast called")

    def __round__(self, ndigits=None):
        return self.__uop_expr__("neg", params={"ndigits": ndigits})

    def __trunc__(self):
        return self.__uop_expr__("trunc")

    def __floor__(self):
        return self.__uop_expr__("floor")

    def __ceil__(self):
        return self.__uop_expr__("ceil")

    # ad-hoc defs

    def max(self):
        return self.__uop_expr__("max")

    def min(self):
        return self.__uop_expr__("min")

    def exp(self):
        return self.__uop_expr__("exp")

    def sum(self):
        return self.__uop_expr__("sum")


class Value(Term):
    def __init__(self, value):
        allowed = [int, float, str, bool]
        if not any([isinstance(value, tp) for tp in allowed]):
            raise Exception("value type must be one of: " + str(allowed))
        self.value = value
        Term.__init__(self)

    def to_python(self, want_inline_parens=False):
        return self.value.__repr__()


class ColumnReference(Term):
    """class to represent referring to a column"""

    view: any  # typically a ViewReference
    column_name: str

    def __init__(self, view, column_name):
        self.view = view
        self.column_name = column_name
        if not isinstance(column_name, str):
            raise Exception("column_name must be a string")
        if column_name not in view.column_set:
            raise Exception("column_name must be a column of the given view")
        Term.__init__(self)

    def to_python(self, want_inline_parens=False):
        return self.column_name

    def get_column_names(self, columns_seen):
        columns_seen.add(self.column_name)


# map from op-name to special Python formatting code
py_formatters = {"___": lambda expression: expression.to_python()}
pd_formatters = {"___": lambda expression: expression.to_pandas()}
r_formatters = {"___": lambda expression: expression.to_R()}


class Expression(Term):
    def __init__(self, op, args, *, params=None, inline=False):
        if not isinstance(op, str):
            raise Exception("op is supposed to be a string")
        self.op = op
        self.args = args
        self.params = params
        self.inline = inline
        Term.__init__(self)

    def get_column_names(self, columns_seen):
        for a in self.args:
            a.get_column_names(columns_seen)

    def to_python(self, *, want_inline_parens=False):
        if self.op in py_formatters.keys():
            return py_formatters[self.op](self)
        subs = [ai.to_python(want_inline_parens=True) for ai in self.args]
        if len(subs) <= 0:
            return "_" + self.op + "()"
        if len(subs) == 1:
            return subs[0] + "." + self.op + "()"
        if len(subs) == 2 and self.inline:
            if want_inline_parens:
                return "(" + subs[0] + " " + self.op + " " + subs[1] + ")"
            else:
                return subs[0] + " " + self.op + " " + subs[1]
        return self.op + "(" + ", ".join(subs) + ")"

    def to_pandas(self, *, want_inline_parens=False):
        if self.op in pd_formatters.keys():
            return pd_formatters[self.op](self)
        if len(self.args) <= 0:
            return "_" + self.op + "()"
        if len(self.args) == 1:
            return (
                self.op + "(" + self.args[0].to_pandas(want_inline_parens=False) + ")"
            )
        subs = [ai.to_pandas(want_inline_parens=True) for ai in self.args]
        if len(subs) == 2 and self.inline:
            if want_inline_parens:
                return "(" + subs[0] + " " + self.op + " " + subs[1] + ")"
            else:
                return subs[0] + " " + self.op + " " + subs[1]
        return self.op + "(" + ", ".join(subs) + ")"

    def to_R(self, *, want_inline_parens=False):
        if self.op in r_formatters.keys():
            return r_formatters[self.op](self)
        if len(self.args) <= 0:
            return self.op + "()"
        if len(self.args) == 1:
            return (
                self.op + "(" + self.args[0].to_pandas(want_inline_parens=False) + ")"
            )
        subs = [ai.to_pandas(want_inline_parens=True) for ai in self.args]
        if len(subs) == 2 and self.inline:
            if want_inline_parens:
                return "(" + subs[0] + " " + self.op + " " + subs[1] + ")"
            else:
                return subs[0] + " " + self.op + " " + subs[1]
        return self.op + "(" + ", ".join(subs) + ")"


# Some notes on trying to harden eval:
#  http://lybniz2.sourceforge.net/safeeval.html


def _eval_by_parse(source_str, *, data_def, outter_environemnt=None):
    if not isinstance(source_str, str):
        source_str = str(source_str)
    if outter_environemnt is None:
        outter_environemnt = {}
    v = eval(
        source_str, outter_environemnt, data_def
    )  # eval is eval(source, globals, locals)- so mp is first
    if not isinstance(v, Term):
        v = Value(v)
    v.source_string = source_str
    return v


def check_convert_op_dictionary(ops, column_defs, *, parse_env=None):
    """
    Convert all entries of ops map to Term-expressions

    Note: eval() is called to interpret expressions on some nodes, so this
       function is not safe to use on untrusted code (though a somewhat restricted
       version of eval() is used to try and catch some issues).
    """
    if not isinstance(ops, dict):
        raise Exception("ops should be a dictionary")
    if not isinstance(column_defs, dict):
        raise Exception("column_defs should be a dictionary")
    if parse_env is None:
        parse_env = data_algebra.env.outer_namespace()
        if parse_env is None:
            parse_env = {}
    # first: make sure all entries are parsed
    columns_used = set()
    newops = collections.OrderedDict()
    mp = column_defs.copy()
    data_algebra.env.populate_specials(
        column_defs=column_defs, destination=mp, user_values=parse_env
    )
    sub_env = {k: v for (k, v) in parse_env.items() if not k.startswith("_")}
    sub_env["__builtins__"] = None
    for k in ops.keys():
        if not isinstance(k, str):
            raise Exception("ops keys should be strings")
        ov = ops[k]
        v = ov
        if not isinstance(v, Term):
            failue = False
            # noinspection PyBroadException
            try:
                v = _eval_by_parse(
                    source_str=v, data_def=mp, outter_environemnt=sub_env
                )
            except Exception:
                failue = True
            if failue:
                raise Exception("parse failed on " + k + " = " + ov)
        newops[k] = v
        used_here = set()
        v.get_column_names(used_here)
        columns_used = columns_used.union(
            used_here - {k}
        )  # can use a column to update itself
    intersect = set(ops.keys()).intersection(columns_used)
    if len(intersect) > 0:
        raise Exception(
            "columns both produced and used in same expression set: " + str(intersect)
        )
    return newops
