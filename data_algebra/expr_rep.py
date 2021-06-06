from abc import ABC
from typing import Union

import numpy

import data_algebra
import data_algebra.util


# for some ideas in capturing expressions in Python see:
#  scipy
# pipe-like idea
#  http://code.activestate.com/recipes/384122-infix-operators/
#  http://tomerfiliba.com/blog/Infix-Operators/


# list of window/aggregation functons that must be windowed/aggegated
# (note some other functions work in more than one mode)
fn_names_that_imply_windowed_situation = {
    "all",
    "any",
    "bfill",
    "count",
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "ffill",
    "first",
    "head",
    "is_monotonic_decreasing",
    "is_monotonic_increasing",
    "last",
    "max",
    "mean",
    "median",
    "min",
    "ngroup",
    "nlargest",
    "nsmallest",
    "nth",
    "nunique",
    "ohlc",
    "pct_change",
    "rank",
    "shift",
    "size",
    "std",
    "sum",
    "tail",
    "unique",
    "value_counts",
    "var",
}


fn_names_that_imply_ordered_windowed_situation = {
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "row_number",
}


# a competing idea sould be to remove ordering if
# operator is one of these (instead of forbid)
fn_names_that_contradict_ordered_windowed_situation = {
    "count",
    "max",
    "min",
    "prod",
    "sum",
}


class PreTerm(ABC):
    """
    abstract base class, without combination ability
    """

    source_string: Union[str, None]

    def __init__(self):
        self.source_string = None

    def is_equal(self, other):
        # can't use == as that builds a larger expression
        raise NotImplementedError("base method called")

    # tree re-write

    def replace_view(self, view):
        raise NotImplementedError("base class called")

    # analysis

    def get_column_names(self, columns_seen):
        """
        Add column names to columns_seen
        :param columns_seen: set of strings
        :return:
        """
        pass

    # eval

    def evaluate(self, data_frame):
        raise NotImplementedError("base class called")

    # emitters

    def to_python(self, *, want_inline_parens=False):
        raise NotImplementedError("base class method called")  # https://docs.python.org/3/library/exceptions.html

    def to_source(self, *, want_inline_parens=False, dialect="Python"):
        if dialect == "Python":
            return self.to_python(want_inline_parens=want_inline_parens)
        else:
            raise ValueError("unexpected dialect string: " + str(dialect))

    # printing

    def __repr__(self):
        return self.to_python(want_inline_parens=False)

    def __str__(self):
        return self.to_python(want_inline_parens=False)


class Term(PreTerm, ABC):
    """
    Abstract intermediate class with combination ability
    """

    def __init__(self):
        PreTerm.__init__(self)

    # builders

    def __op_expr__(self, op, other, *, inline=True, method=False):
        """binary expression"""
        if not isinstance(op, str):
            raise TypeError("op is supposed to be a string")
        if not isinstance(other, Term):
            other = _enc_value(other)
        return Expression(op, (self, other), inline=inline, method=method)

    def __rop_expr__(self, op, other):
        """reversed binary expression"""
        if not isinstance(op, str):
            raise TypeError("op is supposed to be a string")
        if not isinstance(other, Term):
            other = _enc_value(other)
        return Expression(op, (other, self), inline=True)

    def __uop_expr__(self, op, *, params=None, inline=False):
        """unary expression"""
        if not isinstance(op, str):
            raise TypeError("op is supposed to be a string")
        return Expression(op, (self,), params=params, inline=inline)

    def __triop_expr__(self, op, x, y, inline=False, method=False):
        """three argument expression"""
        if not isinstance(op, str):
            raise TypeError("op is supposed to be a string")
        if not isinstance(x, Term):
            x = _enc_value(x)
        if not isinstance(y, Term):
            y = _enc_value(y)
        return Expression(op, (self, x, y), inline=inline, method=method)

    # try to get at == and other comparison operators

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

    def __pow__(self, other):
        return self.__op_expr__("**", other)

    def __rpow__(self, other, inline=False):
        return self.__rop_expr__("**", other)

    def __neg__(self):
        return self.__uop_expr__("-", inline=True)

    def __pos__(self):
        return self  # Treat as a no-op

    # # TODO: need to work out how to send to Pandas and SQL
    # def __lshift__(self, other):
    #     return self.__op_expr__("<<", other)
    #
    # def __rlshift__(self, other):
    #     return self.__rop_expr__("<<", other)
    #
    # def __rshift__(self, other):
    #     return self.__op_expr__(">>", other)
    #
    # def __rrshift__(self, other):
    #     return self.__rop_expr__(">>", other)

    def __and__(self, other):
        return self.__op_expr__("&", other)

    def __rand__(self, other):
        return self.__rop_expr__("&", other)

    def __xor__(self, other):
        return self.__op_expr__("^", other)

    def __rxor__(self, other):
        return self.__rop_expr__("^", other)

    def __or__(self, other):
        return self.__op_expr__("|", other)

    def __ror__(self, other):
        return self.__rop_expr__("|", other)

    # ~ is bitwise negation in Python
    # not/~ isn't applicable to objects
    # https://docs.python.org/3/library/operator.html

    # math functions
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html

    # TODO: double check https://docs.python.org/3/library/operator.html
    # for more ops such as concat() and so on

    def sign(self):
        return self.__uop_expr__("sign")

    def sin(self):
        return self.__uop_expr__("sin")

    def cos(self):
        return self.__uop_expr__("cos")

    def arcsin(self):
        return self.__uop_expr__("arcsin")

    def arccos(self):
        return self.__uop_expr__("arccos")

    def arctan(self):
        return self.__uop_expr__("arctan")

    def arctan2(self, other):
        return self.__op_expr__("arctan2", other)

    def sinh(self):
        return self.__uop_expr__("sinh")

    def cosh(self):
        return self.__uop_expr__("cosh")

    def tanh(self):
        return self.__uop_expr__("tanh")

    def arcsinh(self):
        return self.__uop_expr__("arcsinh")

    def arccosh(self):
        return self.__uop_expr__("arccosh")

    def arctanh(self):
        return self.__uop_expr__("arctanh")

    def floor(self):
        return self.__uop_expr__("floor")

    def ceil(self):
        return self.__uop_expr__("ceil")

    def sum(self):
        return self.__uop_expr__("sum")

    def cumprod(self):
        return self.__uop_expr__("cumprod")

    def cumsum(self):
        return self.__uop_expr__("cumsum")

    def exp(self):
        return self.__uop_expr__("exp")

    def expm1(self):
        return self.__uop_expr__("expm1")

    def log(self):
        return self.__uop_expr__("log")

    def log10(self):
        return self.__uop_expr__("log10")

    def log1p(self):
        return self.__uop_expr__("log1p")

    def mod(self, other):
        return self.__op_expr__("mod", other)

    def remainder(self, other):
        return self.__op_expr__("remainder", other, inline=False, method=True)

    def sqrt(self):
        return self.__uop_expr__("sqrt")

    def abs(self):
        return self.__uop_expr__("abs")

    def maximum(self, other):
        return self.__op_expr__("maximum", other, method=True, inline=False)

    def minimum(self, other):
        return self.__op_expr__("minimum", other, method=True, inline=False)

    def fmax(self, other):
        return self.__op_expr__("fmax", other, inline=False)

    def fmin(self, other):
        return self.__op_expr__("fmin", other, inline=False)

    # more numpy stuff
    def round(self):
        return self.__uop_expr__("round")

    def around(self, other):
        assert isinstance(other, Value)  # digits control
        return self.__op_expr__("around", other, inline=False)

    # pandas style definitions
    # https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html

    def all(self):
        return self.__uop_expr__("all")

    def any(self):
        return self.__uop_expr__("any")

    def bfill(self):
        return self.__uop_expr__("bfill")

    def count(self):
        return self.__uop_expr__("count")

    def cumcount(self):
        return self.__uop_expr__("cumcount")

    def cummax(self):
        return self.__uop_expr__("cummax")

    def cummin(self):
        return self.__uop_expr__("cummin")

    def ffill(self):
        return self.__uop_expr__("ffill")

    def first(self):
        return self.__uop_expr__("first")

    def head(self):
        return self.__uop_expr__("head")

    def is_monotonic_decreasing(self):
        return self.__uop_expr__("is_monotonic_decreasing")

    def is_monotonic_increasing(self):
        return self.__uop_expr__("is_monotonic_increasing")

    def last(self):
        return self.__uop_expr__("last")

    def max(self):
        return self.__uop_expr__("max")

    def mean(self):
        return self.__uop_expr__("mean")

    def median(self):
        return self.__uop_expr__("median")

    def min(self):
        return self.__uop_expr__("min")

    def ngroup(self):
        return self.__uop_expr__("ngroup")

    def nlargest(self):
        return self.__uop_expr__("nlargest")

    def nsmallest(self):
        return self.__uop_expr__("nsmallest")

    def nth(self):
        return self.__uop_expr__("nth")

    def nunique(self):
        return self.__uop_expr__("nunique")

    def rank(self):
        return self.__uop_expr__("rank")

    def size(self):
        return self.__uop_expr__("size")

    def std(self):
        return self.__uop_expr__("std")

    def unique(self):
        return self.__uop_expr__("unique")

    def value_counts(self):
        return self.__uop_expr__("value_counts")

    def var(self):
        return self.__uop_expr__("var")

    # pandas shift

    def shift(self):
        return self.__uop_expr__("shift")

    # our ad-hoc definitions

    def is_null(self):
        return self.__uop_expr__("is_null")

    def is_bad(self):
        return self.__uop_expr__("is_bad")

    def if_else(self, x, y):
        return self.__triop_expr__("if_else", x, y, method=True)

    def is_in(self, x):
        return self.__op_expr__("is_in", x, inline=False, method=True)

    def concat(self, x):
        return self.__op_expr__("concat", x, inline=False, method=True)

    def coalesce(self, x):
        return self.__op_expr__("coalesce", x, inline=False, method=True)

    def co_equalizer(self, x):
        return self.__op_expr__("co_equalizer", x, inline=False, method=True)


class Value(Term):
    def __init__(self, value):
        allowed = [int, float, str, bool]
        if not any([isinstance(value, tp) for tp in allowed]):
            raise TypeError("value type must be one of: " + str(allowed))
        self.value = value
        Term.__init__(self)

    def is_equal(self, other):
        # can't use == as that builds a larger expression
        if not isinstance(other, Value):
            return False
        return self.value == other.value

    def replace_view(self, view):
        return self

    def evaluate(self, data_frame):
        return self.value

    def to_python(self, *, want_inline_parens=False):
        return self.value.__repr__()

    # don't collect -5 as a complex expression
    def __neg__(self):
        return Value(-self.value)


class UnQuotedStr(str):
    def __init__(self, v):
        self.v = v
        str.__init__(v)

    def str(self):
        return self.v

    def __repr__(self):
        return self.v


class ListTerm(PreTerm):
    # derived from PreTerm as this is not combinable
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("value type must be a list or tuple")
        self.value = [v for v in value]  # copy and standardize to a list
        PreTerm.__init__(self)

    def is_equal(self, other):
        # can't use == as that builds a larger expression
        if not isinstance(other, ListTerm):
            return False
        return self.value == other.value

    def replace_view(self, view):
        new_list = [ai.replace_view(view) for ai in self.value]
        return ListTerm(new_list)

    def evaluate(self, data_frame):
        res = [None] * len(self.value)
        for i in range(len(self.value)):
            vi = self.value[i]
            if isinstance(vi, PreTerm):
                vi = vi.evaluate(data_frame)
            res[i] = vi
        return res

    def to_python(self, *, want_inline_parens=False):
        def li_to_python(value, *, want_inline_parens=False):
            try:
                return value.to_python(want_inline_parens=want_inline_parens)
            except AttributeError:
                return str(value)  # TODO: check if this should be repr?

        return (
            "["
            + ", ".join(
                [
                    li_to_python(ai, want_inline_parens=want_inline_parens)
                    for ai in self.value
                ]
            )
            + "]"
        )

    def get_column_names(self, columns_seen):
        for ti in self.value:
            ti.get_column_names(columns_seen)


def _enc_value(value):
    if isinstance(value, PreTerm):
        return value
    if callable(value):
        raise ValueError("callable as an argument")
    if isinstance(value, list):
        return ListTerm(value)
    return Value(value)


class ColumnReference(Term):
    """class to represent referring to a column"""

    view: any  # typically a ViewReference
    column_name: str

    def __init__(self, view, column_name):
        self.view = view
        self.column_name = column_name
        if not isinstance(column_name, str):
            raise TypeError("column_name must be a string")
        if view is not None:
            if column_name not in view.column_set:
                raise KeyError("column_name '" + str(column_name) + "' must be a column of the given view")
        Term.__init__(self)

    def evaluate(self, data_frame):
        return data_frame[self.column_name]

    def is_equal(self, other):
        # can't use == as that builds a larger expression
        if not isinstance(other, ColumnReference):
            return False
        if self.view != other.view:
            return False
        return self.column_name == other.column_name

    def replace_view(self, view):
        return ColumnReference(view=view, column_name=self.column_name)

    def to_python(self, want_inline_parens=False):
        return self.column_name

    def get_column_names(self, columns_seen):
        columns_seen.add(self.column_name)


class Expression(Term):
    def __init__(self, op, args, *, params=None, inline=False, method=False):
        if not isinstance(op, str):
            raise TypeError("op is supposed to be a string")
        if inline:
            if method:
                raise ValueError("can't set both inline and method")
            if len(args) > 2:
                raise ValueError(
                    "must have no more than two arguments if inline is True"
                )
        self.op = op
        self.args = [_enc_value(ai) for ai in args]
        self.params = params
        self.inline = inline
        self.method = method
        Term.__init__(self)

    def is_equal(self, other):
        # can't use == as that builds a larger expression
        if not isinstance(other, Expression):
            return False
        if self.op != other.op:
            return False
        if self.inline != other.inline:
            return False
        if self.params is None:
            if other.params is not None:
                return False
        else:
            if set(self.params.keys()) != set(other.params.keys()):
                return False
            for k in self.params.keys():
                if self.params[k] != other.params[k]:
                    return False
        if len(self.args) != len(other.args):
            return False
        for lft, rgt in zip(self.args, other.args):
            if not lft.is_equal(rgt):
                return False
        return True

    def replace_view(self, view):
        new_args = [oi.replace_view(view) for oi in self.args]
        return Expression(
            op=self.op, args=new_args, params=self.params, inline=self.inline
        )

    def get_column_names(self, columns_seen):
        for a in self.args:
            a.get_column_names(columns_seen)

    def evaluate(self, data_frame):
        args = [ai.evaluate(data_frame) for ai in self.args]
        # first check chosen mappings
        try:
            return data_algebra.default_data_model.impl_map[self.op](*args)
        except KeyError:
            pass
        # now see if argument (usually Pandas) can do this
        # doubt we hit in this, as most exposed methods are window methods
        try:
            method = getattr(args[0], self.op)
            if callable(method):
                return method(*args[1: ])
        except AttributeError:
            pass
        # new see if numpy can do this
        try:
            fn = numpy.__dict__[self.op]
            if callable(fn):
                return fn(*args)
        except KeyError:
            pass

    def to_python(self, *, want_inline_parens=False):
        subs = [ai.to_python(want_inline_parens=True) for ai in self.args]
        if len(subs) <= 0:
            return "_" + self.op + "()"
        if len(subs) == 1:
            if self.inline:
                return self.op + self.args[0].to_python(want_inline_parens=True)
            if isinstance(self.args[0], ColumnReference):
                return subs[0] + "." + self.op + "()"
            else:
                return subs[0] + "." + self.op + "()"
        if len(subs) == 2 and self.inline:
            if want_inline_parens:
                return "(" + subs[0] + " " + self.op + " " + subs[1] + ")"
            else:
                return subs[0] + " " + self.op + " " + subs[1]
        if self.method:
            if isinstance(self.args[0], ColumnReference):
                return subs[0] + "." + self.op + "(" + ", ".join(subs[1:]) + ")"
            else:
                return subs[0] + "." + self.op + "(" + ", ".join(subs[1:]) + ")"
        return self.op + "(" + ", ".join(subs) + ")"


# define with def so function has usable __name__
def connected_components(f, g):
    return data_algebra.expr_rep.Expression(op="connected_components", args=[f, g])


# TODO: get rid of user_values
def populate_specials(*, column_defs, destination, user_values=None):
    """populate a dictionary with special values
       column_defs is a dictionary,
         usually formed from a ViewRepresentation.column_map
       destination is a dictionary,
         usually formed from a ViewRepresentation.column_map.copy()
    """

    if not isinstance(column_defs, dict):
        raise TypeError("column_defs should be a dictionary")
    if not isinstance(destination, dict):
        raise TypeError("destination should be a dictionary")
    if user_values is None:
        user_values = {}
    if not isinstance(user_values, dict):
        raise TypeError("user_values should be a dictionary")
    nd = column_defs.copy()
    # more doc!
    # makes these symbols available for parsing step
    # need to enter user functions here or as methods on variables
    # Note: a lot of the re-emitters add back the underbar, allow the non
    # underbar names we see here.  See expr_rep.py Expression to_python.
    destination["_row_number"] = lambda: data_algebra.expr_rep.Expression(
        op="row_number", args=[]
    )
    destination["_ngroup"] = lambda: data_algebra.expr_rep.Expression(
        op="ngroup", args=[]
    )
    destination["_size"] = lambda: data_algebra.expr_rep.Expression(op="size", args=[])
    destination["connected_components"] = connected_components


def standardize_join_type(join_str):
    if not isinstance(join_str, str):
        raise TypeError("Expected join_str to be a string")
    join_str = join_str.upper()
    allowed = set(['INNER', 'LEFT', 'RIGHT', 'OUTER', 'FULL', 'CROSS'])
    if not join_str in allowed:  # TOOO put this back in!
        raise KeyError(f"join type {join_str} not supported")
    return join_str


def get_columns_used(parsed_exprs):
    if not isinstance(parsed_exprs, dict):
        raise TypeError(
            "expected parsed_exprs to be a dictionary of data_algebra.expr_rep.Term(s)"
        )
    columns_seen = set()
    for node in parsed_exprs.values():
        node.get_column_names(columns_seen)
    return columns_seen


def implies_windowed(parsed_exprs):
    if not isinstance(parsed_exprs, dict):
        raise TypeError(
            "expected parsed_exprs to be a dictionary of data_algebra.expr_rep.Term(s)"
        )
    for opk in parsed_exprs.values():  # look for aggregation functions
        if isinstance(opk, data_algebra.expr_rep.Expression):
            if opk.op in data_algebra.expr_rep.fn_names_that_imply_windowed_situation:
                return True
    return False
