"""
Represent data processing expressions.
"""

import abc
from typing import Any, List, Optional, Set, Union

import data_algebra.util
from data_algebra.expression_walker import ExpressionWalker
import data_algebra.data_model


# for some ideas in capturing expressions in Python see:
#  scipy
# pipe-like idea
#  http://code.activestate.com/recipes/384122-infix-operators/
#  http://tomerfiliba.com/blog/Infix-Operators/


class PythonText:
    """
    Class for holding text representation of Python, with possible additional annotations.
    str() method returns only the text for interoperability.
    """

    s: str
    is_in_parens: bool

    def __init__(self, s: str, *, is_in_parens: bool = False):
        assert isinstance(s, str)
        assert isinstance(is_in_parens, bool)
        self.s = s
        self.is_in_parens = is_in_parens

    def __str__(self):
        return self.s

    def __repr__(self):
        return self.s.__repr__()


# list of window/aggregation functions that must be windowed/aggregated
# (note some other functions work in more than one mode)
# noinspection SpellCheckingInspection
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
    "_ngroup",
    "nlargest",
    "nsmallest",
    "nth",
    "nunique",
    "ohlc",
    "pct_change",
    "rank",
    "_row_number",
    "row_number",
    "shift",
    "size",
    "std",
    "sum",
    "tail",
    "unique",
    "value_counts",
    "var",
}


# noinspection SpellCheckingInspection
fn_names_that_imply_ordered_windowed_situation = {
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "row_number",
    "shift",
    "lag",
    "lead",
    "_row_number",
    "row_number",
}


# noinspection SpellCheckingInspection
fn_names_not_allowed_in_project = {
    "ngroup",
    "_ngroup",
}.union(fn_names_that_imply_ordered_windowed_situation)


# fns that don't have consistent windowed implementations we want to support
fn_names_that_contradict_windowed_situation = set()


# a competing idea should be to remove ordering if
# operator is one of these (instead of forbid)
# noinspection SpellCheckingInspection
fn_names_that_contradict_ordered_windowed_situation = {
    "count",
    "max",
    "min",
    "prod",
    "sum",
    "std",
    "var",
}


class PreTerm(abc.ABC):
    """
    abstract base class, without combination ability
    """

    source_string: Union[str, None]

    def __init__(self):
        self.source_string = None

    @abc.abstractmethod
    def is_equal(self, other):
        """
        Check if this expression code is the same as another expression.
        """
        # can't use == as that builds a larger expression

    # analysis

    def get_column_names(self, columns_seen: Set[str]) -> None:
        """
        Add column names to columns_seen
        :param columns_seen: set of strings
        :return: None
        """
        pass

    def get_method_names(self, methods_seen: Set[str]) -> None:
        """
        Add method names to methods_seen
        :param methods_seen: set of strings
        :return: None
        """
        pass

    # eval

    @abc.abstractmethod
    def act_on(self, arg, *, expr_walker: ExpressionWalker):
        """
        Apply expression to argument.
        """

    # emitters

    @abc.abstractmethod
    def to_python(self, *, want_inline_parens: bool = False) -> PythonText:
        """
        Convert parsed expression into a string

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText
        """

    def to_source(self, *, want_inline_parens=False, dialect="Python") -> PythonText:
        """
        Convert to source code.

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :param dialect: dialect to emit (not currently used)
        :return: PythonText
        """
        if dialect == "Python":
            return self.to_python(want_inline_parens=want_inline_parens)
        else:
            raise ValueError("unexpected dialect string: " + str(dialect))

    # printing

    def __repr__(self):
        return str(self.to_python(want_inline_parens=False))

    def __str__(self):
        return str(self.to_python(want_inline_parens=False))


def _is_none_value(x):
    if x is None:
        return True
    if isinstance(x, Value):
        return x.value is None
    return False


def _check_expr_incompatible_types(a, b):
    """
    return None if no type problem, else return pair of types
    """

    def obvious_declared_type(v):
        """
        Return type of value, if obvious (usually it is not).
        """
        if v is None:
            return None  # None may be a placeholder for "don't know at this point"
        if isinstance(v, Value):
            return type(v.value)
        if isinstance(v, ColumnReference):
            return None
        if isinstance(v, PreTerm):
            return None
        return None  # dunno

    type_a = obvious_declared_type(a)
    if type_a is None:
        return None
    type_b = obvious_declared_type(b)
    if type_b is None:
        return None
    looks_compatible = data_algebra.util.compatible_types([type_a, type_b])
    if looks_compatible:
        return None
    return type_a, type_b


# noinspection SpellCheckingInspection
class Term(PreTerm, abc.ABC):
    """
    Abstract intermediate class with combination ability
    """

    def __init__(self):
        PreTerm.__init__(self)

    # builders

    def __op_expr__(self, op, other, *, inline=True, method=False, check_types=True):
        """binary expression"""
        assert isinstance(op, str)
        if not isinstance(other, Term):
            other = enc_value(other)
        assert not _is_none_value(self)
        assert not _is_none_value(other)
        if check_types:
            obvious_type_problem = _check_expr_incompatible_types(self, other)
            if obvious_type_problem is not None:
                raise TypeError(
                    f"trying to combine incompatible values:"
                    + f" {self}:{obvious_type_problem[0]}"
                    + f" and {other}:{obvious_type_problem[0]} with {op}"
                )
        return Expression(op, (self, other), inline=inline, method=method)

    def __rop_expr__(self, op, other, *, check_types=True):
        """reversed binary expression"""
        inline = True
        method = False
        assert isinstance(op, str)
        if not isinstance(other, Term):
            other = enc_value(other)
        assert not _is_none_value(self)
        assert not _is_none_value(other)
        if check_types:
            obvious_type_problem = _check_expr_incompatible_types(self, other)
            if obvious_type_problem is not None:
                raise TypeError(
                    f"trying to combine incompatible values:"
                    + f" {other}:{obvious_type_problem[0]}"
                    + f" and {self}:{obvious_type_problem[0]} with {op}"
                )
        return Expression(op, (other, self), inline=inline, method=method)

    def __uop_expr__(self, op, *, params=None, inline=False):
        """unary expression"""
        assert isinstance(op, str)
        assert not _is_none_value(self)
        return Expression(op, (self,), params=params, inline=inline, method=not inline)

    def __triop_expr__(self, op, x, y, inline=False, method=False):
        """three argument expression"""
        assert isinstance(op, str)
        if not isinstance(x, Term):
            x = enc_value(x)
        if not isinstance(y, Term):
            y = enc_value(y)
        assert not _is_none_value(self)
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

    def float_divide(self, other):
        return self.__op_expr__("%/%", other)

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

    # Python operators: https://docs.python.org/3/library/operator.html

    def sign(self):
        """
        Return -1, 0, 1 as sign of item (vectorized).
        """
        return self.__uop_expr__("sign")

    def sin(self):
        """
        Return trigometric sin() (in radians) of item (vectorized).
        """
        return self.__uop_expr__("sin")

    def cos(self):
        """
        Return trigometric cos() (in radians) of item (vectorized).
        """
        return self.__uop_expr__("cos")

    def arcsin(self):
        """
        Return trigometric arcsin() (in radians) of item (vectorized).
        """
        return self.__uop_expr__("arcsin")

    def arccos(self):
        """
        Return trigometric arccos() (in radians) of item (vectorized).
        """
        return self.__uop_expr__("arccos")

    def arctan(self):
        """
        Return trigometric arctan() (in radians) of item (vectorized).
        """
        return self.__uop_expr__("arctan")

    def arctan2(self, other):
        """
        Return trigometric arctan2() (in radians) of item (vectorized).
        """
        return self.__op_expr__("arctan2", other, inline=False, method=True)

    def sinh(self):
        """
        Return hyperbolic sinh() of item (vectorized).
        """
        return self.__uop_expr__("sinh")

    def cosh(self):
        """
        Return hyperbolic cosh() of item (vectorized).
        """
        return self.__uop_expr__("cosh")

    def tanh(self):
        """
        Return hyperbolic tanh() of item (vectorized).
        """
        return self.__uop_expr__("tanh")

    def arcsinh(self):
        """
        Return hyperbolic arcsinh() of item (vectorized).
        """
        return self.__uop_expr__("arcsinh")

    def arccosh(self):
        """
        Return hyperbolic arccosh() of item (vectorized).
        """
        return self.__uop_expr__("arccosh")

    def arctanh(self):
        """
        Return hyperbolic arctanh() of item (vectorized).
        """
        return self.__uop_expr__("arctanh")

    def floor(self):
        """
        Return floor() (largest int no larger than, as real type) of item (vectorized).
        """
        return self.__uop_expr__("floor")

    def ceil(self):
        """
        Return ceil() (smallest int no smaller than, as real type) of item (vectorized).
        """
        return self.__uop_expr__("ceil")

    def sum(self):
        """
        Return sum() of items (vectorized).
        """
        return self.__uop_expr__("sum")

    def cumprod(self):
        """
        Return cumprod() of items (vectorized).
        """
        return self.__uop_expr__("cumprod")

    def cumsum(self):
        """
        Return cumsum() of items (vectorized).
        """
        return self.__uop_expr__("cumsum")

    def exp(self):
        """
        Return exp() of items (vectorized).
        """
        return self.__uop_expr__("exp")

    def expm1(self):
        """
        Return exp() - 1 of items (vectorized).
        """
        return self.__uop_expr__("expm1")

    def log(self):
        """
        Return base e logarithm of items (vectorized).
        """
        return self.__uop_expr__("log")

    def log10(self):
        """
        Return base 10 logarithm of items (vectorized).
        """
        return self.__uop_expr__("log10")

    def log1p(self):
        """
        Return base e logarithm of 1 + items (vectorized).
        """
        return self.__uop_expr__("log1p")

    def mod(self, other):
        """
        Return modulo of items (vectorized).
        """
        return self.__op_expr__("mod", other, inline=False, method=True)

    def remainder(self, other):
        """
        Return remainder of items (vectorized).
        """
        return self.__op_expr__("remainder", other, inline=False, method=True)

    def sqrt(self):
        """
        Return sqrt of items (vectorized).
        """
        return self.__uop_expr__("sqrt")

    def abs(self):
        """
        Return absolute value of items (vectorized).
        """
        return self.__uop_expr__("abs")

    def maximum(self, other):
        """
        Return per row maximum of items and other (propogate missing, vectorized).
        """
        return self.__op_expr__("maximum", other, method=True, inline=False)

    def minimum(self, other):
        """
        Return per row minimum of items and other (propogate missing, vectorized).
        """
        return self.__op_expr__("minimum", other, method=True, inline=False)

    def fmax(self, other):
        """
        Return per row fmax of items and other (ignore missing, vectorized).
        """
        return self.__op_expr__("fmax", other, inline=False)

    def fmin(self, other):
        """
        Return per row fmin of items and other (ignore missing, vectorized).
        """
        return self.__op_expr__("fmin", other, inline=False)

    # more numpy stuff
    def round(self):
        """
        Return rounded values (nearest integer, subject to some rules) as real (vectorized).
        """
        return self.__uop_expr__("round")

    def around(self, other):
        """
        Return rounded values (given numer of decimals) as real (vectorized).
        """
        assert isinstance(other, Value)  # digits control
        return self.__op_expr__("around", other, inline=False)

    # pandas style definitions
    # https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html

    def all(self):
        """
        Return True if all items True (vectorized).
        """
        return self.__uop_expr__("all")

    def any(self):
        """
        Return True if any items True (vectorized).
        """
        return self.__uop_expr__("any")

    def bfill(self):
        """
        Return vector with missing vallues filled (vectorized).
        """
        return self.__uop_expr__("bfill")

    def count(self):
        """
        Return number of non-NA cells (vectorized).
        """
        return self.__uop_expr__("count")

    def cumcount(self):
        """
        Return cumulative number of non-NA cells (vectorized).
        """
        return self.__uop_expr__("cumcount")

    def cummax(self):
        """
        Return cumulative maximum (vectorized).
        """
        return self.__uop_expr__("cummax")

    def cummin(self):
        """
        Return cumulative minimum (vectorized).
        """
        return self.__uop_expr__("cummin")

    def ffill(self):
        """
        Return vector with missing vallues filled (vectorized).
        """
        return self.__uop_expr__("ffill")

    def is_monotonic_decreasing(self):
        """
        Return vector True if monotonic decreasing (vectorized).
        """
        return self.__uop_expr__("is_monotonic_decreasing")

    def is_monotonic_increasing(self):
        """
        Return vector True if monotonic increasing (vectorized).
        """
        return self.__uop_expr__("is_monotonic_increasing")

    def any_value(self):
        """
        Return any_value (vectorized).
        """
        return self.__uop_expr__("any_value")

    def first(self):
        """
        Return first (vectorized).
        """
        return self.__uop_expr__("first")

    def last(self):
        """
        Return last (vectorized).
        """
        return self.__uop_expr__("last")

    def max(self):
        """
        Return max (vectorized).
        """
        return self.__uop_expr__("max")

    def mean(self):
        """
        Return mean (vectorized).
        """
        return self.__uop_expr__("mean")

    def median(self):
        """
        Return median (vectorized).
        """
        return self.__uop_expr__("median")

    def min(self):
        """
        Return min (vectorized).
        """
        return self.__uop_expr__("min")

    def nunique(self):
        """
        Return number of unique items (vectorized).
        """
        return self.__uop_expr__("nunique")

    def rank(self):
        """
        Return item rangings (vectorized).
        """
        return self.__uop_expr__("rank")

    def size(self):
        """
        Return number of items (vectorized).
        """
        return self.__uop_expr__("size")

    def std(self):
        """
        Return sample standard devaition (vectorized).
        """
        return self.__uop_expr__("std")

    def var(self):
        """
        Return sample variance (vectorized).
        """
        return self.__uop_expr__("var")

    # pandas shift

    def shift(self, periods=None):
        """
        Return shifted items (vectorized).
        """
        if periods is None:
            periods = Value(1)
        assert isinstance(periods, Value)
        assert isinstance(periods.value, int)
        if periods.value == 0:
            raise ValueError("0-period shift not allowed")
        return self.__op_expr__("shift", periods, inline=False, method=True)

    # our ad-hoc definitions

    def is_null(self):
        """
        Return which items are null (vectorized).
        """
        return self.__uop_expr__("is_null")

    def is_nan(self):
        """
        Return which items are nan (vectorized).
        """
        return self.__uop_expr__("is_nan")

    def is_inf(self):
        """
        Return which items are inf (vectorized).
        """
        return self.__uop_expr__("is_inf")

    def is_bad(self):
        """
        Return which items in a numeric column are bad (null, None, nan, or infinite) (vectorized).
        """
        return self.__uop_expr__("is_bad")

    def if_else(self, x, y):
        """
        Vectorized selection between two argument vectors.
        if_else(True, 1, 2) > 1, if_else(False, 1, 2) -> 2.
        None propagating behavior if_else(None, 1, 2) -> None.
        """
        # could check if x and y are compatible types
        return self.__triop_expr__("if_else", x, y, method=True)

    def where(self, x, y):
        """
        Vectorized selection between two argument vectors.
        if_else(True, 1, 2) > 1, if_else(False, 1, 2) -> 2.
        numpy.where behavior: where(None, 1, 2) -> 2
        """
        # could check if x and y are compatible types
        return self.__triop_expr__("where", x, y, method=True)

    def is_in(self, x):
        """
        Set membership (vectorized).
        """
        return self.__op_expr__(
            "is_in",
            x,
            inline=False,
            method=True,
            check_types=False,
        )

    def concat(self, x):
        """
        Concatinate strings (vectorized).
        """
        # TODO: see if we can format back to infix notation
        return self.__op_expr__(
            "concat",
            x,
            inline=False,
            method=True,
            check_types=False,
        )

    def coalesce(self, x):
        """
        Replace missing values with alternative (vectorized).
        """
        # TODO: see if we can format back to infix notation
        return self.__op_expr__("coalesce", x, inline=False, method=True)

    def co_equalizer(self, x):
        """
        Compute the connected components (co-equalizer).
        """
        return self.__op_expr__("co_equalizer", x, inline=False, method=True)

    def mapv(self, value_map, default_value=None):
        """
        Map values to values (vectorized).
        """
        assert isinstance(value_map, DictTerm)
        if default_value is None:
            default_value = Value(None)
        assert isinstance(default_value, Value)
        return self.__triop_expr__(
            "mapv", x=value_map, y=default_value, inline=False, method=True
        )

    # additional fns

    def as_int64(self):
        """
        Cast as int (vectorized).
        """
        return self.__uop_expr__("as_int64")

    def as_str(self):
        """
        Cast as string (vectorized).
        """
        return self.__uop_expr__("as_str")

    def trimstr(self, start, stop):
        """
        Trim string start (inclusive) to stop (exclusive) (vectorized).
        """
        assert isinstance(start, Value)
        assert isinstance(stop, Value)
        return self.__triop_expr__(
            "trimstr", x=start, y=stop, inline=False, method=True
        )

    def coalesce_0(self):
        """
        Replace missing values with zero (vectorized).
        """
        return self.coalesce(Value(0))

    def datetime_to_date(self):
        """
        Convert date time to date (vectorized).
        """
        return self.__uop_expr__("datetime_to_date")

    def parse_datetime(self, format=None):
        """
        Parse string as a date time (vectorized).
        """
        if format is None:
            format = Value("%Y-%m-%d %H:%M:%S")
        assert isinstance(format, Value)
        return self.__op_expr__(
            "parse_datetime",
            other=format,
            inline=False,
            method=True,
            check_types=False,
        )

    def parse_date(self, format=None):
        """
        Parse string as a date (vectorized).
        """
        if format is None:
            format = Value("%Y-%m-%d")
        return self.__op_expr__(
            "parse_date",
            other=format,
            inline=False,
            method=True,
            check_types=False,
        )

    def format_datetime(self, format=None):
        """
        Format string as a date time (vectorized).
        """
        if format is None:
            format = Value("%Y-%m-%d %H:%M:%S")
        assert isinstance(format, Value)
        return self.__op_expr__(
            "format_datetime",
            other=format,
            inline=False,
            method=True,
            check_types=False,
        )

    def format_date(self, format=None):
        """
        Format string as a date (vectorized).
        """
        if format is None:
            format = Value("%Y-%m-%d")
        return self.__op_expr__(
            "format_date",
            other=format,
            inline=False,
            method=True,
            check_types=False,
        )

    def dayofweek(self):
        """
        Convert date to date of week (vectorized).
        """
        return self.__uop_expr__("dayofweek")

    def dayofyear(self):
        """
        Convert date to date of year (vectorized).
        """
        return self.__uop_expr__("dayofyear")

    def dayofmonth(self):
        """
        Convert date to day of month (vectorized).
        """
        return self.__uop_expr__("dayofmonth")

    def weekofyear(self):
        """
        Convert date to week of year (vectorized).
        """
        return self.__uop_expr__("weekofyear")

    def month(self):
        """
        Convert date to month (vectorized).
        """
        return self.__uop_expr__("month")

    def quarter(self):
        """
        Convert date to quarter (vectorized).
        """
        return self.__uop_expr__("quarter")

    def year(self):
        """
        Convert date to year (vectorized).
        """
        return self.__uop_expr__("year")

    def timestamp_diff(self, other):
        """
        Compute difference in timestamps in seconds (vectorized).
        """
        return self.__op_expr__("timestamp_diff", other, inline=False, method=True)

    def date_diff(self, other):
        """
        Compute difference in dates in days (vectorized).
        """
        return self.__op_expr__(
            "date_diff",
            other,
            inline=False,
            method=True,
            check_types=False,
        )

    # noinspection PyPep8Naming
    def base_Sunday(self):
        """
        Compute prior Sunday date from date (self for Sundays) (vectorized).
        """
        return self.__uop_expr__("base_Sunday")


def kop_expr(op, args, inline=False, method=False):
    """three argument expression"""
    assert isinstance(op, str)
    args = [(ai if isinstance(ai, Term) else enc_value(ai)) for ai in args]
    return Expression(op, args, inline=inline, method=method)


class Value(Term):
    """
    Class for holding constants.
    """

    def __init__(self, value):
        allowed = {
            data_algebra.util.map_type_to_canonical(t)
            for t in [int, float, str, bool, type(None)]
        }
        disallowed = {data_algebra.util.map_type_to_canonical(type(value))} - allowed
        if len(disallowed) != 0:
            raise TypeError(
                "value type must be one of: "
                + str(allowed)
                + ", saw "
                + str(list(disallowed)[0])
            )
        self.value = value
        Term.__init__(self)

    def is_equal(self, other):
        """
        Check if this expression code is the same as another expression.
        """
        # can't use == as that builds a larger expression
        if not isinstance(other, Value):
            return False
        return self.value == other.value

    def act_on(self, arg, *, expr_walker: ExpressionWalker):
        """
        Apply expression to argument.
        """
        assert isinstance(expr_walker, ExpressionWalker)
        return expr_walker.act_on_literal(value=self.value)

    def to_python(self, *, want_inline_parens: bool = False) -> PythonText:
        """
        Convert parsed expression into a string

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText
        """
        return PythonText(self.value.__repr__(), is_in_parens=False)

    # don't collect -5 as a complex expression
    def __neg__(self):
        return Value(-self.value)


def lit(x):
    """Represent a value"""
    return Value(x)


class ListTerm(PreTerm):
    """
    Class to hold a collection.
    """

    # derived from PreTerm as this is not combinable
    def __init__(self, value):
        assert isinstance(value, (list, tuple))
        self.value = list(value)  # copy and standardize to a list
        PreTerm.__init__(self)

    def is_equal(self, other):
        """
        Check if this expression code is the same as another expression.
        """
        # can't use == as that builds a larger expression
        if not isinstance(other, ListTerm):
            return False
        return self.value == other.value

    def act_on(self, arg, *, expr_walker: ExpressionWalker):
        """
        Apply expression to argument.
        """
        assert isinstance(expr_walker, ExpressionWalker)
        res = [None] * len(self.value)
        for i in range(len(self.value)):
            vi = self.value[i]
            if isinstance(vi, PreTerm):
                vi = vi.act_on(arg, expr_walker=expr_walker)
            else:
                vi = expr_walker.act_on_literal(value=vi)
            res[i] = vi
        return res

    def to_python(self, *, want_inline_parens: bool = False) -> PythonText:
        """
        Convert parsed expression into a string

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText
        """

        def li_to_python(value):
            """convert a list item to Python"""
            try:
                return str(value.to_python(want_inline_parens=False))
            except AttributeError:
                return str(value)  # TODO: check if this should be repr?

        return PythonText(
            "[" + ", ".join([li_to_python(ai) for ai in self.value]) + "]",
            is_in_parens=False,
        )

    def get_column_names(self, columns_seen: Set[str]) -> None:
        """
        Add column names to columns_seen
        :param columns_seen: set of strings
        :return:
        """
        for ti in self.value:
            ti.get_column_names(columns_seen)


class DictTerm(PreTerm):
    """Class for carrying a dictionary or map."""

    # derived from PreTerm as this is not combinable
    # only holds values
    def __init__(self, value):
        assert isinstance(value, dict)
        self.value = value.copy()
        PreTerm.__init__(self)

    def is_equal(self, other):
        """
        Check if this expression code is the same as another expression.
        """
        # can't use == as that builds a larger expression
        if not isinstance(other, DictTerm):
            return False
        return self.value == other.value

    def act_on(self, arg, *, expr_walker: ExpressionWalker):
        """
        Apply expression to argument.
        """
        assert isinstance(expr_walker, ExpressionWalker)
        return expr_walker.act_on_literal(value=self.value.copy())

    def to_python(self, *, want_inline_parens: bool = False) -> PythonText:
        """
        Convert parsed expression into a string

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText
        """

        def li_to_python(value):
            """Convert an item to python"""
            try:
                return str(value.to_python(want_inline_parens=False))
            except AttributeError:
                return value.__repr__()

        terms = [
            li_to_python(k) + ": " + li_to_python(v) for k, v in self.value.items()
        ]
        return PythonText("{" + ", ".join(terms) + "}", is_in_parens=False)


def enc_value(value):
    """
    Encode a value as a PreTerm or derived class.
    """
    if isinstance(value, PreTerm):
        return value
    if callable(value):
        raise ValueError("can't use a callable as an argument")
    if isinstance(value, list):
        return ListTerm(value)
    return Value(value)


class ColumnReference(Term):
    """class to represent referring to a column"""
    column_name: str

    def __init__(self, column_name):
        self.column_name = column_name
        assert isinstance(column_name, str)
        Term.__init__(self)

    def act_on(self, arg, *, expr_walker: ExpressionWalker):
        """
        Apply expression to argument.
        """
        assert isinstance(expr_walker, ExpressionWalker)
        return expr_walker.act_on_column_name(arg=arg, value=self.column_name)

    def is_equal(self, other):
        """
        Check if this expression code is the same as another expression.
        """
        # can't use == as that builds a larger expression
        if not isinstance(other, ColumnReference):
            return False
        return self.column_name == other.column_name

    def to_python(self, *, want_inline_parens: bool = False) -> PythonText:
        """
        Convert parsed expression into a string

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText
        """
        return PythonText(self.column_name, is_in_parens=False)

    def get_column_names(self, columns_seen: Set[str]) -> None:
        """
        Add column names to columns_seen
        :param columns_seen: set of strings
        :return:
        """
        columns_seen.add(self.column_name)


def col(nm: str):
    """represent a column or value"""
    assert isinstance(nm, str)
    return ColumnReference(column_name=nm)


# noinspection SpellCheckingInspection
def _can_find_method_by_name(op):
    assert isinstance(op, str)
    # from populate_specials
    if op in {
        "_count",
        "_row_number",
        "_size",
        "_connected_components",
        "_ngroup",
        "_uniform",
    }:
        return True
    # check user fns
    # first check chosen mappings
    data_model = data_algebra.data_model.default_data_model()  # just use default for checking defs
    try:
        # noinspection PyUnusedLocal
        check_val = data_model.user_fun_map[op]  # for KeyError
        return True
    except KeyError:
        pass
    # check chosen mappings
    try:
        # noinspection PyUnusedLocal
        check_val = data_model.impl_map[op]  # for KeyError
        return True
    except KeyError:
        pass
    # now see if argument (usually Pandas) can do this
    # doubt we hit in this, as most exposed methods are window methods
    try:
        method = getattr(Value(0), op)
        if callable(method):
            return True
    except AttributeError:
        pass
    return False


class Expression(Term):
    """
    Class for carrying an expression.
    """

    def __init__(
        self, op: str, args, *, params=None, inline: bool = False, method: bool = False
    ):
        assert isinstance(op, str)
        if not _can_find_method_by_name(op):
            raise KeyError(f"can't find implementation for function/method {op}")
        if inline:
            if method:
                raise ValueError("can't set both inline and method")
        self.op = op
        self.args = [enc_value(ai) for ai in args]
        self.params = params
        self.inline = inline
        self.method = method
        Term.__init__(self)

    def is_equal(self, other):
        """
        Check if this expression code is the same as another expression.
        """
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

    def get_column_names(self, columns_seen: Set[str]) -> None:
        """
        Add column names to columns_seen
        :param columns_seen: set of strings
        :return:
        """
        for a in self.args:
            a.get_column_names(columns_seen)

    def get_method_names(self, methods_seen: Set[str]) -> None:
        """
        Add names of methods used to methods_seen.

        :param methods_seen: set to collect results
        :return: None
        """
        methods_seen.add(self.op)

    def act_on(self, arg, *, expr_walker: ExpressionWalker):
        """
        Apply expression to argument.
        """
        assert isinstance(expr_walker, ExpressionWalker)
        args = [ai.act_on(arg, expr_walker=expr_walker) for ai in self.args]
        res = expr_walker.act_on_expression(arg=arg, values=args, op=self)
        return res

    def to_python(self, *, want_inline_parens: bool = False) -> PythonText:
        """
        Convert parsed expression into a string

        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText
        """
        n_args = len(self.args)
        if n_args <= 0:
            return PythonText(self.op + "()", is_in_parens=False)
        if n_args == 1:
            sub_0 = self.args[0].to_python(want_inline_parens=False)
            if self.inline:
                if sub_0.is_in_parens:
                    return PythonText(self.op + str(sub_0), is_in_parens=False)
                return PythonText(self.op + "(" + str(sub_0) + ")", is_in_parens=False)
            if self.method:
                if sub_0.is_in_parens or isinstance(self.args[0], ColumnReference):
                    return PythonText(
                        str(sub_0) + "." + self.op + "()", is_in_parens=False
                    )
                return PythonText(
                    "(" + str(sub_0) + ")." + self.op + "()", is_in_parens=False
                )
        if self.inline:
            subs_strs = [str(ai.to_python(want_inline_parens=True)) for ai in self.args]
            result = (" " + self.op + " ").join(subs_strs)
            if want_inline_parens:
                return PythonText("(" + result + ")", is_in_parens=True)
            return PythonText(result, is_in_parens=False)
        subs: List[PythonText] = [
            ai.to_python(want_inline_parens=False) for ai in self.args
        ]
        subs_0 = subs[0]
        subs_strs = [str(si) for si in subs]
        if self.method:
            if subs_0.is_in_parens or isinstance(self.args[0], ColumnReference):
                return PythonText(
                    subs_strs[0] + "." + self.op + "(" + ", ".join(subs_strs[1:]) + ")",
                    is_in_parens=False,
                )
            else:
                return PythonText(
                    "("
                    + subs_strs[0]
                    + ")."
                    + self.op
                    + "("
                    + ", ".join(subs_strs[1:])
                    + ")",
                    is_in_parens=False,
                )
        # treat as fn call
        return PythonText(
            self.op + "(" + ", ".join(subs_strs) + ")", is_in_parens=False
        )


# define with def so function has usable __name__
def connected_components(f, g):
    """
    Compute connected components.
    """
    return data_algebra.expr_rep.Expression(op="connected_components", args=[f, g])


def standardize_join_type(join_str):
    """
    Replace join name with standard name.
    """
    assert isinstance(join_str, str)
    join_str = join_str.upper()
    allowed = {"INNER", "LEFT", "RIGHT", "OUTER", "FULL", "CROSS"}
    if join_str not in allowed:
        raise KeyError(f"join type {join_str} not supported")
    return join_str


# noinspection SpellCheckingInspection
def get_columns_used(parsed_exprs) -> Set[str]:
    """
    Return set of columns used in an expression.
    """
    assert isinstance(parsed_exprs, dict)
    columns_seen: Set[str] = set()
    for node in parsed_exprs.values():
        node.get_column_names(columns_seen)
    return columns_seen


# noinspection SpellCheckingInspection
def implies_windowed(parsed_exprs: dict) -> bool:
    """
    Return true if expression implies a windowed calculation is needed.
    """
    assert isinstance(parsed_exprs, dict)
    for opk in parsed_exprs.values():  # look for aggregation functions
        if isinstance(opk, data_algebra.expr_rep.Expression):
            if opk.op in data_algebra.expr_rep.fn_names_that_imply_windowed_situation:
                return True
    return False
