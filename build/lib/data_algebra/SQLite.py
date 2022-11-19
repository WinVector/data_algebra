"""
Adapt data_algebra to SQLite database.
"""

import functools
import math
import copy
import numpy
import numbers
import warnings
import abc

import sqlite3

import data_algebra.util
import data_algebra.db_model
import data_algebra.data_ops

import data_algebra.near_sql
from data_algebra.data_ops import *


# map from op-name to special SQL formatting code

# Standard SQL code for checking isbad doesn't work in SQLlite, so
# at least capture to is_bad, which appears to not be implemented
# unless we call prepare connection
def _sqlite_is_bad_expr(dbmodel, expression):
    """
    Return SQL to check for bad values.
    """

    return (
        "is_bad("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _sqlite_is_nan_expr(dbmodel, expression):
    """
    Return SQL to check for nan values.
    """

    return (
        "is_nan("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _sqlite_is_inf_expr(dbmodel, expression):
    """
    Return SQL to check for nan values.
    """

    return (
        "is_inf("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _sqlite_RAND_expr(dbmodel, expression):
    """
    Return independent uniform numbers in the range [0, 1]
    # ref: https://www.sqlite.org/lang_corefunc.html#random
    """

    return "(1.0 + random() / 9223372036854775808.0) / 2.0"


def _sqlite_remainder_expr(dbmodel, expression):
    """
    Return SQL remainder.
    """

    return (
        "("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
        + " % "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
        + ")"
    )

def _sqlite_logical_or_expr(dbmodel, expression):
    """
    Return SQL or.
    """

    return (
        "ANY(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _sqlite_logical_and_expr(dbmodel, expression):
    """
    Return SQL and.
    """

    return (
        "ALL(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


SQLite_formatters = {
    "is_bad": _sqlite_is_bad_expr,
    "is_nan": _sqlite_is_nan_expr,
    "is_inf": _sqlite_is_inf_expr,
    "rand": _sqlite_RAND_expr,
    "remainder": _sqlite_remainder_expr,
    "%": _sqlite_remainder_expr,
    "mod": _sqlite_remainder_expr,
    "logical_or": _sqlite_logical_or_expr,
    "logical_and": _sqlite_logical_and_expr,
}


def _check_scalar_bad(x):
    """
    Return True if scalar value is none or nan, else False.
    """

    if x is None:
        return True
    if not isinstance(x, numbers.Number):
        return False
    if numpy.isinf(x) or numpy.isnan(x):
        return True
    return False


def _check_scalar_nan(x):
    """
    Return True if scalar value is nan, else False.
    """

    if x is None:
        return True  # sqlite can't tell this from nan
    if not isinstance(x, numbers.Number):
        return False
    if numpy.isnan(x):
        return True
    return False


def _check_scalar_inf(x):
    """
    Return True if scalar value is inf, else False.
    """

    if x is None:
        return False
    if not isinstance(x, numbers.Number):
        return False
    if numpy.isinf(x):
        return True
    return False


def _sign_fn(x):
    # noinspection PyBroadException
    try:
        if _check_scalar_bad(x):
            return numpy.nan
        if x > 0:
            return 1.0
        if x < 0:
            return -1.0
        return 0.0
    except Exception:
        return numpy.nan


def _abs_fn(x):
    if _check_scalar_bad(x):
        return numpy.nan
    if x >= 0:
        return x
    return -x


def _wrap_scalar_fn(f, x):
    if _check_scalar_bad(x):
        return numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(x)


def _wrap_scalar_fn2(f, x, y):
    if _check_scalar_bad(x) or _check_scalar_bad(y):
        return numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(x, y)


def _wrap_numpy_fn(f, x):
    if _check_scalar_bad(x):
        return numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f([x])[0]


class CollectingAgg(abc.ABC):
    """
    Aggregate from a collection. SQLite user class.
    """

    def __init__(self):
        self.collection = []

    def step(self, value):
        """
        Observe value
        """
        if not _check_scalar_bad(value):
            self.collection.append(value)

    @abc.abstractmethod
    def calc(self) -> float:
        """
        Perform the calculation (only called with non-trivial data)
        """

    def finalize(self):
        """
        Return result.
        """
        if len(self.collection) < 1:
            res = numpy.nan
        else:
            res = self.calc()
        self.collection = []
        return res


class MedianAgg(CollectingAgg):
    """
    Aggregate as median. SQLite user class.
    """

    def __init__(self):
        CollectingAgg.__init__(self)

    def calc(self) -> float:
        "do it"
        return float(numpy.median(self.collection))


class SampVarDevAgg(CollectingAgg):
    """
    Aggregate as sample standard deviation. SQLite user class.
    This version keeps the data instead of using the E[(x-E[x])^2] = E[x^2] - E[x]^2 formula
    """

    def __init__(self):
        CollectingAgg.__init__(self)

    def calc(self) -> float:
        "do it"
        # pandas.DataFrame({'x': [1., 3.], 'g': ['a', 'a']}).groupby(['g']).transform('var')['x']
        # [2, 2]  # sample variance, what we want
        # numpy.var([[1., 3.]])
        # 1.0  # population variance- not what we want
        n = len(self.collection)
        if n < 2:
            return numpy.nan
        return float(numpy.var(self.collection) * n / (n - 1))


class SampStdDevAgg(CollectingAgg):
    """
    Aggregate as sample standard deviation. SQLite user class.
    This version keeps the data instead of using the E[(x-E[x])^2] = E[x^2] - E[x]^2 formula
    """

    def __init__(self):
        CollectingAgg.__init__(self)

    def calc(self) -> float:
        "do it"
        # pandas.DataFrame({'x': [1., 3.], 'g': ['a', 'a']}).groupby(['g']).transform('std')['x']
        # [1.4124214, 1.4124214]  # sample std deviation, what we want
        # numpy.std([[1., 3.]])
        # 1.0  # population std deviation- not what we want
        n = len(self.collection)
        if n < 2:
            return numpy.nan
        return float(numpy.std(self.collection) * numpy.sqrt(n / (n - 1)))


class SQLiteModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SQLite"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=SQLite_formatters,
            supports_cte_elim=False,
            on_joiner="AND",
            union_all_term_start="",
            union_all_term_end="",
        )

    def _unquote_identifier(self, s: str) -> str:
        # good enough
        assert s.startswith(self.identifier_quote)
        assert s.endswith(self.identifier_quote)
        res = s[1 : (len(s) - 1)]
        assert self.identifier_quote not in res
        return res

    def prepare_connection(self, conn):
        """
        Insert user functions into db.
        """
        # # https://stackoverflow.com/questions/52416482/load-sqlite3-extension-in-python3-sqlite
        # conn.enable_load_extension(True)
        # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.create_function
        conn.create_function("is_bad", 1, _check_scalar_bad)
        conn.create_function("is_nan", 1, _check_scalar_nan)
        conn.create_function("is_inf", 1, _check_scalar_inf)
        saw = {"is_bad", "is_nan", "is_inf"}
        # math fns
        math_fns = {
            "acos": functools.partial(_wrap_scalar_fn, math.acos),
            "acosh": functools.partial(_wrap_scalar_fn, math.acosh),
            "asin": functools.partial(_wrap_scalar_fn, math.asin),
            "asinh": functools.partial(_wrap_scalar_fn, math.asinh),
            "atan": functools.partial(_wrap_scalar_fn, math.atan),
            "atanh": functools.partial(_wrap_scalar_fn, math.atanh),
            "ceil": functools.partial(_wrap_scalar_fn, math.ceil),
            "ceiling": functools.partial(_wrap_scalar_fn, math.ceil),
            "cos": functools.partial(_wrap_scalar_fn, math.cos),
            "cosh": functools.partial(_wrap_scalar_fn, math.cosh),
            "degrees": functools.partial(_wrap_scalar_fn, math.degrees),
            "erf": functools.partial(_wrap_scalar_fn, math.erf),
            "erfc": functools.partial(_wrap_scalar_fn, math.erfc),
            "exp": functools.partial(_wrap_scalar_fn, math.exp),
            "expm1": functools.partial(_wrap_scalar_fn, math.expm1),
            "fabs": functools.partial(_wrap_scalar_fn, math.fabs),
            "factorial": functools.partial(_wrap_scalar_fn, math.factorial),
            "floor": functools.partial(_wrap_scalar_fn, math.floor),
            "frexp": functools.partial(_wrap_scalar_fn, math.frexp),
            "gamma": functools.partial(_wrap_scalar_fn, math.gamma),
            "isfinite": functools.partial(_wrap_scalar_fn, math.isfinite),
            "lgamma": functools.partial(_wrap_scalar_fn, math.lgamma),
            "log": functools.partial(_wrap_scalar_fn, math.log),
            "log10": functools.partial(_wrap_scalar_fn, math.log10),
            "log1p": functools.partial(_wrap_scalar_fn, math.log1p),
            "log2": functools.partial(_wrap_scalar_fn, math.log2),
            "modf": functools.partial(_wrap_scalar_fn, math.modf),
            "radians": functools.partial(_wrap_scalar_fn, math.radians),
            "sign": _sign_fn,
            "abs": _abs_fn,
            "sin": functools.partial(_wrap_scalar_fn, math.sin),
            "sinh": functools.partial(_wrap_scalar_fn, math.sinh),
            "sqrt": functools.partial(_wrap_scalar_fn, math.sqrt),
            "tan": functools.partial(_wrap_scalar_fn, math.tan),
            "tanh": functools.partial(_wrap_scalar_fn, math.tanh),
            "trunc": functools.partial(_wrap_scalar_fn, math.trunc),
        }
        for k, f in math_fns.items():
            if k not in saw:
                conn.create_function(k, 1, f)
                saw.add(k)
        math_fns_2 = {
            "atan2": functools.partial(_wrap_scalar_fn2, math.atan2),
            "copysign": functools.partial(_wrap_scalar_fn2, math.copysign),
            "fmod": functools.partial(_wrap_scalar_fn2, math.fmod),
            "gcd": functools.partial(_wrap_scalar_fn2, math.gcd),
            "hypot": functools.partial(_wrap_scalar_fn2, math.hypot),
            "isclose": functools.partial(_wrap_scalar_fn2, math.isclose),
            "ldexp": functools.partial(_wrap_scalar_fn2, math.ldexp),
            "pow": functools.partial(_wrap_scalar_fn2, math.pow),
            "power": functools.partial(_wrap_scalar_fn2, math.pow),
        }
        for k, f in math_fns_2.items():
            if k not in saw:
                conn.create_function(k, 2, f)
                saw.add(k)
        # numpy fns
        numpy_fns = {
            # string being passed to numpy
            "abs": functools.partial(_wrap_numpy_fn, numpy.abs),
            "arccos": functools.partial(_wrap_numpy_fn, numpy.arccos),
            "arccosh": functools.partial(_wrap_numpy_fn, numpy.arccosh),
            "arcsin": functools.partial(_wrap_numpy_fn, numpy.arcsin),
            "arcsinh": functools.partial(_wrap_numpy_fn, numpy.arcsinh),
            "arctan": functools.partial(_wrap_numpy_fn, numpy.arctan),
            "arctanh": functools.partial(_wrap_numpy_fn, numpy.arctanh),
            "ceil": functools.partial(_wrap_numpy_fn, numpy.ceil),
            "cos": functools.partial(_wrap_numpy_fn, numpy.cos),
            "cosh": functools.partial(_wrap_numpy_fn, numpy.cosh),
            "exp": functools.partial(_wrap_numpy_fn, numpy.exp),
            "expm1": functools.partial(_wrap_numpy_fn, numpy.expm1),
            "floor": functools.partial(_wrap_numpy_fn, numpy.floor),
            "log": functools.partial(_wrap_numpy_fn, numpy.log),
            "log10": functools.partial(_wrap_numpy_fn, numpy.log10),
            "log1p": functools.partial(_wrap_numpy_fn, numpy.log1p),
            "sin": functools.partial(_wrap_numpy_fn, numpy.sin),
            "sinh": functools.partial(_wrap_numpy_fn, numpy.sinh),
            "sqrt": functools.partial(_wrap_numpy_fn, numpy.sqrt),
            "tanh": functools.partial(_wrap_numpy_fn, numpy.tanh),
        }
        for k, f in numpy_fns.items():
            if k not in saw:
                conn.create_function(k, 1, f)
                saw.add(k)

        # https://docs.python.org/3/library/sqlite3.html
        conn.create_aggregate("median", 1, MedianAgg)
        conn.create_aggregate("std", 1, SampStdDevAgg)
        conn.create_aggregate("var", 1, SampVarDevAgg)

    def _emit_right_join_as_left_join(
        self, join_node, *, using=None, temp_id_source, sql_format_options=None
    ):
        assert join_node.node_name == "NaturalJoinNode"
        assert join_node.jointype == "RIGHT"
        # convert to left to avoid SQLite not having a right jone
        join_node_copy_right = copy.copy(join_node)
        join_node_copy_right.jointype = "LEFT"
        join_node_copy_right.sources = [join_node.sources[1], join_node.sources[0]]
        near_sql_right = data_algebra.db_model.DBModel.natural_join_to_near_sql(
            self,
            join_node=join_node_copy_right,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
            left_is_first=False,
        )
        return near_sql_right

    def _emit_full_join_as_complex(
        self, join_node, *, using=None, temp_id_source, sql_format_options=None
    ):
        # this is an example how to tree-rewrite the operator platform before emitting SQL.
        assert join_node.node_name == "NaturalJoinNode"
        assert join_node.jointype == "FULL"
        assert len(join_node.on_a) > 0  # could special case zero case later
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = set(join_node.column_names)
        join_columns = join_node.on_a
        assert join_node.on_a == join_node.on_b  # TODO: relax this
        left_descr = join_node.sources[0]
        right_descr = join_node.sources[1]
        ops_simulate = (
            # get shared key set
            left_descr.project({}, group_by=join_columns)
            .concat_rows(
                b=right_descr.project({}, group_by=join_columns),
                id_column=None,
            )
            .project({}, group_by=join_columns)
            # simulate full join with left joins
            .natural_join(b=left_descr, on=join_columns, jointype="left")
            .natural_join(b=right_descr, on=join_columns, jointype="left")
        )
        assert isinstance(ops_simulate, NaturalJoinNode)
        simulate_near_sql = self.natural_join_to_near_sql(
            join_node=ops_simulate,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )
        return simulate_near_sql

    def natural_join_to_near_sql(
        self,
        join_node,
        *,
        using=None,
        temp_id_source=None,
        sql_format_options=None,
        left_is_first=True
    ):
        """
        Translate a join into SQL, converting right and full joins to replacement code (as SQLite doesn't have these).
        """
        if join_node.node_name != "NaturalJoinNode":
            raise TypeError(
                "Expected join_node to be a data_algebra.data_ops.NaturalJoinNode)"
            )
        assert left_is_first
        if temp_id_source is None:
            temp_id_source = [0]
        if join_node.jointype == "RIGHT":
            return self._emit_right_join_as_left_join(
                join_node,
                using=using,
                temp_id_source=temp_id_source,
                sql_format_options=sql_format_options,
            )
        if join_node.jointype == "FULL":
            return self._emit_full_join_as_complex(
                join_node,
                using=using,
                temp_id_source=temp_id_source,
                sql_format_options=sql_format_options,
            )
        # delegate back to parent class
        return data_algebra.db_model.DBModel.natural_join_to_near_sql(
            self,
            join_node=join_node,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
            left_is_first=left_is_first,
        )


def example_handle() -> data_algebra.db_model.DBHandle:
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    db_handle = SQLiteModel().db_handle(conn=sqlite3.connect(":memory:"))
    db_handle.db_model.prepare_connection(db_handle.conn)
    return db_handle
