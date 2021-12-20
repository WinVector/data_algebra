
"""
Adapt data_algebra to SQLite database.
"""

import math
import copy
import numpy
import numbers

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


SQLite_formatters = {
    "is_bad": _sqlite_is_bad_expr,
}


def _check_scalar_bad(x):
    """
    Return 1 if scalar value is none or nan, else 0.
    """

    if x is None:
        return 1
    if not isinstance(x, numbers.Number):
        return 0
    if numpy.isinf(x) or numpy.isnan(x):
        return 1
    return 0


class MedianAgg:
    """
    Aggregate as median. SQLite user class.
    """

    def __init__(self):
        self.collection = []

    def step(self, value):
        """
        Observe value
        """
        self.collection.append(value)

    def finalize(self):
        """
        Return result.
        """
        return numpy.median(self.collection)


class SQLiteModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SQLite"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=SQLite_formatters,
            on_joiner="AND",
            union_all_term_start="",
            union_all_term_end="",
        )

    def _unquote_identifier(self, s: str) -> str:
        # good enough
        assert s.startswith(self.identifier_quote)
        assert s.endswith(self.identifier_quote)
        res = s[1:(len(s) - 1)]
        assert self.identifier_quote not in res
        return res

    def prepare_connection(self, conn):
        """
        Insert user functions into db.
        """
        # https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.create_function
        conn.create_function("is_bad", 1, _check_scalar_bad)
        saw = set()
        # math fns
        math_fns = {
            "acos": math.acos,
            "acosh": math.acosh,
            "asin": math.asin,
            "asinh": math.asinh,
            "atan": math.atan,
            "atanh": math.atanh,
            "ceil": math.ceil,
            "ceiling": math.ceil,
            "cos": math.cos,
            "cosh": math.cosh,
            "degrees": math.degrees,
            "erf": math.erf,
            "erfc": math.erfc,
            "exp": math.exp,
            "expm1": math.expm1,
            "fabs": math.fabs,
            "factorial": math.factorial,
            "floor": math.floor,
            "frexp": math.frexp,
            "gamma": math.gamma,
            "isfinite": math.isfinite,
            "isinf": math.isinf,
            "isnan": math.isnan,
            "lgamma": math.lgamma,
            "log": math.log,
            "log10": math.log10,
            "log1p": math.log1p,
            "log2": math.log2,
            "modf": math.modf,
            "radians": math.radians,
            "sign": numpy.sign,
            "sin": math.sin,
            "sinh": math.sinh,
            "sqrt": math.sqrt,
            "tan": math.tan,
            "tanh": math.tanh,
            "trunc": math.trunc,
        }
        for k, f in math_fns.items():
            if k not in saw:
                conn.create_function(k, 1, f)
                saw.add(k)
        math_fns_2 = {
            "atan2": math.atan2,
            "copysign": math.copysign,
            "fmod": math.fmod,
            "gcd": math.gcd,
            "hypot": math.hypot,
            "isclose": math.isclose,
            "ldexp": math.ldexp,
            "pow": math.pow,
            "power": math.pow,
        }
        for k, f in math_fns_2.items():
            if k not in saw:
                conn.create_function(k, 2, f)
                saw.add(k)
        # numpy fns
        numpy_fns = {
            "abs": numpy.abs,
            "arccos": numpy.arccos,
            "arccosh": numpy.arccosh,
            "arcsin": numpy.arcsin,
            "arcsinh": numpy.arcsinh,
            "arctan": numpy.arctan,
            "arctanh": numpy.arctanh,
            "ceil": numpy.ceil,
            "cos": numpy.cos,
            "cosh": numpy.cosh,
            "exp": numpy.exp,
            "expm1": numpy.expm1,
            "floor": numpy.floor,
            "log": numpy.log,
            "log10": numpy.log10,
            "log1p": numpy.log1p,
            "sin": numpy.sin,
            "sinh": numpy.sinh,
            "sqrt": numpy.sqrt,
            "tanh": numpy.tanh,
        }
        for k, f in numpy_fns.items():
            if k not in saw:
                conn.create_function(k, 1, f)
                saw.add(k)

        # https://docs.python.org/3/library/sqlite3.html
        conn.create_aggregate("median", 1, MedianAgg)

    # noinspection PyMethodMayBeStatic,SqlNoDataSourceInspection
    def insert_table(
        self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False
    ):
        """

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :param qualifiers: schema and such
        :param allow_overwrite logical, if True drop previous table
        """

        if qualifiers is not None:
            raise ValueError("non-empty qualifiers not yet supported on insert")
        cur = conn.cursor()
        # check for table
        table_exists = True
        # noinspection PyBroadException
        try:
            self.read_query(conn, "SELECT * FROM " + table_name + " LIMIT 1")
        except Exception:
            table_exists = False
        if table_exists:
            if not allow_overwrite:
                raise ValueError("table " + table_name + " already exists")
            else:
                cur.execute("DROP TABLE " + table_name)
        # Note: the Pandas to_sql() method appears to have SQLite hard-wired into it
        # it refers to sqlite_master
        d.to_sql(name=table_name, con=conn, index=False)

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
        assert len(join_node.by) > 0  # could special case zero case later
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = set(join_node.column_names)
        join_columns = join_node.by
        left_descr = join_node.sources[0]
        right_descr = join_node.sources[1]
        ops_simulate = (
            # get shared key set
            left_descr.project({}, group_by=join_columns)
            .concat_rows(
                b=right_descr.project({}, group_by=join_columns), id_column=None,
            )
            .project({}, group_by=join_columns)
            # simulate full join with left joins
            .natural_join(b=left_descr, by=join_columns, jointype="left")
            .natural_join(b=right_descr, by=join_columns, jointype="left")
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


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    db_handle = SQLiteModel().db_handle(conn=sqlite3.connect(":memory:"))
    db_handle.db_model.prepare_connection(db_handle.conn)
    return db_handle
