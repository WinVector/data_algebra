import math
import numpy
import numbers

import data_algebra.util
import data_algebra.db_model
import data_algebra.data_ops
import data_algebra.eval_model


# map from op-name to special SQL formatting code


def _sqlite_is_bad_expr(dbmodel, expression):
    return (
        "is_bad("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _sqlite_mean_expr(dbmodel, expression):
    return (
        "avg(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _sqlite_lag_expr(dbmodel, expression):
    return (
        "LAG(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


# noinspection PyUnusedLocal
def _sqlite_size_expr(dbmodel, expression):
    return "SUM(1)"


SQLite_formatters = {
    "is_bad": _sqlite_is_bad_expr,
    "mean": _sqlite_mean_expr,
    "shift": _sqlite_lag_expr,
    "size": _sqlite_size_expr,
}


def _check_scalar_bad(x):
    if x is None:
        return 1
    if not isinstance(x, numbers.Number):
        return 0
    if numpy.isinf(x) or numpy.isnan(x):
        return 1
    return 0


class SQLiteModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SQLite"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=SQLite_formatters,
        )

    def prepare_connection(self, conn):
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
            "sin": math.sin,
            "sinh": math.sinh,
            "sqrt": math.sqrt,
            "tan": math.tan,
            "tanh": math.tanh,
            "trunc": math.trunc,
        }
        for k, f in math_fns.items():
            if not k in saw:
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
        }
        for k, f in math_fns_2.items():
            if not k in saw:
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
            if not k in saw:
                conn.create_function(k, 1, f)
                saw.add(k)

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            raise ValueError('did not expect " in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def quote_table_name(self, table_description):
        if isinstance(table_description, str):
            return self.quote_identifier(table_description)
        if not isinstance(table_description, data_algebra.data_ops.TableDescription):
            raise TypeError(
                "Expected table_description to be a data_algebra.data_ops.TableDescription)"
            )
        if len(table_description.qualifiers) > 0:
            raise RuntimeError("SQLite adapter does not currently support qualifiers")
        qt = self.quote_identifier(table_description.table_name)
        return qt

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
        d.to_sql(name=table_name, con=conn)
