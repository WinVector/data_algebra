import math
import numpy
import numbers

import data_algebra.data_types
import data_algebra.util
import data_algebra.db_model


# map from op-name to special SQL formatting code


def _sqlite_is_bad_expr(dbmodel, expression):
    return (
        "is_bad("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _sqlite_mean_expr(dbmodel, expression):
    return (
        "avg("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _sqlite_lag_expr(dbmodel, expression):
    return (
        "LAG("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
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
        # math fns
        conn.create_function('acos', 1, math.acos)
        conn.create_function('acosh', 1, math.acosh)
        conn.create_function('asin', 1, math.asin)
        conn.create_function('asinh', 1, math.asinh)
        conn.create_function('atan', 1, math.atan)
        conn.create_function('atanh', 1, math.atanh)
        conn.create_function('ceil', 1, math.ceil)
        conn.create_function('cos', 1, math.cos)
        conn.create_function('cosh', 1, math.cosh)
        conn.create_function('degrees', 1, math.degrees)
        conn.create_function('erf', 1, math.erf)
        conn.create_function('erfc', 1, math.erfc)
        conn.create_function('exp', 1, math.exp)
        conn.create_function('expm1', 1, math.expm1)
        conn.create_function('fabs', 1, math.fabs)
        conn.create_function('factorial', 1, math.factorial)
        conn.create_function('floor', 1, math.floor)
        conn.create_function('frexp', 1, math.frexp)
        conn.create_function('gamma', 1, math.gamma)
        conn.create_function('isfinite', 1, math.isfinite)
        conn.create_function('isinf', 1, math.isinf)
        conn.create_function('isnan', 1, math.isnan)
        conn.create_function('lgamma', 1, math.lgamma)
        conn.create_function('log', 1, math.log)
        conn.create_function('log10', 1, math.log10)
        conn.create_function('log1p', 1, math.log1p)
        conn.create_function('log2', 1, math.log2)
        conn.create_function('modf', 1, math.modf)
        conn.create_function('radians', 1, math.radians)
        conn.create_function('sin', 1, math.sin)
        conn.create_function('sinh', 1, math.sinh)
        conn.create_function('sqrt', 1, math.sqrt)
        conn.create_function('tan', 1, math.tan)
        conn.create_function('tanh', 1, math.tanh)
        conn.create_function('trunc', 1, math.trunc)
        conn.create_function('atan2', 2, math.atan2)
        conn.create_function('copysign', 2, math.copysign)
        conn.create_function('fmod', 2, math.fmod)
        conn.create_function('gcd', 2, math.gcd)
        conn.create_function('hypot', 2, math.hypot)
        conn.create_function('isclose', 2, math.isclose)
        conn.create_function('ldexp', 2, math.ldexp)
        conn.create_function('pow', 2, math.pow)

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            raise ValueError('did not expect " in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def quote_table_name(self, table_description):
        if not isinstance(table_description, data_algebra.data_ops.TableDescription):
            raise TypeError(
                "Expected table_description to be a data_algebra.data_ops.TableDescription)"
            )
        if len(table_description.qualifiers) > 0:
            raise RuntimeError("SQLite adapter does not currently support qualifiers")
        qt = self.quote_identifier(table_description.table_name)
        return qt

    # noinspection PyMethodMayBeStatic,SqlNoDataSourceInspection
    def insert_table(self, conn, d, table_name):
        """

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :return:
        """

        d = data_algebra.data_types.convert_to_pandas_dataframe(d, "d")
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS " + table_name)
        d.to_sql(name=table_name, con=conn)
        return data_algebra.data_ops.TableDescription(
            table_name=table_name, column_names=[c for c in d.columns]
        )
