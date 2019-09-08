import math
import pandas
import numpy
import numbers

import data_algebra.util
import data_algebra.data_ops
import data_algebra.db_model


# map from op-name to special SQL formatting code

def _sqlite_is_bad_expr(dbmodel, expression):
    return "is_bad(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"


SQLite_formatters = {
    "is_bad": _sqlite_is_bad_expr,
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
        conn.create_function("exp", 1, math.exp)
        conn.create_function("is_bad", 1, _check_scalar_bad)

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise Exception("expected identifier to be a str")
        if self.identifier_quote in identifier:
            raise Exception('did not expect " in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def quote_table_name(self, table_description):
        if not isinstance(table_description, data_algebra.data_ops.TableDescription):
            raise Exception(
                "Expected table_description to be a data_algebra.data_ops.TableDescription)"
            )
        if len(table_description.qualifiers) > 0:
            raise Exception("SQLite adapter does not currently support qualifiers")
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

        if not isinstance(d, pandas.DataFrame):
            raise Exception("d is supposed to be a pandas.DataFrame")
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS " + table_name)
        d.to_sql(name=table_name, con=conn)
        return data_algebra.data_ops.TableDescription(
            table_name=table_name, column_names=[c for c in d.columns]
        )
