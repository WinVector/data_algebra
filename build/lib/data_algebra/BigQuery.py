"""
Adapter for Google BigQuery database
"""


from typing import Optional

import gzip
import os
import os.path

import data_algebra
import data_algebra.data_ops
import data_algebra.db_model

_have_bigquery = False
try:
    # noinspection PyUnresolvedReferences
    import google.cloud.bigquery

    _have_bigquery = True
except ImportError:
    pass


def _bigquery_median_expr(dbmodel, expression):
    return (
        "PERCENTILE_CONT("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", 0.5)"
    )


def _bigquery_std_expr(dbmodel, expression):
    return (
        "STDDEV_SAMP("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _bigquery_var_expr(dbmodel, expression):
    return (
        "VAR_SAMP("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _bigquery_is_bad_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return (
        "("
        + subexpr
        + " IS NULL OR "
        + "IS_INF("
        + subexpr
        + ")"
        + " OR ("
        + subexpr
        + " != 0 AND "
        + subexpr
        + " = -"
        + subexpr
        + "))"
    )


def _bigquery_any_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    return f"LOGICAL_OR({subexpr})"


def _bigquery_all_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    return f"LOGICAL_AND({subexpr})"


def _bigquery_any_value_expr(dbmodel, expression):
    return (
        "ANY_VALUE("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _bigquery_ieee_divide_expr(dbmodel, expression):
    # don't issue an error
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/mathematical_functions#ieee_divide
    assert len(expression.args) == 2
    e0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    e1 = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return f"IEEE_DIVIDE({e0}, 1.0 * {e1})"


BigQuery_formatters = {
    "median": _bigquery_median_expr,
    "is_bad": _bigquery_is_bad_expr,
    "std": _bigquery_std_expr,
    "var": _bigquery_var_expr,
    "any": _bigquery_any_expr,
    "all": _bigquery_all_expr,
    "any_value": _bigquery_any_value_expr,
    "%/%": _bigquery_ieee_divide_expr,
}


class BigQueryModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for BigQuery
    connection should be google.cloud.bigquery.client.Client"""

    def __init__(self, *, table_prefix: Optional[str] = None):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote="`",
            string_quote='"',
            sql_formatters=BigQuery_formatters,
            on_start="(",
            on_end=")",
            on_joiner=" AND ",
            string_type="STRING",
        )
        self.table_prefix = table_prefix

    def get_table_name(self, table_description):
        if not isinstance(table_description, str):
            try:
                if table_description.node_name == "TableDescription":
                    table_description = table_description.table_name
                else:
                    raise TypeError(
                        "Expected table_description to be a string or data_algebra.data_ops.TableDescription)"
                    )
            except KeyError:
                raise TypeError(
                    "Expected table_description to be a string or data_algebra.data_ops.TableDescription)"
                )
        if self.table_prefix is not None:
            table_description = self.table_prefix + "." + table_description
        return table_description

    def quote_table_name(self, table_description):
        table_name = self.get_table_name(table_description)
        return self.quote_identifier(table_name)

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        """
        assert _have_bigquery
        assert isinstance(conn, google.cloud.bigquery.client.Client)
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        assert isinstance(q, str)
        conn.query(q).result()

    def read_query(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        :return: query results as table
        """
        assert _have_bigquery
        assert isinstance(conn, google.cloud.bigquery.client.Client)
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        assert isinstance(q, str)
        r = self.local_data_model.pd.DataFrame(conn.query(q).result().to_dataframe())
        self.local_data_model.clean_copy(r)
        return r.copy()  # fresh copy

    def insert_table(
        self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False
    ):
        prepped_table_name = table_name
        if self.table_prefix is not None:
            prepped_table_name = self.table_prefix + "." + table_name
        if allow_overwrite:
            self.drop_table(conn, table_name)
        else:
            table_exists = True
            # noinspection PyBroadException
            try:
                self.read_query(
                    conn,
                    "SELECT * FROM " + self.quote_table_name(table_name) + " LIMIT 1",
                )
            except Exception:
                table_exists = False
            if table_exists:
                raise ValueError("table " + prepped_table_name + " already exists")
        job = conn.load_table_from_dataframe(d, prepped_table_name)
        job.result()

    def db_handle(self, conn, *, db_engine=None):
        return BigQuery_DBHandle(db_model=self, conn=conn)


class BigQuery_DBHandle(data_algebra.db_model.DBHandle):
    def __init__(self, *, db_model=BigQueryModel(), conn):
        assert isinstance(db_model, BigQueryModel)
        data_algebra.db_model.DBHandle.__init__(self, db_model=db_model, conn=conn)

    def describe_bq_table(
        self, *, table_catalog, table_schema, table_name, row_limit=7
    ) -> data_algebra.data_ops.TableDescription:
        full_name = f"{table_catalog}.{table_schema}.{table_name}"
        head = self.db_model.read_query(
            conn=self.conn,
            q="SELECT * FROM "
            + self.db_model.quote_identifier(
                full_name
            )  # don't quote table name: adds more qualifiers
            + " LIMIT "
            + str(row_limit),
        )
        cat_name = f"{table_catalog}.{table_schema}.INFORMATION_SCHEMA.COLUMNS"
        sql_meta = self.db_model.read_query(
            self.conn,
            f"SELECT * FROM {self.db_model.quote_identifier(cat_name)} "
            + f"WHERE table_name={self.db_model.quote_string(table_name)}",
        )
        qualifiers = {
            "table_catalog": table_catalog,
            "table_schema": table_schema,
            "table_name": table_name,
            "full_name": full_name,
        }
        td = data_algebra.data_ops.describe_table(
            head,
            table_name=full_name,
            row_limit=row_limit,
            qualifiers=qualifiers,
            sql_meta=sql_meta,
        )
        return td

    def query_to_csv(self, q, *, res_name) -> None:
        """Write query to csv"""
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(self.db_model)
        else:
            q = str(q)
        if res_name.endswith(".gz"):
            op = lambda: gzip.open(res_name, "w", encoding="utf-8")
        else:
            op = lambda: open(res_name, "w", encoding="utf-8")
        with op() as res:
            res_iter = self.conn.query(q).result().to_dataframe_iterable()
            is_first = True
            for block in res_iter:
                block.to_csv(res, index=False, header=is_first)
                is_first = False


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.
    Note: binds in a data_catalog and data schema prefix. So this handle is specific
    to one database.

    """
    # TODO: parameterize this
    assert _have_bigquery
    credential_file = "/Users/johnmount/big_query/big_query_jm.json"
    # assert os.path.isfile(credential_file)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_file
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  # trigger key error if not present
    # noinspection PyBroadException
    try:
        data_catalog = "data-algebra-test"
        data_schema = "test_1"
        db_handle = BigQueryModel(
            table_prefix=f"{data_catalog}.{data_schema}"
        ).db_handle(google.cloud.bigquery.Client())
        db_handle.db_model.prepare_connection(db_handle.conn)
        return db_handle
    except Exception:
        return None
