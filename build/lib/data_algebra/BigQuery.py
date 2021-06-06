
import gzip

import data_algebra
import data_algebra.data_ops
import data_algebra.db_model
import data_algebra.bigquery_user_fns

_have_bigquery = False
try:
    # noinspection PyUnresolvedReferences
    import google.cloud.bigquery

    _have_bigquery = True
except ImportError:
    pass


def _bigquery_nunique_expr(dbmodel, expression):
    return (
        "COUNT(DISTINCT(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + "))"
    )


def _bigquery_median_expr(dbmodel, expression):
    return (
        "PERCENTILE_CONT(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ", 0.5)"
    )


def _bigquery_mean_expr(dbmodel, expression):
    return (
        "avg(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _bigquery_size_expr(dbmodel, expression):
    return "SUM(1)"


BigQuery_formatters = {
    "nunique": _bigquery_nunique_expr,
    "median": _bigquery_median_expr,
    "mean": _bigquery_mean_expr,
    "size": _bigquery_size_expr,
}


class BigQueryModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for BigQuery
       connection should be google.cloud.bigquery.client.Client"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='`',
            string_quote='"',
            sql_formatters=BigQuery_formatters,
            on_start='(',
            on_end=')',
            on_joiner=' AND ',
            string_type='STRING',
        )

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            # TODO: escape quotes
            raise ValueError('did not expect ' + self.identifier_quote + ' in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

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
        r.reset_index(drop=True, inplace=True)
        return r.copy()  # fresh copy

    def db_handle(self, conn):
        return BigQuery_DBHandle(db_model=self, conn=conn)


class BigQuery_DBHandle(data_algebra.db_model.DBHandle):
    def __init__(self, *, db_model=BigQueryModel(), conn, fns=data_algebra.bigquery_user_fns.fns):
        if not isinstance(db_model, BigQueryModel):
            raise TypeError(
                "expected db_model to be of class data_algebra.BigQuery.BigQueryModel"
            )
        data_algebra.db_model.DBHandle.__init__(self, db_model=db_model, conn=conn, fns=fns)

    def describe_bq_table(self, *, table_catalog, table_schema, table_name, row_limit=7):
        full_name = f'{table_catalog}.{table_schema}.{table_name}'
        head = self.db_model.read_query(
            conn=self.conn,
            q="SELECT * FROM "
            + self.db_model.quote_table_name(full_name)
            + " LIMIT "
            + str(row_limit),
        )
        cat_name = f'{table_catalog}.{table_schema}.INFORMATION_SCHEMA.COLUMNS'
        sql_meta = self.db_model.read_query(
            self.conn,
            f'SELECT * FROM {self.db_model.quote_table_name(cat_name)} ' +
                f'WHERE table_name={self.db_model.quote_string(table_name)}')
        qualifiers = {
            'table_catalog': table_catalog,
            'table_schema': table_schema,
            'table_name': table_name,
            'full_name': full_name,
        }
        td = data_algebra.data_ops.describe_table(
            head,
            table_name=full_name,
            row_limit=row_limit,
            qualifiers=qualifiers,
            sql_meta=sql_meta,
        )
        return td

    def query_to_csv(self, q, *, res_name):
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(self.db_model)
        else:
            q = str(q)
        op = lambda: open(res_name, 'w')
        if res_name.endswith('.gz'):
            op = lambda: gzip.open(res_name, 'w')
        with op() as res:
            res_iter = self.conn.query(q).result().to_dataframe_iterable()
            is_first = True
            for block in res_iter:
                block.to_csv(res, index=False, header=is_first)
                is_first = False

    def drop_table(self, table_name):
        self.execute(f'DROP TABLE IF EXISTS `{table_name}`')

    def insert_table(self, d, *, table_name, allow_overwrite=False):
        if allow_overwrite:
            self.drop_table(table_name)
        job = self.conn.load_table_from_dataframe(
            d,
            table_name)
        job.result()
