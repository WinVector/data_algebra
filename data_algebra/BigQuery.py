
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
        assert isinstance(q, str)
        r = self.local_data_model.pd.DataFrame(conn.query(q).result().to_dataframe())
        r = r.reset_index(drop=True)
        return r

    def db_handle(self, conn):
        return BigQuery_DBHandle(db_model=self, conn=conn)


# convert datetime to date
def AS_INT64(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.astype('int64'),  # x is a pandas Series
        args=col,
        name='AS_INT64',
        sql_name='CAST',
        sql_suffix=' AS INT64'
    )


# trim string to date portion
def TRIMSTR(*, start=0, stop):
    assert isinstance(start, int)
    assert isinstance(stop, int)
    def f(col_name):
        assert isinstance(col_name, str)
        return data_algebra.data_ops.user_fn(
            lambda x: x.str.slice(start=start, stop=stop),  # x is a pandas Series
            args=col_name,
            name=f'TRIMSTR_{start+1}_{stop}',
            sql_name='SUBSTR', sql_suffix=f', {start+1}, {stop}')
    return f


# convert datetime to date
def DATE(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.dt.date.copy(),  # x is a pandas Series
        args=col,
        name='DATE',
        sql_name='DATE')


# convert str to date
def PARSE_DATE(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        lambda x: data_algebra.default_data_model.pd.to_datetime(x, format="%Y-%m-%d").dt.date.copy(),  # x is a pandas Series
        args=col,
        name='PARSE_DATE',
        sql_name='PARSE_DATE',  # https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions
        sql_prefix='"%Y-%m-%d", ')


# replace missing with zeros
def COALESCE_0(col):
    assert isinstance(col, str)
    return data_algebra.data_ops.user_fn(
        lambda x: x.fillna(0),
        args=col,
        name='COALESCE_0',
        sql_name='COALESCE',
        sql_suffix=', 0')


class BigQuery_DBHandle(data_algebra.db_model.DBHandle):
    def __init__(self, *, db_model, conn):
        if not isinstance(db_model, BigQueryModel):
            raise TypeError(
                "expected db_model to be of class data_algebra.BigQuery.BigQueryModel"
            )
        data_algebra.db_model.DBHandle.__init__(self, db_model=db_model, conn=conn)

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
