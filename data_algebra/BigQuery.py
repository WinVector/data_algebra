import data_algebra.data_ops
import data_algebra.db_model


_have_bigquery = False
try:
    # noinspection PyUnresolvedReferences
    import google.cloud.bigquery

    _have_bigquery = True
except ImportError:
    pass


class BigQueryModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for BigQuery
       connection should be google.cloud.bigquery.client.Client"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='`',
            string_quote='"',
        )

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            # TODO: escape quotes
            raise ValueError('did not expect ' + self.identifier_quote + ' in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def build_qualified_table_name(self, table_name, *, qualifiers=None):
        qt = table_name
        if qualifiers is None:
            qualifiers = {}
        if "schema" in qualifiers.keys():
            qt = self.quote_identifier(qualifiers["schema"]) + "." + qt
        return self.quote_identifier(qt)

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
