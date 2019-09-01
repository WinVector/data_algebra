import pandas

import data_algebra.data_ops
import data_algebra.db_model

# map from op-name to special SQL formatting code
SQLite_formatters = {"___": lambda dbmodel, expression: expression.to_python()}


class SQLiteModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SQLite"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=SQLite_formatters,
        )

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
        cr = [
            d.columns[i].lower()
            + " "
            + (
                "double precision"
                if data_algebra.util.can_convert_v_to_numeric(d[d.columns[i]])
                else "VARCHAR"
            )
            for i in range(d.shape[1])
        ]
        create_stmt = "CREATE TABLE " + table_name + " ( " + ", ".join(cr) + " )"
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS " + table_name)
        d.to_sql(name = table_name, con=conn)
        return data_algebra.data_ops.TableDescription(table_name=table_name, column_names=[c for c in d.columns])
