import data_algebra.data_ops
import data_algebra.db_model


# map from op-name to special SQL formatting code
SparkSQL_formatters = {"___": lambda dbmodel, expression: expression.to_python()}


class SparkSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SparkSQL"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote="`",
            string_quote='"',
            sql_formatters=SparkSQL_formatters,
        )

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            raise ValueError('did not expect " in identifier')
        return self.identifier_quote + identifier.lower() + self.identifier_quote

    # TODO: see if we need this implementation
    # noinspection SqlNoDataSourceInspection
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

        cr = [
            d.columns[i].lower()
            + " "
            + (
                "double precision"
                if self.local_data_model.can_convert_col_to_numeric(d[d.columns[i]])
                else self.string_type
            )
            for i in range(d.shape[1])
        ]
        q_table_name = self.quote_table_name(
            table_name
        )
        cur = conn.cursor()
        # check for table
        table_exists = True
        # noinspection PyBroadException
        try:
            self.read_query(conn, "SELECT * FROM " + q_table_name + " LIMIT 1")
        except Exception:
            table_exists = False
        if table_exists:
            if not allow_overwrite:
                raise ValueError("table " + q_table_name + " already exists")
            else:
                cur.execute(self.drop_text + ' ' + q_table_name)
                conn.commit()
        create_stmt = "CREATE TABLE " + q_table_name + " ( " + ", ".join(cr) + " )"
        cur.execute(create_stmt)
        conn.commit()
        buf = io.StringIO(d.to_csv(index=False, header=False, sep="\t"))
        cur.copy_from(buf, "d", columns=[c for c in d.columns])
        conn.commit()