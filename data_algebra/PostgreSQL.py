import data_algebra.db_model


# map from op-name to special SQL formatting code
PostgreSQL_formatters = {"___": lambda dbmodel, expression: expression.to_python()}


class PostgreSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for PostgreSQL"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=PostgreSQL_formatters,
        )

    def quote_table_name(self, table_description):
        qt = self.quote_identifier(table_description.table_name)
        ql = table_description.qualifiers
        if "schema" in ql.keys():
            qt = self.quote_identifier(ql["schema"]) + "." + qt
        return qt
