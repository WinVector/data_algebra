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

    def build_qualified_table_name(self, table_name, *, qualifiers=None):
        qt = self.quote_identifier(table_name)
        if qualifiers is None:
            qualifiers = {}
        if "schema" in qualifiers.keys():
            qt = self.quote_identifier(qualifiers["schema"]) + "." + qt
        return qt
