import data_algebra.data_ops
import data_algebra.db_model


def _postgresql_mean_expr(dbmodel, expression):
    return (
        "avg(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _postgresql_size_expr(dbmodel, expression):
    return "SUM(1)"


# map from op-name to special SQL formatting code
PostgreSQL_formatters = {
    "___": lambda dbmodel, expression: expression.to_python(),
    "mean": _postgresql_mean_expr,
    "size": _postgresql_size_expr,
}

class PostgreSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for PostgreSQL.
       Assuming we are using a sqlalhemy engine as our connection
    """

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=PostgreSQL_formatters,
        )

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            # TODO: escape quotes
            raise ValueError('did not expect ' + self.identifier_quote + ' in identifier')
        return self.identifier_quote + identifier + self.identifier_quote
