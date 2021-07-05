import data_algebra.data_ops
import data_algebra.db_model


have_sqlalchemy = False
try:
    # noinspection PyUnresolvedReferences
    import sqlalchemy

    have_sqlalchemy = True
except ImportError:
    have_sqlalchemy = False


def _MySQL_mean_expr(dbmodel, expression):
    return (
        "avg(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _MySQL_size_expr(dbmodel, expression):
    return "SUM(1)"

def _MySQL_is_bad_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return (
        "("
        + subexpr
        + " IS NULL "   # TODO get infinity checks here
        + " OR ("
        + subexpr
        + " != 0 AND "
        + subexpr
        + " = -"
        + subexpr
        + "))"
    )

def _MySQL_concat_expr(dbmodel, expression):
    return (
        "CONCAT("  # TODO: cast each to char on way in
        + ", ".join([dbmodel.expr_to_sql(ai, want_inline_parens=False) for ai in expression.args])
        + ")"
    )

# map from op-name to special SQL formatting code
MySQL_formatters = {
    "___": lambda dbmodel, expression: expression.to_python(),
    "mean": _MySQL_mean_expr,
    "size": _MySQL_size_expr,
    "is_bad": _MySQL_is_bad_expr,
    'concat': _MySQL_concat_expr,
}

class MySQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for MySQL.
       Assuming we are using a sqlalhemy engine as our connection
    """

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            string_type='CHAR',
            identifier_quote='`',
            string_quote="'",
            sql_formatters=MySQL_formatters,
        )

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            # TODO: escape quotes
            raise ValueError('did not expect ' + self.identifier_quote + ' in identifier')
        return self.identifier_quote + identifier + self.identifier_quote


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    # TODO: parameterize this
    if not have_sqlalchemy:
        return None
    db_handle = MySQLModel().db_handle(
        sqlalchemy.engine.create_engine("mysql+pymysql://jmount@localhost/jmount")
    )
    db_handle.db_model.prepare_connection(db_handle.conn)
    return db_handle
