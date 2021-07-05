import data_algebra.data_ops
import data_algebra.db_model


have_sqlalchemy = False
try:
    # noinspection PyUnresolvedReferences
    import sqlalchemy

    have_sqlalchemy = True
except ImportError:
    have_sqlalchemy = False


def _postgresql_mean_expr(dbmodel, expression):
    return (
        "AVG(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
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


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    # TODO: parameterize this
    if not have_sqlalchemy:
        return None
    db_handle = PostgreSQLModel().db_handle(
        sqlalchemy.engine.create_engine(r'postgresql://johnmount@localhost/johnmount')
    )
    db_handle.db_model.prepare_connection(db_handle.conn)
    return db_handle
