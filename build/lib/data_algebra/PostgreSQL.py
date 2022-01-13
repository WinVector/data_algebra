"""
PostgreSQL database adapter for data algebra.
"""

import data_algebra.data_ops
import data_algebra.db_model


have_sqlalchemy = False
try:
    # noinspection PyUnresolvedReferences
    import sqlalchemy

    have_sqlalchemy = True
except ImportError:
    have_sqlalchemy = False


# map from op-name to special SQL formatting code
PostgreSQL_formatters = {
    "___": lambda dbmodel, expression: str(expression.to_python()),
}


class PostgreSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for PostgreSQL.
       Assuming we are using a sqlalchemy engine as our connection
    """

    def __init__(self):
        op_replacements = data_algebra.db_model.db_default_op_replacements.copy()
        op_replacements['log'] = 'LN'
        op_replacements['_uniform'] = 'RANDOM'
        op_replacements['std'] = 'STDDEV_SAMP'
        op_replacements['var'] = 'VAR_SAMP'
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=PostgreSQL_formatters,
            op_replacements=op_replacements,
        )


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    # TODO: parameterize this
    assert have_sqlalchemy
    db_engine = sqlalchemy.engine.create_engine(r"postgresql://johnmount@localhost/johnmount")
    db_handle = PostgreSQLModel().db_handle(conn=db_engine, db_engine=db_engine)
    db_handle.db_model.prepare_connection(db_handle.conn)
    return db_handle
