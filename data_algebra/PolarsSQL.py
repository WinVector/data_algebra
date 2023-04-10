"""
Polars SQL adapter for data algebra.
"""

import data_algebra.sql_model


def _polars_as_int64(dbmodel, expression):
    return (
        "CAST("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " AS BIGINT)"
    )


def _polars_null_divide_expr(dbmodel, expression):
    assert len(expression.args) == 2
    e0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    e1 = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return f"({e0} / NULLIF(1.0 * {e1}, 0))"


# map from op-name to special SQL formatting code
PolarsSQL_formatters = {
    "___": lambda dbmodel, expression: str(expression.to_python()),
    "as_int64": _polars_as_int64,
    "%/%": _polars_null_divide_expr,
}


class PolarsSQLModel(data_algebra.sql_model.SQLModel):
    """
    A model of how SQL should be generated for PolarsSQL.
    Model is just a stand-in for now, as we don't have a good description of Polars SQL dialect yet.
    """

    def __init__(self):
        op_replacements = data_algebra.sql_model.db_default_op_replacements.copy()
        op_replacements["log"] = "LN"
        op_replacements["_uniform"] = "RANDOM"
        op_replacements["std"] = "STDDEV_SAMP"
        op_replacements["var"] = "VAR_SAMP"
        data_algebra.sql_model.SQLModel.__init__(
            self,
            identifier_quote='"',
            string_quote="'",
            sql_formatters=PolarsSQL_formatters,
            op_replacements=op_replacements,
            float_type="DOUBLE PRECISION",
            supports_with=False,
        )
