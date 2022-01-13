
"""
SparkSQL adapter for the data algebra.
"""

import data_algebra.data_ops
import data_algebra.db_model


have_Spark = False
try:
    # noinspection PyUnresolvedReferences
    import pyspark
    import pyspark.sql

    have_Spark = True
except ImportError:
    pass


def _sparksql_is_bad_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    assert isinstance(subexpr, str)
    return (
        "("
        + subexpr
        + " IS NULL OR "
        + subexpr
        + " >= "
        + dbmodel.value_to_sql("+infinity")
        + " OR "
        + subexpr
        + " <= "
        + dbmodel.value_to_sql("-infinity")
        + " OR "
        + " isNaN("
        + subexpr
        + ")"
        + ")"
    )


# treat NaN as NULL, as Pandas has a hard time distinguishing the two
def _sparksql_coalesce_expr(dbmodel, expression) -> str:
    """
    Return coalesce expression.
    """
    def coalesce_step(x: str) -> str:
        """
        Return one caes of coalesce.
        """
        assert isinstance(x, str)
        return f" WHEN ({x} IS NOT NULL) AND (NOT isNaN({x})) THEN {x} "

    return (
        "CASE "
        + " ".join(
            [
                coalesce_step(dbmodel.expr_to_sql(ai, want_inline_parens=False))
                for ai in expression.args
            ]
        )
        + " ELSE NULL END"
    )


def _sparksql_db_mapv(dbmodel, expression):
    # https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-case.html
    if_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    mapping_dict = expression.args[1]
    default_value_expr = dbmodel.expr_to_sql(
        expression.args[2], want_inline_parens=True
    )
    terms = [
        "WHEN ("
        + if_expr
        + " = "
        + dbmodel.value_to_sql(k)
        + ") THEN "
        + dbmodel.value_to_sql(v)
        for k, v in mapping_dict.value.items()
    ]
    if len(terms) <= 0:
        return default_value_expr
    res = "CASE " + " ".join(terms) + " ELSE " + default_value_expr + " END"
    if isinstance(expression.args[2].value, float):
        res = "DOUBLE(" + res + ")"
    elif isinstance(expression.args[2].value, int):
        res = "INT(" + res + ")"
    return res


def _spark_var_expr(dbmodel, expression):
    return (
        "VAR_SAMP("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _spark_std_expr(dbmodel, expression):
    return (
        "STDDEV_SAMP("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


# map from op-name to special SQL formatting code
SparkSQL_formatters = {
    "___": lambda dbmodel, expression: str(expression.to_python()),
    "is_bad": _sparksql_is_bad_expr,
    "coalesce": _sparksql_coalesce_expr,
    "mapv": _sparksql_db_mapv,
    'var': _spark_var_expr,
    'std': _spark_std_expr,
}


class SparkConnection:
    """
    Holder for spark conext and session as a connection (defines close).
    """
    def __init__(self, *, spark_context, spark_session):
        assert have_Spark
        assert isinstance(spark_context, pyspark.context.SparkContext)
        assert isinstance(spark_session, pyspark.sql.session.SparkSession)
        self.spark_context = spark_context
        self.spark_session = spark_session

    def close(self):
        """
        Stop context and release reference to context and session.
        """
        if self.spark_context is not None:
            self.spark_context.stop()  # probably only for local demos
            self.spark_context = None
        if self.spark_session is not None:
            self.spark_session = None


class SparkSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SparkSQL.

    Known issue: doesn't coalesce NaN
    """

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote="`",
            string_quote='"',
            string_type="STRING",
            sql_formatters=SparkSQL_formatters,
        )

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        """
        Execute a SQL query or operator dag.
        """
        assert isinstance(conn, SparkConnection)
        assert isinstance(q, str)
        conn.spark_session.sql(q)

    def read_query(self, conn, q):
        """
        Execute a SQL query or operator dag, return result as Pandas data frame.
        """
        assert isinstance(conn, SparkConnection)
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        res = conn.spark_session.sql(q)
        # or res.collect()
        return res.toPandas()  # TODO: make sure it is our dataframe type

    # noinspection PyMethodMayBeStatic
    def insert_table(
        self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False
    ):
        """Insert table into database."""
        assert isinstance(conn, SparkConnection)
        assert isinstance(table_name, str)
        if qualifiers is not None:
            raise ValueError("non-empty qualifiers not yet supported on insert")
        if self.table_exists(conn, table_name):
            if not allow_overwrite:
                raise ValueError("table " + table_name + " already exists")
            else:
                self.drop_table(conn, table_name, check=False)
        try:
            d_spark = conn.spark_session.createDataFrame(d)
            # https://stackoverflow.com/a/57292987/6901725
            # d_spark.replace(float("nan"), None)  # to get coalesce effects (didn't work)
            d_spark.createOrReplaceTempView(table_name)  # TODO: non-temps
        except Exception as ex:
            raise ValueError("Spark problem inserting table, " + str(ex))


cached_spark_context = None


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    if not have_Spark:
        return None
    global cached_spark_context
    if cached_spark_context is None:
        cached_spark_context = pyspark.SparkContext()
    return SparkSQLModel().db_handle(
        SparkConnection(
            spark_context=cached_spark_context,
            spark_session=pyspark.sql.SparkSession.builder.appName(
                "pandasToSparkDF"
            ).getOrCreate(),
        )
    )
