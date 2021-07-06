import data_algebra.data_ops
import data_algebra.db_model


have_Spark = False
try:
    # noinspection PyUnresolvedReferences
    import pyspark
    import pyspark.sql

    have_Spark = True
except ImportError:
    have_Spark = False


def _sparksql_mean_expr(dbmodel, expression):
    return (
        "AVG(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _sparksql_size_expr(dbmodel, expression):
    return "SUM(1)"



# map from op-name to special SQL formatting code
SparkSQL_formatters = {
    "___": lambda dbmodel, expression: expression.to_python(),
    "mean": _sparksql_mean_expr,
    "size": _sparksql_size_expr,
}


class SparkConnection:
    def __init__(self, *, spark_context, spark_session):
        assert have_Spark
        assert isinstance(spark_context, pyspark.context.SparkContext)
        assert isinstance(spark_session, pyspark.sql.session.SparkSession)
        self.spark_context = spark_context
        self.spark_session = spark_session

    def close(self):
        if self.spark_conext is not None:
            self.spark_conext.stop()  # probably only for local demos
            self.spark_conext = None
        if self.spark_session is not None:
            self.spark_session = None


class SparkSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for SparkSQL"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(
            self,
            identifier_quote="`",
            string_quote='"',
            sql_formatters=SparkSQL_formatters,
        )

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        assert isinstance(conn, SparkConnection)
        assert isinstance(q, str)
        conn.spark_session.sql(q)

    def read_query(self, conn, q):
        assert isinstance(conn, SparkConnection)
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            temp_tables = dict()
            q = q.to_sql(db_model=self, temp_tables=temp_tables)
            if len(temp_tables) > 1:
                raise ValueError("ops require management of temp tables, please collect them via to_sql(temp_tables)")
        else:
            q = str(q)
        res = conn.spark_session.sql(q)
        # or res.collect()
        return res.toPandas()  # TODO: make sure it is our dataframe type

    # noinspection PyMethodMayBeStatic
    def insert_table(
        self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False
    ):
        assert isinstance(conn, SparkConnection)
        assert isinstance(table_name, str)
        if qualifiers is not None:
            raise ValueError("non-empty qualifiers not yet supported on insert")
        if self.table_exists(conn, table_name):
            if not allow_overwrite:
                raise ValueError("table " + table_name + " already exists")
            else:
                self.drop_table(conn, table_name, check=False)
        d_spark = conn.spark_session.createDataFrame(d)
        d_spark.createOrReplaceTempView(table_name)  # TODO: non-temps and non allow_overwrite


spark_context = None


def example_handle():
    """
    Return an example db handle for testing. Returns None if helper packages not present.

    """
    if not have_Spark:
        return None
    global spark_context
    if spark_context is None:
        spark_context = pyspark.SparkContext()
    return SparkSQLModel().db_handle(
            SparkConnection(
                spark_context=spark_context,
                spark_session=pyspark.sql.SparkSession.builder.appName('pandasToSparkDF').getOrCreate()))
