import pandas
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.SparkSQL
import data_algebra.util


def test_spark_sql():
    ops = TableDescription(
        "stocks", ["date", "trans", "symbol", "qty", "price"]
    ).extend({"cost": "qty * price"})

    pp = ops.to_python(pretty=True)

    db_model = data_algebra.SparkSQL.SparkSQLModel()

    sql = ops.to_sql(db_model, pretty=True)
