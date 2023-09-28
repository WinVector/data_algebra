import pytest
import numpy

import data_algebra
from data_algebra.data_ops_types import OperatorPlatform
from data_algebra.view_representations import ViewRepresentation, TableDescription, ExtendNode, MethodUse
from data_algebra.data_ops import data, descr, describe_table, ex  # https://github.com/WinVector/data_algebra
import data_algebra.SparkSQL
import data_algebra.util
import data_algebra.test_util


def test_spark_sql():
    ops = TableDescription(
        table_name="stocks", column_names=["date", "trans", "symbol", "qty", "price"]
    ).extend({"cost": "qty * price"})
    ops_source = ops.to_python(pretty=True)
    assert isinstance(ops_source, str)
    db_model = data_algebra.SparkSQL.SparkSQLModel()
    spark_sql = ops.to_sql(db_model)
    assert isinstance(spark_sql, str)


def test_spark_sql_insert():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'x': [1.0, None, numpy.nan, numpy.inf],
    })
    if data_algebra.test_util.test_Spark:
        with data_algebra.SparkSQL.example_handle() as db_handle:
            db_handle.insert_table(d, table_name='d', allow_overwrite=True)
            back = db_handle.read_query('SELECT * FROM d')
        expect = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'x': [1.0, numpy.nan, numpy.nan, numpy.inf],
        })
        assert data_algebra.test_util.equivalent_frames(back, expect)


def test_spark_sql_non_rec():
    ops = TableDescription(
        table_name="stocks", column_names=["date", "trans", "symbol", "qty", "price"]
    ).extend({"qty": "qty.ffill()"}, order_by=['date'])
    db_model = data_algebra.SparkSQL.SparkSQLModel()
    non_rec = db_model.non_recommended_methods(ops)
    assert non_rec == [MethodUse(op_name='ffill', is_project=False, is_windowed=True, is_ordered=True)]
    with pytest.warns(UserWarning):
        db_model.to_sql(ops)
