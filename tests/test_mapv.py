
import numpy
import pytest

import data_algebra
from data_algebra.data_ops import *
import data_algebra.SparkSQL
import data_algebra.test_util


def test_mapv_1():
    pd = data_algebra.default_data_model.pd
    d = pd.DataFrame({
            'x': ['a', 'b', 'c', None, numpy.nan, 'b'],
        })
    ops = (
        data(d=d)
            .extend({'x_mapped': '0.0'})  # not deleted, as it is a constant
            .extend({
                'z': '1 + -3',
                'x_mapped': 'x.mapv({"a": 1.0, "b": 2.0, "q": -3}, 0.5)'
                })
        )
    ops_str = str(ops)
    assert isinstance(ops_str, str)

    transformed = ops.transform(d)
    expect = pd.DataFrame({
        'x': ['a', 'b', 'c', None, None, 'b'],
        'z': -2,
        'x_mapped': [1.0, 2.0, 0.5, 0.5, 0.5, 2.0],
        })
    assert data_algebra.test_util.equivalent_frames(transformed, expect)

    db_model = data_algebra.SparkSQL.SparkSQLModel()
    with pytest.raises(ValueError):
        db_model.to_sql(ops)

    data_algebra.test_util.check_transform(
        ops=ops, data=d, expect=expect,
        models_to_skip=[str(data_algebra.SparkSQL.SparkSQLModel())])
    # Spark rejects the following correct query with:
    # TypeError: field x: Can not merge type <class 'pyspark.sql.types.StringType'> and <class 'pyspark.sql.types.DoubleType'>
    # SELECT  -- .extend({ 'z': '1 + -3', 'x_mapped': "x.mapv({'a': 1.0, 'b': 2.0, 'q': -3.0}, 0.5)"})
    #  `x` ,
    #  1 + -3 AS `z` ,
    #  CASE `x` WHEN "a" THEN 1.0 WHEN "b" THEN 2.0 WHEN "q" THEN -3.0 ELSE 0.5 END AS `x_mapped`
    # FROM
    #  `d`
    #
    # also fails for
    # SELECT  -- .extend({ 'z': '1 + -3', 'x_mapped': "x.mapv({'a': 1.0, 'b': 2.0, 'q': -3}, 0.5)"})
    # 1 + -3 AS `z` ,
    # CASE WHEN (`x` = "a") THEN 1.0 WHEN (`x` = "b") THEN 2.0 WHEN (`x` = "q") THEN -3 ELSE 0.5 END AS `x_mapped`
    # FROM
    # `d`
