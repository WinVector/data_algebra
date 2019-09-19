
import io
import re
import numpy

import pandas

from data_algebra.data_ops import *
from data_algebra.cdata_impl import RecordMap
import data_algebra.yaml
from data_algebra.cdata_impl import record_map_from_simple_obj
import data_algebra.util


def test_cdata_example():

    buf = io.StringIO(
        re.sub(
            "[ \\t]+",
            "",
            """
    Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Species,id
    5.1,3.5,1.4,0.2,setosa,0
    4.9,3.0,1.4,0.2,setosa,1
    4.7,3.2,1.3,0.2,setosa,2
    """,
        )
    )
    iris = pandas.read_csv(buf)

    td = describe_table(iris, 'iris')

    control_table = pandas.DataFrame(
        {
            "Part": ["Sepal", "Sepal", "Petal", "Petal"],
            "Measure": ["Length", "Width", "Length", "Width"],
            "Value": ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
        }
    )

    record_spec = data_algebra.cdata.RecordSpecification(
        control_table,
        control_table_keys=['Part', 'Measure'],
        record_keys=['id', 'Species']
    )

    ops = td.convert_records(record_map=RecordMap(blocks_out=record_spec))

    res = iris >> ops

    yaml_obj = ops.collect_representation()
    back = data_algebra.yaml.to_pipeline(yaml_obj)


def test_keras_example():
    obj = {'blocks_out': {'record_keys': 'epoch', 'control_table_keys': 'measure',
                    'control_table': {'measure': ['minus binary cross entropy', 'accuracy'],
                                      'training': ['loss', 'acc'], 'validation': ['val_loss', 'val_acc']}}}
    record_map = record_map_from_simple_obj(obj)
    data = pandas.DataFrame({
        'val_loss': [-0.377, -0.2997],
        'val_acc': [0.8722, 0.8895],
        'loss': [-0.5067, -0.3002],
        'acc': [0.7852, 0.904],
        'epoch': [1, 2],
    })
    res = record_map.transform(data)
    expect = pandas.DataFrame({
        'epoch': [1, 1, 2, 2],
        'measure': ['accuracy', 'minus binary cross entropy', 'accuracy', 'minus binary cross entropy'],
        'training': [0.7852, -0.5067, 0.9040, -0.3002],
        'validation': [0.8722, -0.3770, 0.8895, -0.2997],
    })
    assert data_algebra.util.equivalent_frames(res, expect)


def test_cdata_block():
    data = pandas.DataFrame({
        'record_id': [1, 1, 1, 2, 2, 2],
        'row': ['row1', 'row2', 'row3', 'row1', 'row2', 'row3'],
        'col1': [1, 4, 7, 11, 14, 17],
        'col2': [2, 5, 8, 12, 15, 18],
        'col3': [3, 6, 9, 13, 16, 19],
    })

    record_keys = ['record_id']

    incoming_shape = pandas.DataFrame({
        'row': ['row1', 'row2', 'row3'],
        'col1': ['v11', 'v21', 'v31'],
        'col2': ['v12', 'v22', 'v32'],
        'col3': ['v13', 'v23', 'v33'],
    })

    outgoing_shape = pandas.DataFrame({
        'column_label': ['rec_col1', 'rec_col2', 'rec_col3'],
        'c_row1': ['v11', 'v12', 'v13'],
        'c_row2': ['v21', 'v22', 'v23'],
        'c_row3': ['v31', 'v32', 'v33'],
    })

    record_map = data_algebra.cdata_impl.RecordMap(
        blocks_in=data_algebra.cdata.RecordSpecification(
            control_table=incoming_shape,
            record_keys=record_keys
        ),
        blocks_out=data_algebra.cdata.RecordSpecification(
            control_table=outgoing_shape,
            record_keys=record_keys
        ),
    )

    res = record_map.transform(data)

    inv = record_map.inverse()

    back = inv.transform(res)

    assert data_algebra.util.equivalent_frames(data, back)


def test_cdata_missing():
    data = pandas.DataFrame({
        'record_id': [1, 1, 1, 2, 2, 2],
        'row': ['row1', 'row2', 'row3', 'row1', 'row2', 'row3'],
        'col1': [1, 4, 7, 11, 14, 17],
        'col2': [2, 5, 8, 12, 15, 18],
        'col3': [3, 6, 9, 13, 16, 19],
    })

    record_keys = ['record_id']

    incoming_shape = pandas.DataFrame({
        'row': ['row1', 'row2', 'row3'],
        'col1': ['v11', 'v21', 'v31'],
        'col2': [None, 'v22', 'v32'],
        'col3': ['v13', 'v23', 'v33'],
    })

    outgoing_shape = pandas.DataFrame({
        'column_label': ['rec_col1', 'rec_col2', 'rec_col3'],
        'c_row1': ['v11', numpy.nan, 'v13'],
        'c_row2': ['v21', 'v22', 'v23'],
        'c_row3': ['v31', 'v32', 'v33'],
    })

    record_map = data_algebra.cdata_impl.RecordMap(
        blocks_in=data_algebra.cdata.RecordSpecification(
            control_table=incoming_shape,
            record_keys=record_keys
        ),
        blocks_out=data_algebra.cdata.RecordSpecification(
            control_table=outgoing_shape,
            record_keys=record_keys
        ),
    )

    res = record_map.transform(data)

    expect = pandas.DataFrame({
        'record_id': [1, 1, 1, 2, 2, 2],
        'column_label': ['rec_col1', 'rec_col2', 'rec_col3', 'rec_col1', 'rec_col2', 'rec_col3'],
        'c_row1': [1.0, None, 3.0, 11.0, None, 13.0],
        'c_row2': [4, 5, 6, 14, 15, 16],
        'c_row3': [7, 8, 9, 17, 18, 19],
        })

    assert data_algebra.util.equivalent_frames(res, expect)
