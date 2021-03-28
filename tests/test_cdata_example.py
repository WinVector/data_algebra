import io
import re
import numpy

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *
from data_algebra.cdata import *
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
    iris = data_algebra.default_data_model.pd.read_csv(buf)

    td = describe_table(iris, "iris")

    control_table = data_algebra.default_data_model.pd.DataFrame(
        {
            "Part": ["Sepal", "Sepal", "Petal", "Petal"],
            "Measure": ["Length", "Width", "Length", "Width"],
            "Value": ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
        }
    )

    record_spec = RecordSpecification(
        control_table,
        control_table_keys=["Part", "Measure"],
        record_keys=["id", "Species"],
    )

    ops = td.convert_records(record_map=RecordMap(blocks_out=record_spec))


def test_cdata_block():
    data = data_algebra.default_data_model.pd.DataFrame(
        {
            "record_id": [1, 1, 1, 2, 2, 2],
            "row": ["row1", "row2", "row3", "row1", "row2", "row3"],
            "col1": [1, 4, 7, 11, 14, 17],
            "col2": [2, 5, 8, 12, 15, 18],
            "col3": [3, 6, 9, 13, 16, 19],
        }
    )

    record_keys = ["record_id"]

    incoming_shape = data_algebra.default_data_model.pd.DataFrame(
        {
            "row": ["row1", "row2", "row3"],
            "col1": ["v11", "v21", "v31"],
            "col2": ["v12", "v22", "v32"],
            "col3": ["v13", "v23", "v33"],
        }
    )

    outgoing_shape = data_algebra.default_data_model.pd.DataFrame(
        {
            "column_label": ["rec_col1", "rec_col2", "rec_col3"],
            "c_row1": ["v11", "v12", "v13"],
            "c_row2": ["v21", "v22", "v23"],
            "c_row3": ["v31", "v32", "v33"],
        }
    )

    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=incoming_shape, record_keys=record_keys
        ),
        blocks_out=RecordSpecification(
            control_table=outgoing_shape, record_keys=record_keys
        ),
    )

    res = record_map.transform(data)

    inv = record_map.inverse()

    back = inv.transform(res)

    assert data_algebra.test_util.equivalent_frames(data, back)

    ex_inp = record_map.example_input()
    e_f = record_map.transform(ex_inp)
    ex_inp_inv = inv.example_input()
    e_r = inv.transform(ex_inp_inv)
    assert data_algebra.test_util.equivalent_frames(ex_inp_inv, e_f)
    assert data_algebra.test_util.equivalent_frames(ex_inp, e_r)
    id_f = record_map.compose(inv).transform(ex_inp_inv)
    assert data_algebra.test_util.equivalent_frames(id_f, ex_inp_inv)
    id_r = inv.compose(record_map).transform(ex_inp)
    assert data_algebra.test_util.equivalent_frames(id_r, ex_inp)


def test_cdata_missing():
    data = data_algebra.default_data_model.pd.DataFrame(
        {
            "record_id": [1, 1, 1, 2, 2, 2],
            "row": ["row1", "row2", "row3", "row1", "row2", "row3"],
            "col1": [1, 4, 7, 11, 14, 17],
            "col2": [2, 5, 8, 12, 15, 18],
            "col3": [3, 6, 9, 13, 16, 19],
        }
    )

    record_keys = ["record_id"]

    incoming_shape = data_algebra.default_data_model.pd.DataFrame(
        {
            "row": ["row1", "row2", "row3"],
            "col1": ["v11", "v21", "v31"],
            "col2": [None, "v22", "v32"],
            "col3": ["v13", "v23", "v33"],
        }
    )

    outgoing_shape = data_algebra.default_data_model.pd.DataFrame(
        {
            "column_label": ["rec_col1", "rec_col2", "rec_col3"],
            "c_row1": ["v11", numpy.nan, "v13"],
            "c_row2": ["v21", "v22", "v23"],
            "c_row3": ["v31", "v32", "v33"],
        }
    )

    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=incoming_shape, record_keys=record_keys
        ),
        blocks_out=RecordSpecification(
            control_table=outgoing_shape, record_keys=record_keys
        ),
    )

    res = record_map.transform(data)

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "record_id": [1, 1, 1, 2, 2, 2],
            "column_label": [
                "rec_col1",
                "rec_col2",
                "rec_col3",
                "rec_col1",
                "rec_col2",
                "rec_col3",
            ],
            "c_row1": [1.0, None, 3.0, 11.0, None, 13.0],
            "c_row2": [4, 5, 6, 14, 15, 16],
            "c_row3": [7, 8, 9, 17, 18, 19],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res, expect)
