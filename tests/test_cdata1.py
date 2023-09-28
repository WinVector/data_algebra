import re
import io
import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.cdata import pivot_blocks_to_rowrecs,  pivot_rowrecs_to_blocks, pivot_specification, unpivot_specification, RecordMap, RecordSpecification 
import data_algebra.SQLite
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util


def test_small_cdata_example_debug():
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
    iris_orig = data_algebra.data_model.default_data_model().pd.read_csv(buf)

    buf = io.StringIO(
        re.sub(
            "[ \\t]+",
            "",
            """
    id,Species,Part,Measure,Value
    0,setosa,Petal,Length,1.4
    0,setosa,Petal,Width,0.2
    0,setosa,Sepal,Length,5.1
    0,setosa,Sepal,Width,3.5
    1,setosa,Petal,Length,1.4
    1,setosa,Petal,Width,0.2
    1,setosa,Sepal,Length,4.9
    1,setosa,Sepal,Width,3.0
    2,setosa,Petal,Length,1.3
    2,setosa,Petal,Width,0.2
    2,setosa,Sepal,Length,4.7
    2,setosa,Sepal,Width,3.2
    """,
        )
    )
    iris_blocks_orig = data_algebra.data_model.default_data_model().pd.read_csv(buf)

    iris_blocks = iris_blocks_orig.copy()
    iris = iris_orig.copy()

    control_table = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "Part": ["Sepal", "Sepal", "Petal", "Petal"],
            "Measure": ["Length", "Width", "Length", "Width"],
            "Value": ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
        }
    )
    record_spec = data_algebra.cdata.RecordSpecification(
        control_table,
        control_table_keys=["Part", "Measure"],
        record_keys=["id", "Species"],
    )

    mp_to_blocks = RecordMap(blocks_out=record_spec)
    waste_str = str(mp_to_blocks)
    assert isinstance(waste_str, str)
    arranged_blocks = mp_to_blocks.transform(iris)
    assert data_algebra.test_util.equivalent_frames(arranged_blocks, iris_blocks_orig)
    # arranged_blocks

    mp_to_rows = RecordMap(blocks_in=record_spec)
    waste_str = str(mp_to_rows)
    assert isinstance(waste_str, str)
    arranged_rows = mp_to_rows.transform(arranged_blocks)
    assert data_algebra.test_util.equivalent_frames(arranged_rows, iris_orig)
    # arranged_rows


def test_cdata1():
    # From: https://github.com/WinVector/data_algebra/blob/master/Examples/cdata/cdata.ipynb

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
    iris_orig = data_algebra.data_model.default_data_model().pd.read_csv(buf)

    buf = io.StringIO(
        re.sub(
            "[ \\t]+",
            "",
            """
    id,Species,Part,Measure,Value
    0,setosa,Petal,Length,1.4
    0,setosa,Petal,Width,0.2
    0,setosa,Sepal,Length,5.1
    0,setosa,Sepal,Width,3.5
    1,setosa,Petal,Length,1.4
    1,setosa,Petal,Width,0.2
    1,setosa,Sepal,Length,4.9
    1,setosa,Sepal,Width,3.0
    2,setosa,Petal,Length,1.3
    2,setosa,Petal,Width,0.2
    2,setosa,Sepal,Length,4.7
    2,setosa,Sepal,Width,3.2
    """,
        )
    )
    iris_blocks_orig = data_algebra.data_model.default_data_model().pd.read_csv(buf)

    iris_blocks = iris_blocks_orig.copy()
    iris = iris_orig.copy()

    waste_str = str(iris)

    # from:
    #   https://github.com/WinVector/cdata/blob/master/vignettes/control_table_keys.Rmd

    control_table = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "Part": ["Sepal", "Sepal", "Petal", "Petal"],
            "Measure": ["Length", "Width", "Length", "Width"],
            "Value": ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
        }
    )
    record_spec = data_algebra.cdata.RecordSpecification(
        control_table,
        control_table_keys=["Part", "Measure"],
        record_keys=["id", "Species"],
    )
    waste_str = str(record_spec)
    assert isinstance(waste_str, str)
    
    db_model = data_algebra.SQLite.SQLiteModel()

    source_table = data_algebra.data_ops.TableDescription(
        table_name="iris",
        column_names=[
            "id",
            "Sepal.Length",
            "Sepal.Width",
            "Petal.Length",
            "Petal.Width",
            "Species",
        ],
    )

    conn = sqlite3.connect(":memory:")

    sql = db_model.to_sql(source_table.map_records(blocks_out=record_spec))
    waste_str = str(sql)

    db_model.insert_table(conn, iris, "iris")

    res_blocks = db_model.read_query(conn, sql)
    assert data_algebra.test_util.equivalent_frames(res_blocks, iris_blocks_orig)
    waste_str = str(res_blocks)

    db_model.insert_table(conn, res_blocks, "res_blocks")
    source_table2 = data_algebra.data_ops.TableDescription(
        table_name="res_blocks",
        column_names=["id", "Species", "Part", "Measure", "Value"],
    )

    sql_back = db_model.to_sql(source_table2.map_records(blocks_in=record_spec))
    waste_str = str(sql_back)

    res_rows = db_model.read_query(conn, sql_back)
    assert data_algebra.test_util.equivalent_frames(res_rows, iris_orig)
    waste_str = str(res_rows)

    conn.close()

    waste_str = str(iris)

    mp_to_blocks = RecordMap(blocks_out=record_spec)
    waste_str = str(mp_to_blocks)
    arranged_blocks = mp_to_blocks.transform(iris)
    assert data_algebra.test_util.equivalent_frames(arranged_blocks, iris_blocks_orig)
    # arranged_blocks

    mp_to_rows = RecordMap(blocks_in=record_spec)
    waste_str = str(mp_to_rows)
    arranged_rows = mp_to_rows.transform(arranged_blocks)
    assert data_algebra.test_util.equivalent_frames(arranged_rows, iris_orig)
    # arranged_rows

    

    mp_to_and_back = RecordMap(blocks_in=record_spec, blocks_out=record_spec)
    waste_str = str(mp_to_and_back)
    arranged_self = mp_to_and_back.transform(iris_blocks)
    assert data_algebra.test_util.equivalent_frames(arranged_self, iris_blocks_orig)
    arranged_self


def test_cdata_explode():
    # from https://github.com/WinVector/cdata/blob/master/README.Rmd
    control = data_algebra.data_model.default_data_model().pd.read_csv(
        io.StringIO(
            re.sub(
                "[ \\t]+",
                "",
                """
                v1,v2,value1,value2
                Sepal.Length,Sepal.Length,Sepal.Length,Sepal.Length
                Sepal.Width,Sepal.Length,Sepal.Width,Sepal.Length
                Petal.Length,Sepal.Length,Petal.Length,Sepal.Length
                Petal.Width,Sepal.Length,Petal.Width,Sepal.Length
                Sepal.Length,Sepal.Width,Sepal.Length,Sepal.Width
                Sepal.Width,Sepal.Width,Sepal.Width,Sepal.Width
                Petal.Length,Sepal.Width,Petal.Length,Sepal.Width
                Petal.Width,Sepal.Width,Petal.Width,Sepal.Width
                Sepal.Length,Petal.Length,Sepal.Length,Petal.Length
                Sepal.Width,Petal.Length,Sepal.Width,Petal.Length
                Petal.Length,Petal.Length,Petal.Length,Petal.Length
                Petal.Width,Petal.Length,Petal.Width,Petal.Length
                Sepal.Length,Petal.Width,Sepal.Length,Petal.Width
                Sepal.Width,Petal.Width,Sepal.Width,Petal.Width
                Petal.Length,Petal.Width,Petal.Length,Petal.Width
                Petal.Width,Petal.Width,Petal.Width,Petal.Width
                """,
            )
        )
    )
    iris = data_algebra.data_model.default_data_model().pd.read_csv(
        io.StringIO(
            re.sub(
                "[ \\t]+",
                "",
                """
                Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Species,iris_id
                5.1,3.5,1.4,0.2,setosa,1
                4.9,3,1.4,0.2,setosa,2
                4.7,3.2,1.3,0.2,setosa,3
                4.6,3.1,1.5,0.2,setosa,4
                5,3.6,1.4,0.2,setosa,5
                5.4,3.9,1.7,0.4,setosa,6
    """,
            )
        )
    )
    expect = data_algebra.data_model.default_data_model().pd.read_csv(
        io.StringIO(
            re.sub(
                "[ \\t]+",
                "",
                """
                iris_id,Species,v1,v2,value1,value2
                1,setosa,Sepal.Length,Sepal.Length,5.1,5.1
                1,setosa,Sepal.Width,Sepal.Length,3.5,5.1
                1,setosa,Petal.Length,Sepal.Length,1.4,5.1
                1,setosa,Petal.Width,Sepal.Length,0.2,5.1
                1,setosa,Sepal.Length,Sepal.Width,5.1,3.5
                1,setosa,Sepal.Width,Sepal.Width,3.5,3.5
                1,setosa,Petal.Length,Sepal.Width,1.4,3.5
                1,setosa,Petal.Width,Sepal.Width,0.2,3.5
                1,setosa,Sepal.Length,Petal.Length,5.1,1.4
                1,setosa,Sepal.Width,Petal.Length,3.5,1.4
                1,setosa,Petal.Length,Petal.Length,1.4,1.4
                1,setosa,Petal.Width,Petal.Length,0.2,1.4
                1,setosa,Sepal.Length,Petal.Width,5.1,0.2
                1,setosa,Sepal.Width,Petal.Width,3.5,0.2
                1,setosa,Petal.Length,Petal.Width,1.4,0.2
                1,setosa,Petal.Width,Petal.Width,0.2,0.2
                2,setosa,Sepal.Length,Sepal.Length,4.9,4.9
                2,setosa,Sepal.Width,Sepal.Length,3,4.9
                2,setosa,Petal.Length,Sepal.Length,1.4,4.9
                2,setosa,Petal.Width,Sepal.Length,0.2,4.9
                2,setosa,Sepal.Length,Sepal.Width,4.9,3
                2,setosa,Sepal.Width,Sepal.Width,3,3
                2,setosa,Petal.Length,Sepal.Width,1.4,3
                2,setosa,Petal.Width,Sepal.Width,0.2,3
                2,setosa,Sepal.Length,Petal.Length,4.9,1.4
                2,setosa,Sepal.Width,Petal.Length,3,1.4
                2,setosa,Petal.Length,Petal.Length,1.4,1.4
                2,setosa,Petal.Width,Petal.Length,0.2,1.4
                2,setosa,Sepal.Length,Petal.Width,4.9,0.2
                2,setosa,Sepal.Width,Petal.Width,3,0.2
                2,setosa,Petal.Length,Petal.Width,1.4,0.2
                2,setosa,Petal.Width,Petal.Width,0.2,0.2
                3,setosa,Sepal.Length,Sepal.Length,4.7,4.7
                3,setosa,Sepal.Width,Sepal.Length,3.2,4.7
                3,setosa,Petal.Length,Sepal.Length,1.3,4.7
                3,setosa,Petal.Width,Sepal.Length,0.2,4.7
                3,setosa,Sepal.Length,Sepal.Width,4.7,3.2
                3,setosa,Sepal.Width,Sepal.Width,3.2,3.2
                3,setosa,Petal.Length,Sepal.Width,1.3,3.2
                3,setosa,Petal.Width,Sepal.Width,0.2,3.2
                3,setosa,Sepal.Length,Petal.Length,4.7,1.3
                3,setosa,Sepal.Width,Petal.Length,3.2,1.3
                3,setosa,Petal.Length,Petal.Length,1.3,1.3
                3,setosa,Petal.Width,Petal.Length,0.2,1.3
                3,setosa,Sepal.Length,Petal.Width,4.7,0.2
                3,setosa,Sepal.Width,Petal.Width,3.2,0.2
                3,setosa,Petal.Length,Petal.Width,1.3,0.2
                3,setosa,Petal.Width,Petal.Width,0.2,0.2
                4,setosa,Sepal.Length,Sepal.Length,4.6,4.6
                4,setosa,Sepal.Width,Sepal.Length,3.1,4.6
                4,setosa,Petal.Length,Sepal.Length,1.5,4.6
                4,setosa,Petal.Width,Sepal.Length,0.2,4.6
                4,setosa,Sepal.Length,Sepal.Width,4.6,3.1
                4,setosa,Sepal.Width,Sepal.Width,3.1,3.1
                4,setosa,Petal.Length,Sepal.Width,1.5,3.1
                4,setosa,Petal.Width,Sepal.Width,0.2,3.1
                4,setosa,Sepal.Length,Petal.Length,4.6,1.5
                4,setosa,Sepal.Width,Petal.Length,3.1,1.5
                4,setosa,Petal.Length,Petal.Length,1.5,1.5
                4,setosa,Petal.Width,Petal.Length,0.2,1.5
                4,setosa,Sepal.Length,Petal.Width,4.6,0.2
                4,setosa,Sepal.Width,Petal.Width,3.1,0.2
                4,setosa,Petal.Length,Petal.Width,1.5,0.2
                4,setosa,Petal.Width,Petal.Width,0.2,0.2
                5,setosa,Sepal.Length,Sepal.Length,5,5
                5,setosa,Sepal.Width,Sepal.Length,3.6,5
                5,setosa,Petal.Length,Sepal.Length,1.4,5
                5,setosa,Petal.Width,Sepal.Length,0.2,5
                5,setosa,Sepal.Length,Sepal.Width,5,3.6
                5,setosa,Sepal.Width,Sepal.Width,3.6,3.6
                5,setosa,Petal.Length,Sepal.Width,1.4,3.6
                5,setosa,Petal.Width,Sepal.Width,0.2,3.6
                5,setosa,Sepal.Length,Petal.Length,5,1.4
                5,setosa,Sepal.Width,Petal.Length,3.6,1.4
                5,setosa,Petal.Length,Petal.Length,1.4,1.4
                5,setosa,Petal.Width,Petal.Length,0.2,1.4
                5,setosa,Sepal.Length,Petal.Width,5,0.2
                5,setosa,Sepal.Width,Petal.Width,3.6,0.2
                5,setosa,Petal.Length,Petal.Width,1.4,0.2
                5,setosa,Petal.Width,Petal.Width,0.2,0.2
                6,setosa,Sepal.Length,Sepal.Length,5.4,5.4
                6,setosa,Sepal.Width,Sepal.Length,3.9,5.4
                6,setosa,Petal.Length,Sepal.Length,1.7,5.4
                6,setosa,Petal.Width,Sepal.Length,0.4,5.4
                6,setosa,Sepal.Length,Sepal.Width,5.4,3.9
                6,setosa,Sepal.Width,Sepal.Width,3.9,3.9
                6,setosa,Petal.Length,Sepal.Width,1.7,3.9
                6,setosa,Petal.Width,Sepal.Width,0.4,3.9
                6,setosa,Sepal.Length,Petal.Length,5.4,1.7
                6,setosa,Sepal.Width,Petal.Length,3.9,1.7
                6,setosa,Petal.Length,Petal.Length,1.7,1.7
                6,setosa,Petal.Width,Petal.Length,0.4,1.7
                6,setosa,Sepal.Length,Petal.Width,5.4,0.4
                6,setosa,Sepal.Width,Petal.Width,3.9,0.4
                6,setosa,Petal.Length,Petal.Width,1.7,0.4
                6,setosa,Petal.Width,Petal.Width,0.4,0.4
    """,
            )
        )
    )

    record_spec = data_algebra.cdata.RecordSpecification(
        control, control_table_keys=["v1", "v2"], record_keys=["iris_id", "Species"], strict=False,
    )
    transform = RecordMap(blocks_out=record_spec, strict=False)
    res = transform.transform(iris)
    assert data_algebra.test_util.equivalent_frames(res, expect)


def test_cdata_query_details():
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"measure_1": [1, 2], "measure_2": [3, 4], "rec": ["a", "b"]}
    )
    td = describe_table(d, table_name="d")

    ops = td.convert_records(
        pivot_rowrecs_to_blocks(
            attribute_key_column="measurement",  # output column
            attribute_value_column="value",  # output column
            record_keys=["rec"],
            record_value_columns=["measure_1", "measure_2"],
        )
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "rec": ["a", "a", "b", "b"],
            "measurement": ["measure_1", "measure_2", "measure_1", "measure_2"],
            "value": [1, 3, 2, 4],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)

