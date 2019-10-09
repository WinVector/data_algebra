import re
import io
import sqlite3
import yaml
import pandas
from data_algebra.cdata import *
from data_algebra.cdata_impl import record_map_from_simple_obj
import data_algebra.SQLite
from data_algebra.data_ops import *
import data_algebra.util


def test_cdata1():

    # From: https://github.com/WinVector/data_algebra/blob/master/Examples/cdata/cdata.ipynb

    # %%

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
    iris_orig = pandas.read_csv(buf)

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
    iris_blocks_orig = pandas.read_csv(buf)

    iris_blocks = iris_blocks_orig.copy()
    iris = iris_orig.copy()

    waste_str = str(iris)

    # %%

    # from:
    #   https://github.com/WinVector/cdata/blob/master/vignettes/control_table_keys.Rmd

    control_table = pandas.DataFrame(
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

    # %%

    db_model = data_algebra.SQLite.SQLiteModel()

    source_table = data_algebra.data_ops.TableDescription(
        "iris",
        ["id", "Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"],
    )

    temp_table = data_algebra.data_ops.TableDescription(
        "control_table", record_spec.control_table.columns
    )

    conn = sqlite3.connect(":memory:")

    sql = db_model.row_recs_to_blocks_query(
        source_table.to_sql(db_model), record_spec, temp_table
    )
    waste_str = str(sql)

    # %%

    db_model.insert_table(conn, iris, "iris")
    db_model.insert_table(conn, record_spec.control_table, temp_table.table_name)

    res_blocks = db_model.read_query(conn, sql)
    assert data_algebra.util.equivalent_frames(res_blocks, iris_blocks_orig)
    waste_str = str(res_blocks)

    # %%

    db_model.insert_table(conn, res_blocks, "res_blocks")
    source_table2 = data_algebra.data_ops.TableDescription(
        "res_blocks", ["id", "Species", "Part", "Measure", "Value"]
    )

    sql_back = db_model.blocks_to_row_recs_query(
        source_table2.to_sql(db_model), record_spec
    )
    waste_str = str(sql_back)

    # %%

    res_rows = db_model.read_query(conn, sql_back)
    assert data_algebra.util.equivalent_frames(res_rows, iris_orig)
    waste_str = str(res_rows)

    # %%

    conn.close()

    # %%

    waste_str = str(iris)

    # %%

    mp_to_blocks = RecordMap(blocks_out=record_spec)
    waste_str = str(mp_to_blocks)
    arranged_blocks = mp_to_blocks.transform(iris)
    assert data_algebra.util.equivalent_frames(arranged_blocks, iris_blocks_orig)
    arranged_blocks

    # %%

    mp_to_rows = RecordMap(blocks_in=record_spec)
    waste_str = str(mp_to_rows)
    arranged_rows = mp_to_rows.transform(arranged_blocks)
    assert data_algebra.util.equivalent_frames(arranged_rows, iris_orig)
    arranged_rows

    # %%

    mp_to_and_back = RecordMap(
        blocks_in=record_spec, blocks_out=record_spec
    )
    waste_str = str(mp_to_and_back)
    arranged_self = mp_to_and_back.transform(iris_blocks)
    assert data_algebra.util.equivalent_frames(arranged_self, iris_blocks_orig)
    arranged_self

    #%%

    obj = mp_to_blocks.to_simple_obj()
    waste_str = str(yaml.dump(obj))

    #%%

    recovered_transform = record_map_from_simple_obj(obj)
    waste_str = str(recovered_transform)
