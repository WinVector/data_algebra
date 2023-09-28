
import data_algebra
import data_algebra.SQLite
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.cdata import pivot_blocks_to_rowrecs,  pivot_rowrecs_to_blocks, pivot_specification, unpivot_specification, RecordMap, RecordSpecification 


def test_extend_transform_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2]})
    ops = describe_table(d, table_name="d").extend({"y": 1})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [1, 1],})
    data_algebra.test_util.check_transform(ops, d, expect)


def test_extend_transform_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2]})
    ops = describe_table(d, table_name="d").extend({"y": "x + x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [2, 4],})
    data_algebra.test_util.check_transform(ops, d, expect)


def test_project_transform_2():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2]})
    ops = describe_table(d, table_name="d").project({"y": "x.max()"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [2],})
    data_algebra.test_util.check_transform(ops, d, expect, empty_produces_empty=False)


def test_natural_join_transform_1():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame({"k": ["a", "b"], "x": [1, 2],})
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"k": ["b", "c"], "x": [3, 4], "y": [5, 6],}
    )
    ops = describe_table(d1, table_name="d1").natural_join(
        b=describe_table(d2, table_name="d2"), by=["k"], jointype="LEFT"
    )
    data = {"d1": d1, "d2": d2}
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"k": ["a", "b"], "x": [1.0, 2.0], "y": [None, 5.0],}
    )
    data_algebra.test_util.check_transform(ops, data, expect)


def test_concat_rows_transform_1():
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame({"k": ["a", "b"], "x": [1, 2],})
    d2 = data_algebra.data_model.default_data_model().pd.DataFrame({"k": ["b", "c"], "x": [3, 4],})
    ops = describe_table(d1, table_name="d1").concat_rows(
        b=describe_table(d2, table_name="d2")
    )
    data = {"d1": d1, "d2": d2}
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "k": ["a", "b", "b", "c"],
            "x": [1, 2, 3, 4],
            "source_name": ["a", "a", "b", "b"],
        }
    )
    data_algebra.test_util.check_transform(ops, data, expect)


def test_select_rows_transform_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = describe_table(d, table_name="d").select_rows("x == 2")
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [2], "y": [4]})
    data_algebra.test_util.check_transform(ops, d, expect)


def test_drop_columns_transform_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = describe_table(d, table_name="d").drop_columns("x")
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [3, 4]})
    data_algebra.test_util.check_transform(ops, d, expect)


def test_select_columns_transform_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = describe_table(d, table_name="d").select_columns("y")
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [3, 4]})
    data_algebra.test_util.check_transform(ops, d, expect)


def test_rename_columns_transform_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = describe_table(d, table_name="d").rename_columns({"x": "y", "y": "x"})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"y": [1, 2], "x": [3, 4]})
    data_algebra.test_util.check_transform(ops, d, expect)


def test_order_rows_transform_1():
    d = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ops = describe_table(d, table_name="d").order_rows(["y"], reverse=["y"])
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({"x": [2, 1], "y": [4, 3]})
    data_algebra.test_util.check_transform(ops, d, expect, check_row_order=True)


def test_convert_records_transform_1():
    iris_small = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "Sepal_Length": [5.1, 4.9, 4.7],
            "Sepal_Width": [3.5, 3.0, 3.2],
            "Petal_Length": [1.4, 1.4, 1.3],
            "Petal_Width": [0.2, 0.2, 0.2],
            "Species": ["setosa", "setosa", "setosa"],
            "id": [0, 1, 2],
        }
    )

    control_table = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "Part": ["Sepal", "Sepal", "Petal", "Petal"],
            "Measure": ["Length", "Width", "Length", "Width"],
            "Value": ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"],
        }
    )

    record_spec = data_algebra.cdata.RecordSpecification(
        control_table,
        control_table_keys=["Part", "Measure"],
        record_keys=["id", "Species"],
    )

    map = RecordMap(blocks_out=record_spec)

    ops = describe_table(iris_small, "iris_small").convert_records(record_map=map)

    data = {
        "iris_small": iris_small,
    }

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "id": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "Species": [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
            ],
            "Part": [
                "Petal",
                "Petal",
                "Sepal",
                "Sepal",
                "Petal",
                "Petal",
                "Sepal",
                "Sepal",
                "Petal",
                "Petal",
                "Sepal",
                "Sepal",
            ],
            "Measure": [
                "Length",
                "Width",
                "Length",
                "Width",
                "Length",
                "Width",
                "Length",
                "Width",
                "Length",
                "Width",
                "Length",
                "Width",
            ],
            "Value": [1.4, 0.2, 5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.3, 0.2, 4.7, 3.2],
        }
    )

    data_algebra.test_util.check_transform(ops, data, expect)
