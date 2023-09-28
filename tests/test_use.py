
from typing import List
import data_algebra.test_util
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util
import data_algebra.MySQL


def test_use_1():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    def add_to_column(pipeline, colname, delta):
        return pipeline.extend({colname: f"{colname} + {delta}"})

    ops = data(d=d).use(add_to_column, "ID", 5)
    res = ops.ex()

    ops2 = data(d=d).extend({"ID": "ID + 5"})
    res2 = ex(ops2)

    assert data_algebra.test_util.equivalent_frames(res, res2)


def test_use_2():
    # https://github.com/WinVector/data_algebra/blob/main/Examples/Macros/use.ipynb
    # some example data
    d1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {"ID": [2, 3, 7, 7], "OP": ["A", "B", "B", "D"],}
    )

    d2 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 1, 2, 3, 4, 2, 4, 4, 5, 5, 6],
            "OP": ["A", "B", "A", "D", "C", "A", "D", "B", "A", "B", "B"],
        }
    )

    keys = ["ID"]

    def merge_in_counts(
        pipeline: ViewRepresentation,
        id_cols: List[str],
        new_table_descr: TableDescription,
    ):
        return pipeline.natural_join(
            b=new_table_descr.project(
                {f"count_{new_table_descr.table_name}": "(1).sum()"}, group_by=id_cols
            ),
            by=id_cols,
            jointype="full",
        )

    ops = (
        data(d1=d1)
        .project({"count_d1": "(1).sum()"}, group_by=keys)
        .use(merge_in_counts, keys, data(d2=d2))
    )

    count_cols = [c for c in ops.column_names if c.startswith("count_")]
    ops = ops.extend({f"{c}": f"{c}.coalesce_0()" for c in count_cols}).order_rows(keys)

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6, 7],
            "count_d1": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0],
            "count_d2": [2.0, 2.0, 1.0, 3.0, 2.0, 1.0, 0.0],
        }
    )

    data_algebra.test_util.check_transform(
        ops,
        data={"d1": d1, "d2": d2},
        expect=expect,
        models_to_skip={str(data_algebra.MySQL.MySQLModel())},
    )
    # MySQL seems to not handle `a`.`b` as a qualified column name
