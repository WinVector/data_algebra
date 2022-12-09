

import numpy as np
import data_algebra
import data_algebra.db_space


def test_dup_detection_example_1():
    # from Examples/DupRows.ipynb
    pd = data_algebra.data_model.default_data_model().pd
    rng = np.random.default_rng(2022)

    def generate_example(*, n_columns: int = 5, n_rows: int = 10):
        assert isinstance(n_columns, int)
        assert isinstance(n_rows, int)
        return pd.DataFrame({
            f"col_{i:03d}": rng.choice(["a", "b", "c", "d"], size=n_rows, replace=True) for i in range(n_columns)
        })
    
    d = generate_example(n_columns=10, n_rows=1000)
    dup_locs_1 = d.duplicated(keep=False)
    dup_locs_2 = d.groupby(list(d.columns)).transform("size") > 1
    assert np.all(dup_locs_1 == dup_locs_2)
    ops = (
    data_algebra.descr(d=d)
        .extend({"count": "(1).sum()"}, partition_by=d.columns)
        .select_rows("count > 1")
        .drop_columns(["count"])
    )
    ops_res = ops.transform(d)
    assert ops_res.shape[0] == np.sum(dup_locs_1)
    db_tables = data_algebra.db_space.DBSpace()
    db_tables.insert(key="d", value=d)
    res_description = db_tables.execute(ops)
    db_res = db_tables.retrieve(res_description.table_name)
    assert db_res.shape[0] == np.sum(dup_locs_1)
    db_tables.close()
