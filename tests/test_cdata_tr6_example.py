
import data_algebra
import data_algebra.data_model
import data_algebra.test_util
import data_algebra.cdata 



def test_cdata_tr6_cdata_example():
    pd = data_algebra.data_model.default_data_model().pd
    c1 = pd.DataFrame({
        "k1": [1, 2, 3],
        "v1": ["a", "c", "e"],
        "v2": ["b", "d", "f"],
    })
    c2 = pd.DataFrame({
        "k2": [4, 5],
        "w1": ["a", "b"],
        "w2": ["c", "d"],
        "w3": ["e", "f"],
    })
    record_map_pandas = data_algebra.cdata.RecordMap(
            blocks_in=data_algebra.cdata.RecordSpecification(
                c1,
                control_table_keys=["k1"],
                record_keys=["id"],
            ),
            blocks_out=data_algebra.cdata.RecordSpecification(
                c2,
                control_table_keys=["k2"],
                record_keys=["id"],
            ),
    )
    d = pd.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "k1": [1, 2, 3, 1, 2, 3],
        "v1": ["a", "c", "e", "g", "i", "k"],
        "v2": ["b", "d", "f", "h", "j", "l"],
    })
    expect =  pd.DataFrame({
        "id": [1, 1, 2, 2],
        "k2": [4, 5, 4, 5],
        "w1": ["a", "b", "g", "h"],
        "w2": ["c", "d", "i", "j"],
        "w3": ["e", "f", "k", "l"],
    })
    conv_pandas_rm = record_map_pandas.transform(d)
    assert data_algebra.test_util.equivalent_frames(conv_pandas_rm, expect)
    ops = (
        data_algebra.descr(d=d)
            .convert_records(record_map=record_map_pandas)
    )
    conv_pandas_ops = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(conv_pandas_ops, expect)
    conv_polars_ops = ops.transform(pd.DataFrame(d))
    assert isinstance(conv_polars_ops, pd.DataFrame)
    assert data_algebra.test_util.equivalent_frames(conv_polars_ops, expect)
    # again with pure Polars structures
    record_map_polars = data_algebra.cdata.RecordMap(
            blocks_in=data_algebra.cdata.RecordSpecification(
                pd.DataFrame(c1),
                control_table_keys=["k1"],
                record_keys=["id"],
            ),
            blocks_out=data_algebra.cdata.RecordSpecification(
                pd.DataFrame(c2),
                control_table_keys=["k2"],
                record_keys=["id"],
            ),
    )
    conv_pure_polars_rm = record_map_polars.transform(pd.DataFrame(d))
    assert isinstance(conv_pure_polars_rm, pd.DataFrame)
    assert data_algebra.test_util.equivalent_frames(conv_pure_polars_rm, expect)
    ops_polars = (
        data_algebra.descr(d=pd.DataFrame(d))
            .convert_records(record_map=record_map_polars)
    )
    conv_pure_polars_ops = ops_polars.transform(pd.DataFrame(d))
    assert isinstance(conv_pure_polars_ops, pd.DataFrame)
    assert data_algebra.test_util.equivalent_frames(conv_pure_polars_ops, expect)
    # db recheck
    ops = (
        data_algebra.descr(d=d)
            .convert_records(record_map=record_map_pandas)
    )
    data_algebra.test_util.check_transform(
        ops=ops, 
        data={"d": d}, 
        expect=expect,
        valid_for_empty=False,
        empty_produces_empty=False,
    )


def test_cdata_tr6_cdata_example_exbb():
    pd = data_algebra.data_model.default_data_model().pd
    c1 = pd.DataFrame({
        "k1": [1, 2, 3],
        "v1": ["a", "c", "e"],
        "v2": ["b", "d", "f"],
    })
    c2 = pd.DataFrame({
        "k2": [4, 5],
        "w1": ["a", "b"],
        "w2": ["c", "d"],
        "w3": ["e", "f"],
    })
    rm = data_algebra.cdata.RecordMap(
            blocks_in=data_algebra.cdata.RecordSpecification(
                c1,
                control_table_keys=["k1"],
                record_keys=["id"],
            ),
            blocks_out=data_algebra.cdata.RecordSpecification(
                c2,
                control_table_keys=["k2"],
                record_keys=["id"],
            ),
    )
    rm_str = str(rm)
    assert isinstance(rm_str, str)
    rm_repr = rm.__repr__()
    assert isinstance(rm_repr, str)
    inp1 = rm.example_input()
    assert isinstance(inp1, pd.DataFrame)
    expect_inp1 = pd.DataFrame({
        "id": ["id record key", "id record key", "id record key"],
        "k1": [1, 2, 3],
        "v1": ["a value", "c value", "e value"],
        "v2": ["b value", "d value", "f value"],
    })
    assert data_algebra.test_util.equivalent_frames(inp1, expect_inp1)
    out1 = rm.transform(inp1)
    assert isinstance(out1, pd.DataFrame)
    expect_out1 = pd.DataFrame({
        "id": ["id record key", "id record key"],
        "k2": [4, 5],
        "w1": ["a value", "b value"],
        "w2": ["c value", "d value"],
        "w3": ["e value", "f value"],
    })
    assert data_algebra.test_util.equivalent_frames(out1, expect_out1)
    back_1 = rm.inverse().transform(out1)
    assert data_algebra.test_util.equivalent_frames(back_1, expect_inp1)
    # db recheck
    ops = (
        data_algebra.descr(d=expect_inp1)
            .convert_records(record_map=rm)
    )
    data_algebra.test_util.check_transform(
        ops=ops, 
        data={"d": expect_inp1}, 
        expect=expect_out1,
        valid_for_empty=False,
        empty_produces_empty=False,
    )


def test_cdata_tr6_cdata_example_exbr():
    pd = data_algebra.data_model.default_data_model().pd
    c1 = pd.DataFrame({
        "k1": [1, 2, 3],
        "v1": ["a", "c", "e"],
        "v2": ["b", "d", "f"],
    })
    rm = data_algebra.cdata.RecordMap(
            blocks_in=data_algebra.cdata.RecordSpecification(
                c1,
                control_table_keys=["k1"],
                record_keys=["id"],
            ),
    )
    rm_str = str(rm)
    assert isinstance(rm_str, str)
    rm_repr = rm.__repr__()
    assert isinstance(rm_repr, str)
    inp1 = rm.example_input()
    assert isinstance(inp1, pd.DataFrame)
    expect_inp1 = pd.DataFrame({
        "id": ["id record key", "id record key", "id record key"],
        "k1": [1, 2, 3],
        "v1": ["a value", "c value", "e value"],
        "v2": ["b value", "d value", "f value"],
    })
    assert data_algebra.test_util.equivalent_frames(inp1, expect_inp1)
    out1 = rm.transform(inp1)
    assert isinstance(out1, pd.DataFrame)
    expect_out1 = pd.DataFrame({
        "id": ["id record key"],
        "a": ["a value"],
        "b": ["b value"],
        "c": ["c value"],
        "d": ["d value"],
        "e": ["e value"],
        "f": ["f value"],
    })
    assert data_algebra.test_util.equivalent_frames(out1, expect_out1)
    back_1 = rm.inverse().transform(out1)
    assert data_algebra.test_util.equivalent_frames(back_1, expect_inp1)
    # db recheck
    ops = (
        data_algebra.descr(d=expect_inp1)
            .convert_records(record_map=rm)
    )
    data_algebra.test_util.check_transform(
        ops=ops, 
        data={"d": expect_inp1}, 
        expect=expect_out1,
        valid_for_empty=False,
        empty_produces_empty=False,
    )


def test_cdata_tr6_cdata_example_exrb():
    pd = data_algebra.data_model.default_data_model().pd
    c2 = pd.DataFrame({
        "k2": [4, 5],
        "w1": ["a", "b"],
        "w2": ["c", "d"],
        "w3": ["e", "f"],
    })
    rm = data_algebra.cdata.RecordMap(
            blocks_out=data_algebra.cdata.RecordSpecification(
                c2,
                control_table_keys=["k2"],
                record_keys=["id"],
            ),
    )
    rm_str = str(rm)
    assert isinstance(rm_str, str)
    rm_repr = rm.__repr__()
    assert isinstance(rm_repr, str)
    inp1 = rm.example_input()
    assert isinstance(inp1, pd.DataFrame)
    expect_inp1 = pd.DataFrame({
        "id": ["id record key"],
        "a": ["a value"],
        "b": ["b value"],
        "c": ["c value"],
        "d": ["d value"],
        "e": ["e value"],
        "f": ["f value"],
    })
    assert data_algebra.test_util.equivalent_frames(inp1, expect_inp1)
    out1 = rm.transform(inp1)
    assert isinstance(out1, pd.DataFrame)
    expect_out1 = pd.DataFrame({
        "id": ["id record key", "id record key"],
        "k2": [4, 5],
        "w1": ["a value", "b value"],
        "w2": ["c value", "d value"],
        "w3": ["e value", "f value"],
    })
    assert data_algebra.test_util.equivalent_frames(out1, expect_out1)
    back_1 = rm.inverse().transform(out1)
    assert data_algebra.test_util.equivalent_frames(back_1, expect_inp1)
    # db recheck
    ops = (
        data_algebra.descr(d=expect_inp1)
            .convert_records(record_map=rm)
    )
    data_algebra.test_util.check_transform(
        ops=ops, 
        data={"d": expect_inp1}, 
        expect=expect_out1,
        valid_for_empty=False,
        empty_produces_empty=False,
    )
