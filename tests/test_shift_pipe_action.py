
import data_algebra.data_model
import data_algebra.test_util
from data_algebra.shift_pipe_action import ShiftPipeAction
from data_algebra.cdata import pivot_rowrecs_to_blocks
from data_algebra import descr, d_, lit
from data_algebra.view_representations import ConvertRecordsNode, TableDescription, ViewRepresentation
import data_algebra.SQLite


def test_shift_pipe_action_1():

    class T1(ShiftPipeAction):
        nm: str
        def __init__(self, nm: str) -> None:
            assert isinstance(nm, str)
            ShiftPipeAction.__init__(self)
            self.nm = nm
        
        def act_on(self, b, *, correct_ordered_first_call: bool = False):
            return f"{self}.act_on({b}, correct_ordered_first_call={correct_ordered_first_call})"
        
        def __str__(self) -> str:
            return self.nm

    a = T1("a")
    b = T1("b")
    assert (a >> b) == "b.act_on(a, correct_ordered_first_call=True)"
    assert (b >> a) == "a.act_on(b, correct_ordered_first_call=True)"
    assert (7 >> a) == "a.act_on(7, correct_ordered_first_call=True)"
    assert (a >> 7) == "a.act_on(7, correct_ordered_first_call=False)"
    assert a(7) == "a.act_on(7, correct_ordered_first_call=True)"
    assert a(b) == "a.act_on(b, correct_ordered_first_call=True)"


def test_shift_pipe_action_mp_ops_data():
    mp = pivot_rowrecs_to_blocks(
        attribute_key_column="curve",
        attribute_value_column="effect_size",
        record_keys=["n"],
        record_value_columns=["T", "T_E"],
    )
    ex_inp = mp.example_input()
    pd = data_algebra.data_model.default_data_model().pd
    expect = pd.DataFrame({
        "n": ["n record key", "n record key"],
        "curve": ["T", "T_E"],
        "effect_size": ["T value", "T_E value"],
    })
    # data interactions
    res_1d = ex_inp >> mp
    assert data_algebra.test_util.equivalent_frames(res_1d, expect)
    res_2d = mp >> ex_inp  # not the preferred notation, consequence of fallback in shiftpipe
    assert data_algebra.test_util.equivalent_frames(res_2d, expect)
    # data pipeline interactions
    ops = mp.as_pipeline()
    assert isinstance(ops, ViewRepresentation)
    res_1 = ex_inp >> ops
    assert data_algebra.test_util.equivalent_frames(res_1, expect)
    res_1b = ex_inp >> ops  # not the preferred notation, consequence of fallback in shiftpipe
    assert data_algebra.test_util.equivalent_frames(res_1b, expect)
    res_2 = ops.ex()
    assert data_algebra.test_util.equivalent_frames(res_2, expect)
    res_3 = ex_inp >> mp
    assert data_algebra.test_util.equivalent_frames(res_3, expect)
    res_ops = data_algebra.data_ops.descr(ex_inp=ex_inp) >> mp
    assert isinstance(res_ops, ViewRepresentation)
    res_5 = res_ops.transform(ex_inp)
    assert data_algebra.test_util.equivalent_frames(res_5, expect)
    # pipeline interactions
    res_p1 = descr(ex_inp=ex_inp) >> mp
    assert isinstance(res_p1, ConvertRecordsNode)
    res_p2 = mp >> descr(ex_inp=ex_inp)  # not the preferred notation, caught by fallbacks
    assert isinstance(res_p2, ConvertRecordsNode)
    assert res_p1 == res_p2
    # pipeline data interactions
    res_pd1 = ex_inp >> res_ops
    assert data_algebra.test_util.equivalent_frames(res_pd1, expect)
    res_pd2 = res_ops >> ex_inp  # not the preferred notation, due to fallback
    assert data_algebra.test_util.equivalent_frames(res_pd2, expect)
    # notational issue, moving back to . after >>
    res_c1 = (descr(ex_inp=ex_inp) >> mp).extend({"z": 1})
    assert isinstance(res_c1, ViewRepresentation)
    pref = descr(ex_inp=ex_inp).convert_records(mp).extend({"z": 1})
    assert pref == res_c1


def test_shift_pipe_dict():
    pd = data_algebra.data_model.default_data_model().pd
    d1 = pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "x", "y"], "z": [4, 5, 6],}
    )
    d2 = pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "y", "y"], "q": [7, 8, 9],}
    )
    ops1 = descr(d1=d1).natural_join(
        b=descr(d2=d2), by=["a"], jointype="inner",
    )
    res = {"d1": d1, "d2": d2} >> ops1
    expect1 = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "a": [1, 1, 1, 1, 2],
            "b": ["x", "x", "x", "x", "y"],
            "z": [4, 4, 5, 5, 6],
            "q": [7, 8, 7, 8, 9],
        }
    )
    assert data_algebra.test_util.equivalent_frames(res, expect1)


def test_shift_pipe_dict_replace():
    pd = data_algebra.data_model.default_data_model().pd
    d1 = pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "x", "y"], "z": [4, 5, 6],}
    )
    d2 = pd.DataFrame(
        {"a": [1, 1, 2], "b": ["x", "y", "y"], "q": [7, 8, 9],}
    )
    ops1 = descr(d1=d1).natural_join(
        b=descr(d2=d2), by=["a"], jointype="inner",
    )
    res = {
        "d1": TableDescription(column_names=["a", "c"], table_name="d1"), 
        "d2": TableDescription(column_names=["a", "d"], table_name="dZ")
        } >> ops1
    expect1 = (
        TableDescription(table_name="d1", column_names=["a", "c"]).natural_join(
            b=TableDescription(table_name="dZ", column_names=["a", "d"]),
            on=["a"],
            jointype="INNER",
        )
    )
    assert res == expect1


def test_shift_pipe_action_db_etc():
    pd = data_algebra.data_model.default_data_model().pd
    mp = pivot_rowrecs_to_blocks(
        attribute_key_column="curve",
        attribute_value_column="effect_size",
        record_keys=["n"],
        record_value_columns=["T", "T_E"],
    )
    ex_inp = mp.example_input()
    ops = mp.as_pipeline(table_name="ex_inp")
    assert isinstance(ops, ViewRepresentation)
    with data_algebra.SQLite.example_handle() as db_handle:
        db_handle.insert_table(ex_inp, table_name="ex_inp")
        res_db = ops >> db_handle
        sql = ops >> db_handle.db_model
        res_db_s = sql >> db_handle
    expect = pd.DataFrame({
        "n": ["n record key", "n record key"],
        "curve": ["T", "T_E"],
        "effect_size": ["T value", "T_E value"],
    })
    assert data_algebra.test_util.equivalent_frames(res_db, expect)
    assert isinstance(sql, str)
    assert data_algebra.test_util.equivalent_frames(res_db_s, expect)
