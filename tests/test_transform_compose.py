
import data_algebra
import data_algebra.data_ops


def test_transform_compose_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({"x": [1, 2, 3]})
    ops = (
        data_algebra.TableDescription(table_name="d", column_names=["x", "y"])
            .extend({"z": "x + y + 1"})
    )
    ops_sub = (
        data_algebra.descr(d=d)
            .extend({"y": "x + 2"})
    )
    # let pipelines act as values
    composed_1 = ops.transform(ops_sub)
    composed_2 = ops.eval({"d": ops_sub})
    assert isinstance(composed_1, data_algebra.data_ops.ViewRepresentation)
    assert isinstance(composed_2, data_algebra.data_ops.ViewRepresentation)
    expect = (
        data_algebra.descr(d=d)
            .extend({"y": "x + 2"})
            .extend({"z": "x + y + 1"})
    )
    assert composed_1 == expect
    assert composed_2 == expect
