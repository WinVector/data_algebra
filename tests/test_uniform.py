
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util
import numpy

have_polars = False
try:
    import polars as pl  # conditional import
    have_polars = True
except ModuleNotFoundError:
    pass


def test_uniform_1():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": range(20),
        }
    )
    ops = (
        descr(d=d)
            .extend({'r': '_uniform()'})
    )
    d_pandas_1 = ops.transform(d)
    assert 'r' in d_pandas_1.columns
    assert numpy.max(d_pandas_1['r']) <= 1.0
    assert numpy.min(d_pandas_1['r']) >= 0.0

    d_pandas_2 = ops.transform(d)
    assert not data_algebra.test_util.equivalent_frames(d_pandas_1, d_pandas_2)

    ops_z = (
        descr(d=d)
            .extend({'r': '_uniform()'})
            .extend({'r': '0 * r'})
    )
    expect =  data_algebra.data_model.default_data_model().pd.DataFrame({
            "ID": range(20),
            'r': 0.0
        })
    data_algebra.test_util.check_transform(ops_z, data={"d": d}, expect=expect,
    )


def test_uniform_2():
    # some example data
    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "ID": range(10000),
        }
    )
    ops = (
        descr(d=d)
            .extend({'r': '_uniform()'})
    )

    res_pandas = ops.transform(d)

    def check_r(res):
        assert 'r' in res.columns
        assert numpy.max(res['r']) <= 1.0
        assert numpy.min(res['r']) >= 0.0
        assert numpy.max(res['r']) > numpy.min(res['r'])
        # Kolmogorov Smirnov style distribution test (using inverse of CDF instead of CDF), should be okay for uniform
        r_sorted = numpy.sort(numpy.array(res['r']))
        r_ideal = numpy.array([i / (len(r_sorted) - 1) for i in range(len(r_sorted))])
        statistic = numpy.max(numpy.abs(r_sorted - r_ideal))
        assert statistic < 0.2

    check_r(res_pandas)

    if have_polars:
        res_polars = ops.transform(pl.DataFrame(d))
        assert isinstance(res_polars, pl.DataFrame)
        check_r(res_polars.to_pandas())

    handles = data_algebra.test_util.get_test_dbs()
    for h in handles:
        # print(h)
        h.insert_table(d, table_name='d', allow_overwrite=True)
        res_h = h.read_query(ops)
        check_r(res_pandas)
        h.drop_table('d')
    for h in handles:
        h.close()
