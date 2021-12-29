

from data_algebra.data_ops import *
import data_algebra.test_util


def xicor_query(*, x_name: str = 'x', y_name: str = 'y'):
    """
    Build a query computing the xicor of y_name as a function of x_name.
    Ref: https://arxiv.org/abs/1909.10140

    xicor(x, y) : 1 - n sum(i = 0, n-2) |r(i+1) - r(i)| / (2 * sum(i=0, n-1) l(i) (n - l(i)),
    where r(i) is the rank of the i-th Y item when ordered by x, and l(i) is the reverse rank of
    the l-th Y item.

    :param x_name: name for explanatory variable column.
    :param y_name: name for dependent variable column.
    :return: data algebra query computing xicor.
    """
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    x_tie_breaker = x_name + "_tie_breaker"
    y_group = y_name + "_group"
    names = [
        x_name, y_name, x_tie_breaker, y_group,
        'l', 'n', 'r',
        'rplus', 'rdiff', 'lterm', 'num_sum', 'den_sum',
        'xicor'
        ]
    assert(len(names) == len(set(names)))
    ops = (
        TableDescription(table_name="data_frame", column_names=[x_name, y_name])
            .extend({y_group: f"{y_name}.as_str()"})  # Google BigQuery won't group by float
            .extend({    # convert types, and add in tie breaking column
                x_name: f"1.0 * {x_name}",
                y_name: f"1.0 * {y_name}",
                x_tie_breaker: "_uniform()"})
            .extend({"n": "(1).sum()"})  # annotate in number of rows
            .extend(  # compute y ranks, that we will use to compare rank changes wrt x
                {"r": "(1).cumsum()"}, order_by=[y_name])
            .extend(  # compute reverse y ranks, used to normalize for ties in denominator
                {"l": "(1).cumsum()"}, order_by=[y_name], reverse=[y_name])
            .extend(  # go to max rank of group tie breaking
                {"l": "l.max()", "r": "r.max()"}, partition_by=[y_group])
            .extend(  # get y rank and y rank of next x-item into same row so we can take a difference
                {"rplus": "r.shift(1)"},
                order_by=[x_name, x_tie_breaker],
                reverse=[x_name, x_tie_breaker],
            )
            .extend(  # compute numerator and denominator terms
                {"rdiff": "((rplus - r).abs()).coalesce(0)", "lterm": "l * (n - l)"})
            .project(   # aggregate to compute sums in xicor definition
                {"num_sum": "rdiff.sum()", "den_sum": "lterm.sum()",
                 "n": "n.max()"  # pseudo-aggregation n is constant across rows
                 })
            .extend(  # actual xicor formula
                {"xicor": "1.0 - ((n * num_sum) / (2.0 * den_sum))"})
            .select_columns(["xicor"])
        )
    return ops


def test_xicor():
    # https://github.com/WinVector/data_algebra/blob/main/Examples/xicor/xicor.ipynb
    x_y_ops = xicor_query(x_name='x', y_name='y')

    d = data_algebra.default_data_model.pd.DataFrame({
        'x': [1., 2., 3.],
        'y': [1., 2., 3.],
    })

    expect = data_algebra.default_data_model.pd.DataFrame({
        'xicor': [0.25],
    })

    res_pandas = x_y_ops.transform(d)

    assert data_algebra.test_util.equivalent_frames(expect, res_pandas)

    data_algebra.test_util.check_transform(
        ops=x_y_ops,
        data=d,
        expect=expect,
        empty_produces_empty=False)
