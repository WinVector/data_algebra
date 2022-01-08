

import os
from typing import Iterable
import numpy as np
import data_algebra.BigQuery
import yaml

from data_algebra.data_ops import *
import data_algebra.test_util

pd = data_algebra.default_data_model.pd  # pandas


def xicor_query(
        data: ViewRepresentation,
        *,
        x_name: str = 'x',
        y_name: str = 'y',
        var_keys: Iterable[str] = tuple()):
    """
    Build a query computing the xicor of y_name as a function of x_name for each var_keys group of rows.
    Ref: https://arxiv.org/abs/1909.10140

    xicor(x, y) : 1 - n sum(i = 0, n-2) |r(i+1) - r(i)| / (2 * sum(i=0, n-1) l(i) (n - l(i)),
    where r(i) is the rank of the i-th Y item when ordered by x, and l(i) is the reverse rank of
    the l-th Y item.

    :param x_name: name for explanatory variable column.
    :param y_name: name for dependent variable column.
    :param var_keys: list of names for variable id columns.
    :param rep_id: name for repetition id column.
    :return: data algebra query computing xicor.
    """
    assert isinstance(x_name, str)
    assert isinstance(y_name, str)
    assert not isinstance(var_keys, str)
    var_keys = list(var_keys)
    x_tie_breaker = x_name + "_tie_breaker"
    y_group = y_name + "_group"
    names = [
        x_name, y_name, x_tie_breaker, y_group,
        'l', 'n', 'r',
        'rplus', 'rdiff', 'lterm', 'num_sum', 'den_sum',
        'xicor'
        ] + var_keys
    assert(len(names) == len(set(names)))
    ops = (
        data
            .extend({y_group: f"{y_name}.as_str()"})  # Google BigQuery won't group by float
            .extend({    # convert types, and add in tie breaking column
                x_name: f"1.0 * {x_name}",
                y_name: f"1.0 * {y_name}",
                x_tie_breaker: "_uniform()"})
            .extend(
                {"n": "(1).sum()"}, partition_by=var_keys)  # annotate in number of rows
            .extend(  # compute y ranks, that we will use to compare rank changes wrt x
                {"r": "(1).cumsum()"}, order_by=[y_name], partition_by=var_keys)
            .extend(  # compute reverse y ranks, used to normalize for ties in denominator
                {"l": "(1).cumsum()"}, order_by=[y_name], reverse=[y_name], partition_by=var_keys)
            .extend(  # go to max rank of group tie breaking
                {"l": "l.max()", "r": "r.max()"}, partition_by=[y_group] + var_keys)
            .extend(  # get y rank and y rank of next x-item into same row so we can take a difference
                {"rplus": "r.shift(1)"},
                order_by=[x_name, x_tie_breaker],
                reverse=[x_name, x_tie_breaker],
                partition_by=var_keys,
                )
            .extend(  # compute numerator and denominator terms
                {"rdiff": "((rplus - r).abs()).coalesce(0)", "lterm": "l * (n - l)"})
            .project(   # aggregate to compute sums in xicor definition
                {"num_sum": "rdiff.sum()", "den_sum": "lterm.sum()",
                 "n": "n.max()"  # pseudo-aggregation n is constant across rows
                 },
                group_by=var_keys,
                )
            .extend(  # apply actual xicor formula
                {"xicor": "1.0 - ((n * num_sum) / (2.0 * den_sum))"})
            .select_columns(var_keys + ["xicor"])
        )
    return ops


def test_xicor():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "xicor_examples.yaml"), "r") as in_f:
        examples = yaml.safe_load(in_f)

    x_y_ops = xicor_query(TableDescription(table_name='df', column_names=['x', 'y']))

    def xicor(x, y):
        """
        Compute xicor of y treated as a function of x.

        :param x: vector of explanatory variable values.
        :param y: vector of dependent variable values.
        :return: xicor score (floating point number).
        """

        res_frame = x_y_ops.transform(pd.DataFrame({'x': x, 'y': y}))
        return res_frame['xicor'].values[0]

    # %%

    x1 = xicor([1, 2, 3], [1, 2, 3])  # expect 0.25
    assert x1 == 0.25
    x1

    # %%

    x2 = xicor([1, 2, 3], [3, 2, 1])  # expect 0.25
    assert x2 == 0.25
    x2

    # %%

    x3 = xicor([1, 2, 3], [1, 3, 2])  # expect -0.125
    assert x3 == -0.125
    x3

    # %%

    def example_to_frame(ei):
        "encode an example into a data frame"
        example = examples[ei]
        a = example['a']
        b = example['b']
        return pd.DataFrame({'x': a, 'y': b, 'vname': f'v_{ei}'})

    example_frames = [example_to_frame(ei) for ei in range(len(examples))]
    example_frames = pd.concat(example_frames).reset_index(drop=True, inplace=False)

    example_frames

    # %%

    rep_frame = pd.DataFrame({
        'rep': range(100)
    })

    # %%

    grouped_calc = (
        xicor_query(
            descr(d=example_frames)
                .natural_join(  # cross join rows to get experiment repetitions
                b=descr(rep_frame=rep_frame),
                by=[],
                jointype='cross',
            ),
            var_keys=['vname', 'rep'])
            .project({
            'xicor_mean': 'xicor.mean()',
            'xicor_std': 'xicor.std()',
        },
            group_by=['vname'])
            .order_rows(['vname'])
    )
    xicor_results = grouped_calc.eval({'d': example_frames, 'rep_frame': rep_frame})

    xicor_results

    # %%

    # compare results
    def compare_res(xicor_results_to_check):
        for ei in range(len(examples)):
            example = examples[ei]
            a = example['a']
            b = example['b']
            ref_xicor = example['xicor']
            our_result = xicor_results_to_check.loc[xicor_results_to_check['vname'] == f'v_{ei}', :]
            our_xicor_mean = our_result['xicor_mean'].values[0]
            our_xicor_std = our_result['xicor_std'].values[0]
            assert np.abs(np.mean(ref_xicor) - our_xicor_mean) < 0.05
            assert np.abs(np.std(ref_xicor) - our_xicor_std) < 0.05
            # print(f'ref: {np.mean(ref_xicor)} {np.std(ref_xicor)}, ours: {our_xicor_mean} {our_xicor_std}')

    compare_res(xicor_results)

    # %%

    if data_algebra.test_util.test_BigQuery:
        # try it in database
        db_handle = data_algebra.BigQuery.example_handle()
        db_handle.insert_table(example_frames, table_name='d', allow_overwrite=True)
        db_handle.insert_table(rep_frame, table_name='rep_frame', allow_overwrite=True)
        db_handle.drop_table("xicor")

        # %%

        db_handle.execute(f"CREATE TABLE {db_handle.db_model.table_prefix}.xicor AS {db_handle.to_sql(grouped_calc)}")
        db_res = db_handle.read_query(f"SELECT * FROM {db_handle.db_model.table_prefix}.xicor ORDER BY vname")

        # %%

        compare_res(db_res)

        db_handle.drop_table("d")
        db_handle.drop_table("rep_frame")
        db_handle.drop_table("xicor")
        db_handle.close()
