

import os
import numpy as np
import data_algebra.BigQuery
import yaml

from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.view_representations import ViewRepresentation, TableDescription
import data_algebra.solutions
import data_algebra.test_util
import data_algebra.data_model


def test_xicor():
    pd = data_algebra.data_model.default_data_model().pd  # pandas

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "xicor_examples.yaml"), "r") as in_f:
        examples = yaml.safe_load(in_f)

    x_y_ops = data_algebra.solutions.xicor_query(
        TableDescription(table_name='df', column_names=['x', 'y']))

    def xicor(x, y):
        """
        Compute xicor of y treated as a function of x.

        :param x: vector of explanatory variable values.
        :param y: vector of dependent variable values.
        :return: xicor score (floating point number).
        """

        res_frame = x_y_ops.transform(pd.DataFrame({'x': x, 'y': y}))
        return res_frame['xicor'].values[0]

    x1 = xicor([1, 2, 3], [1, 2, 3])  # expect 0.25
    assert np.abs(x1 - 0.25) < 1e-5

    x2 = xicor([1, 2, 3], [3, 2, 1])  # expect 0.25
    assert np.abs(x2 - 0.25) < 1e-5

    x3 = xicor([1, 2, 3], [1, 3, 2])  # expect -0.125
    assert np.abs(x3 - -.125) < 1e-5

    def example_to_frame(ei):
        "encode an example into a data frame"
        example = examples[ei]
        a = example['a']
        b = example['b']
        return pd.DataFrame({'x': a, 'y': b, 'vname': f'v_{ei}'})

    example_frames = [example_to_frame(ei) for ei in range(len(examples))]
    example_frames = pd.concat(example_frames).reset_index(drop=True, inplace=False)

    rep_frame = pd.DataFrame({
        'rep': range(100)
    })

    grouped_calc = (
        data_algebra.solutions.xicor_query(
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

    if data_algebra.test_util.test_BigQuery:
        # try it in database
        db_handle = data_algebra.BigQuery.example_handle()
        db_handle.insert_table(example_frames, table_name='d', allow_overwrite=True)
        db_handle.insert_table(rep_frame, table_name='rep_frame', allow_overwrite=True)
        db_handle.drop_table("xicor")

        db_handle.execute(f"CREATE TABLE {db_handle.db_model.table_prefix}.xicor AS {db_handle.to_sql(grouped_calc)}")
        db_res = db_handle.read_query(f"SELECT * FROM {db_handle.db_model.table_prefix}.xicor ORDER BY vname")

        compare_res(db_res)

        db_handle.drop_table("d")
        db_handle.drop_table("rep_frame")
        db_handle.drop_table("xicor")
        db_handle.close()


def test_xicor_frame_calc():
    df = data_algebra.data_model.default_data_model().pd.DataFrame({'x1': [1, 2, 3], 'x2': [1, 1, 2], 'y': [1, 2, 3]})
    ops, rep_frame_name, rep_frame = data_algebra.solutions.xicor_score_variables_plan(
        descr(df=df),
        x_vars=['x1', 'x2'],
        y_name='y',
        n_rep=200,
    )
    res = ops.eval({'df': df, rep_frame_name: rep_frame})
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        'variable_name': ['x1', 'x2'],
        'xicor_mean': [0.25, 0.1],
        'xicor_std': [0.0, 0.18],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect, float_tol=0.1)
    data_algebra.test_util.check_transform(
        ops,
        data={'df': df, rep_frame_name: rep_frame},
        expect=expect,
        valid_for_empty=False,
        empty_produces_empty=False,
        float_tol = 0.1,
    )
