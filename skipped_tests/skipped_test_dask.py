
# dask doesn't seem to be behaving deterministicly (even throwing inside its own world on some re-runs)
# can't add it to the supported test line until we get around this

import pandas

import data_algebra
import data_algebra.env
import data_algebra.util
from data_algebra.data_ops import *



def test_dask1():
    have_dask = False
    try:
        import dask
        import dask.dataframe
        have_dask = True
    except ImportError:
        have_dask = False

    if not have_dask:
        return

    d_local = pandas.DataFrame({
        'subjectID': [1, 1, 2, 2],
        'surveyCategory': ["withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
        'assessmentTotal': [5, 2, 3, 4],
     })

    scale = 0.237

    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription('d',
                               ['subjectID',
                                'surveyCategory',
                                'assessmentTotal']). \
            extend({'probability': '(assessmentTotal * scale).exp()'}). \
            extend({'total': 'probability.sum()'},
                   partition_by='subjectID'). \
            extend({'probability': 'probability/total'}). \
            extend({'sort_key': '-1*probability'}). \
            extend({'row_number': '_row_number()'},
                   partition_by=['subjectID'],
                   order_by=['sort_key']). \
            select_rows('row_number == 1'). \
            select_columns(['subjectID', 'surveyCategory', 'probability'])

    res_local = ops.transform(d_local)

    expect = pandas.DataFrame({
        'subjectID': [1, 2],
        'surveyCategory': ["withdrawal behavior", "positive re-framing"],
        'probability': [0.670622, 0.558974],
    })

    assert data_algebra.util.equivalent_frames(res_local, expect, float_tol=1e-3)

    d_dask = dask.dataframe.from_pandas(d_local, npartitions=2)
    res_dask = ops.transform(d_dask)
    res_dask_local = res_dask.compute()

    assert data_algebra.util.equivalent_frames(res_dask_local, expect, float_tol=1e-3)

def test_dask2():
    have_dask = False
    try:
        import dask
        import dask.dataframe
        have_dask = True
    except ImportError:
        have_dask = False

    if not have_dask:
        return

    d_local = pandas.DataFrame({
        'subjectID': [1, 1, 2, 2],
        'surveyCategory': ["withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
        'assessmentTotal': [5, 2, 3, 4],
        'irrelevantCol1': ['irrel1'] * 4,
        'irrelevantCol2': ['irrel2'] * 4,
    })

    scale = 0.237

    with data_algebra.env.Env(locals()) as env:
        td = TableDescription('d',
                               ['subjectID',
                                'surveyCategory',
                                'assessmentTotal',
                                'irrelevantCol1',
                                'irrelevantCol2'])
        ops = td. \
            extend({'probability': '(assessmentTotal * scale).exp()'}). \
            extend({'total': 'probability.sum()'},
                   partition_by='subjectID'). \
            extend({'probability': 'probability/total'}). \
            extend({'row_number': '_row_number()'},
                   partition_by=['subjectID'],
                   order_by=['probability']). \
            select_rows('row_number==1'). \
            select_columns(['subjectID', 'surveyCategory', 'probability']). \
            rename_columns({'diagnosis': 'surveyCategory'}). \
            order_rows(['subjectID'])
        usc = td . \
            extend({'probability': '(assessmentTotal * scale).exp()'})
        ops_p = usc. \
            project({'total': 'probability.sum()'},
                    group_by='subjectID'). \
            natural_join(usc, by=['subjectID']). \
            extend({'probability': 'probability/total'})

    d_dask = dask.dataframe.from_pandas(d_local, npartitions=2)

    res_dask_p = ops_p.transform(d_dask)
    res_t_p = res_dask_p.compute()

    # res_dask = ops.transform(d_dask)
    # res_t = res_dask.compute()
    #
    # expect = pandas.DataFrame(
    #     {
    #         "subjectID": [1, 2],
    #         "diagnosis": ["withdrawal behavior", "positive re-framing"],
    #         "probability": [0.670622, 0.558974],
    #     }
    # )
    # assert data_algebra.util.equivalent_frames(expect, res_t, float_tol=1e-3)
