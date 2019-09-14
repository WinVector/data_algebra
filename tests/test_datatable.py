

import data_algebra
import data_algebra.env
import data_algebra.util
from data_algebra.data_ops import *



def test_datatable1():
    have_datatable = False
    try:
        import datatable
        have_datatable = True
    except ImportError:
        have_datatable = False

    if not have_datatable:
        return

    d_datatable = datatable.Frame(
        subjectID=[1, 1, 2, 2],
        surveyCategory=["withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
        assessmentTotal=[5, 2, 3, 4],
     )

    scale = 0.237

    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription('d',
                               ['subjectID',
                                'surveyCategory',
                                'assessmentTotal']). \
            extend({'probability': '(assessmentTotal * scale).exp()'})  # . \
            # extend({'total': 'probability.sum()'},
            #        partition_by='subjectID')  # . \
            # extend({'probability': 'probability/total'}). \
            # extend({'sort_key': '-1*probability'}). \
            # extend({'row_number': '_row_number()'},
            #        partition_by=['subjectID'],
            #        order_by=['sort_key']). \      # TODO: implemenmt forward
            # select_rows('row_number == 1'). \
            # select_columns(['subjectID', 'surveyCategory', 'probability'])

    res_datatable = ops.transform(d_datatable)

    expect = pandas.DataFrame({
        'subjectID': [1, 2],
        'surveyCategory': ["withdrawal behavior", "positive re-framing"],
        'probability': [0.670622, 0.558974],
    })

    # assert data_algebra.util.equivalent_frames(res_datatable, expect, float_tol=1e-3)
