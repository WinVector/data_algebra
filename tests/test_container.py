
import data_algebra

from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
from data_algebra.op_container import OpC
import data_algebra.BigQuery
import data_algebra.test_util


def test_container_1():
    d = data_algebra.default_data_model.pd.DataFrame({
        'subjectID': [1, 1, 2, 2],
        'surveyCategory': ["withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
        'assessmentTotal': [5., 2., 3., 4.],
        'irrelevantCol1': ['irrel1'] * 4,
        'irrelevantCol2': ['irrel2'] * 4,
    })

    scale = 0.237

    _ = OpC()
    ops2 = _.describe_table(d, 'd'). \
        extend({'probability': (_.c.assessmentTotal * scale).exp()}). \
        extend({'total': _.c.probability.sum()},
               partition_by='subjectID'). \
        extend({'probability': _.c.probability / _.c.total}). \
        extend({'row_number': _.v(1).cumsum()},
               partition_by=['subjectID'],
               order_by=['probability'], reverse=['probability']). \
        select_rows(_.c.row_number == 1). \
        select_columns(['subjectID', 'surveyCategory', 'probability']). \
        rename_columns({'diagnosis': 'surveyCategory'}). \
        ops()

    expect = data_algebra.default_data_model.pd.DataFrame({
        'subjectID': [1, 2],
        'diagnosis': ["withdrawal behavior", "positive re-framing"],
        'probability': [0.670622, 0.558974],
    })

    data_algebra.test_util.check_transform(
        ops=ops2,
        data=d,
        expect=expect,
        float_tol=1e-4)
