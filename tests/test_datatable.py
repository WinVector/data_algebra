

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

    d = datatable.Frame(
        subjectID=[1, 1, 2, 2],
        surveyCategory=["withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
        assessmentTotal=[5, 2, 3, 4],
     )

    ops = TableDescription('d', ['subjectID', 'surveyCategory', 'assessmentTotal']). \
        select_columns(['subjectID', 'surveyCategory'])

    ops.transform(d)