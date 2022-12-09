import numpy

import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *


def test_exp():
    d_local = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "subjectID": [1, 1, 2, 2],
            "surveyCategory": [
                "withdrawal behavior",
                "positive re-framing",
                "withdrawal behavior",
                "positive re-framing",
            ],
            "assessmentTotal": [5, 2, 3, 4],
        }
    )

    ops = TableDescription(
        table_name="d", column_names=["subjectID", "surveyCategory", "assessmentTotal"]
    ).extend({"v": "assessmentTotal.exp()+1"})

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "subjectID": [1, 1, 2, 2],
            "surveyCategory": [
                "withdrawal behavior",
                "positive re-framing",
                "withdrawal behavior",
                "positive re-framing",
            ],
            "assessmentTotal": [5, 2, 3, 4],
            "v": numpy.exp([5, 2, 3, 4]),
        }
    )
    expect.v = expect.v + 1

    data_algebra.test_util.check_transform(ops=ops, data=d_local, expect=expect)
