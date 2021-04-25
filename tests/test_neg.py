import data_algebra
import data_algebra.test_util
import data_algebra.util
from data_algebra.data_ops import *


def test_neg():
    d_local = data_algebra.default_data_model.pd.DataFrame(
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
            "d", ["subjectID", "surveyCategory", "assessmentTotal"]
        ).extend({"v": "-assessmentTotal"})

    res_local = ops.transform(d_local)

    expect = data_algebra.default_data_model.pd.DataFrame(
        {
            "subjectID": [1, 1, 2, 2],
            "surveyCategory": [
                "withdrawal behavior",
                "positive re-framing",
                "withdrawal behavior",
                "positive re-framing",
            ],
            "assessmentTotal": [5, 2, 3, 4],
            "v": [-5, -2, -3, -4],
        }
    )

    assert data_algebra.test_util.equivalent_frames(res_local, expect, float_tol=1e-3)
