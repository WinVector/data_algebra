import numpy

import data_algebra
import data_algebra.env
import data_algebra.test_util
import data_algebra.util
import data_algebra.yaml
from data_algebra.data_ops import *


def test_exp():
    # ask YAML to write simpler structures
    data_algebra.yaml.fix_ordered_dict_yaml_rep()

    d_local = data_algebra.pd.DataFrame(
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

    scale = 0.237

    with data_algebra.env.Env(locals()) as env:
        ops = TableDescription(
            "d", ["subjectID", "surveyCategory", "assessmentTotal"]
        ).extend({"v": "assessmentTotal.exp()+1"})

    res_local = ops.transform(d_local)

    expect = data_algebra.pd.DataFrame(
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

    assert data_algebra.test_util.equivalent_frames(res_local, expect, float_tol=1e-3)

    data_algebra.test_util.check_op_round_trip(ops)
