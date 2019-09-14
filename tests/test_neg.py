import pandas

import data_algebra
import data_algebra.env
import data_algebra.util
import data_algebra.yaml
from data_algebra.data_ops import *


def test_neg():
    # ask YAML to write simpler structures
    data_algebra.yaml.fix_ordered_dict_yaml_rep()

    d_local = pandas.DataFrame(
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
        ).extend({"v": "-assessmentTotal"})

    res_local = ops.transform(d_local)

    expect = pandas.DataFrame(
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

    assert data_algebra.util.equivalent_frames(res_local, expect, float_tol=1e-3)

    data_algebra.yaml.check_op_round_trip(ops)
