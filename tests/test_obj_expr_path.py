
from data_algebra import descr, d_, one
import data_algebra.test_util

def test_obj_expr_path_1():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({
        'subjectID':[1, 1, 2, 2],
        'surveyCategory': [ "withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
        'assessmentTotal': [5., 2., 3., 4.],
        'irrelevantCol1': ['irrel1']*4,
        'irrelevantCol2': ['irrel2']*4,
    })
    scale = 0.237
    ops2 = (
        descr(d=d)
            .extend({"probability": (d_.assessmentTotal * scale).exp()})
            .extend({"total": d_.probability.sum()}, partition_by=["subjectID"])
            .extend({"probability": d_.probability / d_.total})
            .extend(
                {"row_number": one.cumsum()},
                partition_by=["subjectID"],
                order_by=["probability"],
                reverse=["probability"],
            )
            .select_rows(d_.row_number == 1)
            .select_columns(
                ["subjectID", "surveyCategory", "probability"])
            .rename_columns({"diagnosis": "surveyCategory"})
    )
    res = d >> ops2
    expect = pd.DataFrame({
        "subjectID": [1, 2],
        "diagnosis": ["withdrawal behavior", "positive re-framing"],
        "probability": [0.670622, 0.558974],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect, float_tol=1e-4)
