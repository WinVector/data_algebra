import re
import io
import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.cdata import pivot_blocks_to_rowrecs,  pivot_rowrecs_to_blocks, pivot_specification, unpivot_specification, RecordMap, RecordSpecification 
import data_algebra.SQLite
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util


def test_cdata_wvpy_case():
    # use case of wvpy.util.threshold_plot
    buf = io.StringIO(
        re.sub(
            "[ \\t]+",
            "",
            """
threshold,count,fraction,precision,true_positive_rate,false_positive_rate,true_negative_rate,false_negative_rate,accuracy,cdf,recall,sensitivity,specificity
0.999999,5,1.0,0.4,1.0,1.0,0.0,0.0,0.4,0.0,1.0,1.0,0.0
1.0,5,1.0,0.4,1.0,1.0,0.0,0.0,0.4,0.0,1.0,1.0,0.0
2.0,4,0.8,0.5,1.0,0.6666666666666666,0.3333333333333333,0.0,0.6,0.19999999999999996,1.0,1.0,0.33333333333333337
3.0,3,0.6,0.6666666666666666,1.0,0.3333333333333333,0.6666666666666666,0.0,0.8,0.4,1.0,1.0,0.6666666666666667
4.0,2,0.4,0.5,0.5,0.3333333333333333,0.6666666666666666,0.5,0.6,0.6,0.5,0.5,0.6666666666666667
5.0,1,0.2,0.0,0.0,0.3333333333333333,0.6666666666666666,1.0,0.4,0.8,0.0,0.0,0.6666666666666667
5.000001,0,0.0,0.0,0.0,0.0,1.0,1.0,0.6,1.0,0.0,0.0,1.0
    """,
        )
    )
    to_plot = data_algebra.data_model.default_data_model().pd.read_csv(buf)

    plotvars = ["sensitivity", "specificity"]
    reshaper = RecordMap(
        blocks_out=RecordSpecification(
            data_algebra.data_model.default_data_model().pd.DataFrame(
                {"measure": plotvars, "value": plotvars}
            ),
            control_table_keys=["measure"],
            record_keys=["threshold"],
            strict=False,
        ),
        strict=False,
    )
    prtlong = reshaper.transform(to_plot)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "threshold": [
                0.999999,
                0.999999,
                1.0,
                1.0,
                2.0,
                2.0,
                3.0,
                3.0,
                4.0,
                4.0,
                5.0,
                5.0,
                5.000001,
                5.000001,
            ],
            "measure": [
                "sensitivity",
                "specificity",
                "sensitivity",
                "specificity",
                "sensitivity",
                "specificity",
                "sensitivity",
                "specificity",
                "sensitivity",
                "specificity",
                "sensitivity",
                "specificity",
                "sensitivity",
                "specificity",
            ],
            "value": [
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.3333333333333333,
                1.0,
                0.6666666666666667,
                0.5,
                0.6666666666666667,
                0.0,
                0.6666666666666667,
                0.0,
                1.0,
            ],
        }
    )
    assert data_algebra.test_util.equivalent_frames(expect, prtlong)

    pipe1 = descr(to_plot=to_plot).convert_records(reshaper)
    data_algebra.test_util.check_transform(ops=pipe1, data=to_plot, expect=prtlong,
    )
