
import io
import re

import pandas

from data_algebra.data_ops import *
from data_algebra.cdata_impl import RecordMap


def test_cdata_example():

    buf = io.StringIO(
        re.sub(
            "[ \\t]+",
            "",
            """
    Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Species,id
    5.1,3.5,1.4,0.2,setosa,0
    4.9,3.0,1.4,0.2,setosa,1
    4.7,3.2,1.3,0.2,setosa,2
    """,
        )
    )
    iris = pandas.read_csv(buf)

    td = describe_table(iris, 'iris')

    control_table = pandas.DataFrame(
        {
            "Part": ["Sepal", "Sepal", "Petal", "Petal"],
            "Measure": ["Length", "Width", "Length", "Width"],
            "Value": ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
        }
    )

    record_spec = data_algebra.cdata.RecordSpecification(
        control_table,
        control_table_keys=['Part', 'Measure'],
        record_keys=['id', 'Species']
    )

    ops = td.convert_records(record_map=RecordMap(blocks_out=record_spec))

    iris >> ops
