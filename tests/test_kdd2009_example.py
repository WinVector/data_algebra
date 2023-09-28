
import data_algebra
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
import os
import pandas as pd

import data_algebra.test_util
import data_algebra.util
from data_algebra.test_util import equivalent_frames

have_polars = False
try:
    import polars as pl
    have_polars = True
except ModuleNotFoundError:
    pass


def test_kdd2009_example_pipeline():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data_kdd2009', 'ops.txt')) as f:
        ops_str = f.read()
    ops = eval(ops_str)
    d_test = pd.read_csv(os.path.join(dir_path, 'data_kdd2009', 'd_test.csv.gz'))
    test_processed = pd.read_csv(os.path.join(dir_path, 'data_kdd2009', 'test_processed.csv.gz'))
    transform_as_data = pd.read_csv(os.path.join(dir_path, 'data_kdd2009', 'transform_as_data.csv.gz'))
    test_by_pipeline = ops.eval({
        'd_test': d_test,
        'transform_as_data': transform_as_data,
        })
    assert equivalent_frames(
        test_by_pipeline.loc[:, test_processed.columns], 
        test_processed,
        check_row_order=True)
    if have_polars:
        test_by_pipeline_pl = ops.eval({
            'd_test': pl.DataFrame(d_test),
            'transform_as_data': pl.DataFrame(transform_as_data),
            })
        assert equivalent_frames(
            test_by_pipeline_pl.to_pandas().loc[:, test_processed.columns], 
            test_processed,
            check_row_order=True)
