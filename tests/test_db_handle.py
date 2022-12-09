
"""Test db handle services"""

import tempfile

import data_algebra
import data_algebra.test_util
import data_algebra.BigQuery
from data_algebra.data_ops import descr, TableDescription

import pytest


def test_db_handle_services_to_csv():
    pd = data_algebra.data_model.default_data_model().pd
    db_handles = data_algebra.test_util.get_test_dbs()
    d = pd.DataFrame({'x': [1.0, 2.0], 'y': ['a', 'b']})
    td = descr(d=d)
    for h in db_handles:
        h.insert_table(d, table_name='d', allow_overwrite=True)
        h.drop_table('d')
        h.insert_table(d, table_name='d', allow_overwrite=False)
        with pytest.raises(ValueError):
            h.insert_table(d, table_name='d', allow_overwrite=False)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=True) as tmp:
            h.query_to_csv(td, res_name=tmp.name)
            d_back = pd.read_csv(tmp.name)
            assert data_algebra.test_util.equivalent_frames(d, d_back)
        h.drop_table('d')
    # clean up
    for h in db_handles:
        h.close()


def test_db_handle_servies_bq_desribe():
    pd = data_algebra.data_model.default_data_model().pd
    d = pd.DataFrame({'x': [1.0, 2.0], 'y': ['a', 'b']})
    td = descr(d=d)
    if data_algebra.test_util.test_BigQuery:
        h = data_algebra.BigQuery.example_handle()
        tdb = h.insert_table(d, table_name='d', allow_overwrite=True)
        d_back = h.read_query(td)
        data_algebra.test_util.equivalent_frames(d, d_back)
        data_catalog, data_schema = h.db_model.table_prefix.split('.')
        bqd = h.describe_bq_table(
            table_catalog=data_catalog,
            table_schema=data_schema,
            table_name='d',
            row_limit=7
            )
        assert isinstance(bqd, TableDescription)
        assert set(bqd.column_names) == set(td.column_names)
        assert data_algebra.test_util.equivalent_frames(bqd.head, d)
        h.close()
