
"""test eval cache"""


import data_algebra
import data_algebra.eval_cache
import data_algebra.SQLite
import data_algebra.BigQuery


def test_eval_cache_hash():
    pd = data_algebra.data_model.default_data_model().pd
    d1 = pd.DataFrame({'x': [1]})
    d2 = pd.DataFrame({'x': []})
    h1 = data_algebra.eval_cache.hash_data_frame(d1)
    assert isinstance(h1, str)
    h2 = data_algebra.eval_cache.hash_data_frame(d2)
    assert h1 != h2
    h1b = data_algebra.eval_cache.hash_data_frame(d1)
    assert h1b == h1


def test_eval_cache_mk_cache_key():
    pd = data_algebra.data_model.default_data_model().pd
    d1 = pd.DataFrame({'x': [1]})
    k1 = data_algebra.eval_cache.make_cache_key(
        db_model=data_algebra.SQLite.SQLiteModel(),
        sql='SELECT * from d',
        data_map={'d': d1},
    )
    k2 = data_algebra.eval_cache.make_cache_key(
        db_model=data_algebra.BigQuery.BigQueryModel(),
        sql='SELECT * from d',
        data_map={'d': d1},
    )
    assert k1 == k1
    assert k1 != k2


def test_eval_cache_ResultCache():
    pd = data_algebra.data_model.default_data_model().pd
    d1 = pd.DataFrame({'x': [1]})
    k1 = data_algebra.eval_cache.make_cache_key(
        db_model=data_algebra.SQLite.SQLiteModel(),
        sql='SELECT * from d',
        data_map={'d': d1},
    )
    result_cache = data_algebra.eval_cache.ResultCache()
    assert not result_cache.dirty
    result_cache.store(
        db_model=data_algebra.SQLite.SQLiteModel(),
        sql='SELECT * from d',
        data_map={'d': d1},
        res=d1
    )
    back = result_cache.get(
        db_model=data_algebra.SQLite.SQLiteModel(),
        sql='SELECT * from d',
        data_map={'d': d1},
    )
    assert d1.equals(back)
    assert result_cache.dirty
    result_cache.dirty = False
    result_cache.store(
        db_model=data_algebra.SQLite.SQLiteModel(),
        sql='SELECT * from d',
        data_map={'d': d1},
        res=d1
    )
    assert not result_cache.dirty
