
"""Cache for test evaluations"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import hashlib
import data_algebra
import data_algebra.db_model


def hash_data_frame(d) -> str:
    """
    Get a hash code representing a data frame.

    :param d: data frame
    :return: hash code as a string
    """
    data_algebra.default_data_model.is_appropriate_data_instance(d)
    hash_str = hashlib.sha256(
        data_algebra.default_data_model.pd.util.hash_pandas_object(d).values
    ).hexdigest()
    return f'{d.shape}_{list(d.columns)}_{hash_str}'


class EvalKey(NamedTuple):
    """Carry description of data transform key"""
    db_model_name: str
    sql: str
    dat_map_list: Tuple[Tuple[str, str], ...]


def make_cache_key(
        *,
        db_model: data_algebra.db_model.DBModel,
        sql: str,
        data_map: Dict[str, Any],
):
    """
    Create an immutable, hashable key.
    """
    assert isinstance(db_model, data_algebra.db_model.DBModel)
    assert isinstance(sql, str)
    assert isinstance(data_map, dict)
    data_map_keys = list(data_map.keys())
    data_map_keys.sort()
    for k in data_map_keys:
        assert isinstance(k, str)
        assert data_algebra.default_data_model.is_appropriate_data_instance(data_map[k])
    return EvalKey(
        db_model_name=str(db_model),
        sql=sql,
        dat_map_list=tuple([(k, hash_data_frame(data_map[k])) for k in data_map_keys])
    )


class ResultCache:
    """Cache for test results. Maps keys to data frames."""
    dirty: bool
    data_cache: Optional[Dict[str, Any]]
    result_cache: Dict[EvalKey, Any]

    def __init__(self):
        self.dirty = False
        self.data_cache = dict()
        self.result_cache = dict()

    def get(self,
            *,
            db_model: data_algebra.db_model.DBModel,
            sql: str,
            data_map: Dict[str, Any]):
        """get result from cache, raise KeyError if not present"""
        k = make_cache_key(
            db_model=db_model,
            sql=sql,
            data_map=data_map)
        res = self.result_cache[k]
        assert data_algebra.default_data_model.is_appropriate_data_instance(res)
        return res.copy()

    def store(self,
                *,
                db_model: data_algebra.db_model.DBModel,
                sql: str,
                data_map: Dict[str, Any],
                res) -> None:
        """Store result to cache, mark dirty if change."""
        assert data_algebra.default_data_model.is_appropriate_data_instance(res)
        op_key = make_cache_key(
            db_model=db_model,
            sql=sql,
            data_map=data_map)
        try:
            previous = self.result_cache[op_key]
            if previous.equals(res):
                return
        except KeyError:
            pass
        self.dirty = True
        self.result_cache[op_key] = res.copy()
        # values saved for debugging
        if self.data_cache is not None:
            for d in (list(data_map.values()) + [res]):
                d_key = hash_data_frame(d)
                # assuming no spurious key collisions
                if d_key not in self.data_cache.keys():
                    self.data_cache[d_key] = d.copy()
