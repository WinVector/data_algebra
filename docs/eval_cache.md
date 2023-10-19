Module data_algebra.eval_cache
==============================
Cache for test evaluations

Functions
---------

    
`hash_data_frame(d) ‑> str`
:   Get a hash code representing a data frame.
    
    :param d: data frame
    :return: hash code as a string

    
`make_cache_key(*, db_model: data_algebra.db_model.DBModel, sql: str, data_map: Dict[str, Any])`
:   Create an immutable, hashable key.

Classes
-------

`EvalKey(db_model_name: str, sql: str, dat_map_list: Tuple[Tuple[str, str], ...])`
:   Carry description of data transform key

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `dat_map_list: Tuple[Tuple[str, str], ...]`
    :   Alias for field number 2

    `db_model_name: str`
    :   Alias for field number 0

    `sql: str`
    :   Alias for field number 1

`ResultCache()`
:   Cache for test results. Maps keys to data frames.

    ### Class variables

    `data_cache: Optional[Dict[str, Any]]`
    :

    `dirty: bool`
    :

    `result_cache: Dict[data_algebra.eval_cache.EvalKey, Any]`
    :

    ### Methods

    `get(self, *, db_model: data_algebra.db_model.DBModel, sql: str, data_map: Dict[str, Any])`
    :   get result from cache, raise KeyError if not present

    `store(self, *, db_model: data_algebra.db_model.DBModel, sql: str, data_map: Dict[str, Any], res) ‑> None`
    :   Store result to cache, mark dirty if change.