Module data_algebra.db_space
============================

Classes
-------

`DBSpace(db_handle: Optional[data_algebra.db_model.DBHandle] = None, *, drop_tables_on_close: bool = False)`
:   A data space implemented in a database.

    ### Ancestors (in MRO)

    * data_algebra.data_space.DataSpace
    * abc.ABC

    ### Methods

    `close(self) ‑> None`
    :

    `model_table(self, key: str, *, eligible_for_auto_drop: bool = False) ‑> data_algebra.data_ops.TableDescription`
    :   Insert existing table record into data space model.
        
        :param key: table name and key.
        :return: table description