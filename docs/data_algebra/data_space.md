Module data_algebra.data_space
==============================

Classes
-------

`DataSpace()`
:   Class modeling a space of data keyed by strings, with specified semantics.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.data_model_space.DataModelSpace
    * data_algebra.db_space.DBSpace

    ### Methods

    `close(self) ‑> None`
    :

    `describe(self, key: str) ‑> data_algebra.data_ops.TableDescription`
    :   Retrieve a table description from the DataSpace.
        
        :param key: key
        :return: data description

    `execute(self, ops: data_algebra.data_ops.ViewRepresentation, *, key: Optional[str] = None, allow_overwrite: bool = False) ‑> data_algebra.data_ops.TableDescription`
    :   Execute ops in data space, saving result as a side effect and returning a reference.
        
        :param ops: data algebra operator dag.
        :param key: name for result
        :param allow_overwrite: if True allow table replacement
        :return: data key

    `insert(self, *, key: Optional[str] = None, value, allow_overwrite: bool = True) ‑> data_algebra.data_ops.TableDescription`
    :   Insert value into data space for key.
        
        :param key: key
        :param value: data
        :param allow_overwrite: if True, allow table replacement
        :return: table description

    `keys(self) ‑> Set[str]`
    :   Return keys

    `remove(self, key: str) ‑> None`
    :   Remove value from data space.
        
        :param key: key to remove

    `retrieve(self, key: str)`
    :   Retrieve a table value from the DataSpace.
        
        :param key: key
        :return: data value