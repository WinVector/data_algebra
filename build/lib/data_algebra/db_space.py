
from typing import Optional, Set
import data_algebra.data_model
import data_algebra.data_ops
import data_algebra.data_space
import data_algebra.db_model
import data_algebra.SQLite


class DBSpace(data_algebra.data_space.DataSpace):
    """
    A data space implemented in a database.
    """
    def __init__(
        self, 
        db_handle: Optional[data_algebra.db_model.DBHandle] = None,
        *,
        drop_tables_on_close: bool = False
        ) -> None:
        super().__init__()
        assert isinstance(drop_tables_on_close, bool)
        self.drop_tables_on_close = drop_tables_on_close
        self.close_handle = False
        if db_handle is None:
            db_handle = data_algebra.SQLite.example_handle()
            self.close_handle = True
            self.drop_tables_on_close = False  # disposing of db anyway
        assert isinstance(db_handle, data_algebra.db_model.DBHandle)
        self.db_handle = db_handle
        self.n_tmp = 0
        self.description_map = dict()
        self.eligable_for_auto_drop_list = set()
    
    def model_table(self, key: str, *, eligible_for_auto_drop: bool = False) -> data_algebra.data_ops.TableDescription:
        """
        Insert existing table record into data space model.

        :param key: table name and key.
        :return: table description
        """
        assert isinstance(key, str)
        descr = self.db_handle.describe_table(key)
        self.description_map[key] = descr
        if eligible_for_auto_drop:
            self.eligable_for_auto_drop_list.add(key)
        return descr

    def insert(self, *, key: Optional[str] = None, value, allow_overwrite: bool = True) -> data_algebra.data_ops.TableDescription:
        """
        Insert value into data space for key.

        :param key: key
        :param value: data
        :param allow_overwrite: if True, allow table replacement
        :return: table description
        """
        if key is None:
            self.n_tmp = self.n_tmp + 1
            key = f"da_temp_{self.n_tmp}"
        assert isinstance(key, str)
        assert isinstance(allow_overwrite, bool)
        if not allow_overwrite:
            assert key not in self.description_map.keys()
        self.db_handle.insert_table(value, table_name=key, allow_overwrite=allow_overwrite)
        return self.model_table(key, eligible_for_auto_drop=True)
    
    def remove(self, key: str) -> None:
        """
        Remove value from data space.

        :param key: key to remove
        """
        assert isinstance(key, str)
        del self.description_map[key]
        if key in self.eligable_for_auto_drop_list:
            self.eligable_for_auto_drop_list.remove(key)
        self.db_handle.drop_table(key)
    
    def keys(self) -> Set[str]:
        """
        Return keys
        """
        return set(self.description_map.keys())
    
    def retrieve(self, key: str):
        """
        Retrieve a table value from the DataSpace.

        :param key: key
        :return: data value
        """
        assert isinstance(key, str)
        descr = self.description_map[key]  # force check table is known
        d = self.db_handle.read_table(key)
        return d

    def execute(
        self, 
        ops: data_algebra.data_ops.ViewRepresentation, 
        *, 
        key: Optional[str] = None,
        allow_overwrite: bool = False,
        ) -> data_algebra.data_ops.TableDescription:
        """
        Execute ops in data space, saving result as a side effect and returning a reference.

        :param ops: data algebra operator dag.
        :param key: name for result
        :param allow_overwrite: if True allow table replacement
        :return: data key
        """
        if key is None:
            self.n_tmp = self.n_tmp + 1
            key = f"da_temp_{self.n_tmp}"
        assert isinstance(key, str)
        assert isinstance(allow_overwrite, bool)
        if key in self.description_map.keys():
            assert allow_overwrite
            self.remove(key)
        descr = self.db_handle.create_table(table_name=key, q=ops)
        self.description_map[key] = descr
        self.eligable_for_auto_drop_list.add(key)
        return descr

    def describe(self, key: str) -> data_algebra.data_ops.TableDescription:
        """
        Retrieve a table description from the DataSpace.

        :param key: key
        :return: data description
        """
        assert isinstance(key, str)
        return self.description_map[key]

    def close(self) -> None:
        if self.drop_tables_on_close:
            key_list = list(self.eligable_for_auto_drop_list)
            for key in key_list:
                self.remove(key)
        if self.close_handle:
            self.db_handle.close()
        self.db_handle = None
        self.description_map = None
