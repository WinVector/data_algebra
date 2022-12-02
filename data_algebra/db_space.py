
from typing import Optional, Set
import data_algebra.data_model
import data_algebra.data_ops
import data_algebra.data_space
import data_algebra.pandas_model
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
        self.known_keys = set()

    def insert(self, *, key: Optional[str] = None, value, allow_overwrite: bool = True) -> str:
        """
        Insert value into data space for key.

        :param key: key
        :param value: data
        :param allow_overwrite: if True, allow table replacement
        :return: None
        """
        if key is None:
            self.n_tmp = self.n_tmp + 1
            key = f"da_temp_{self.n_tmp}"
        assert isinstance(key, str)
        assert isinstance(allow_overwrite, bool)
        if not allow_overwrite:
            assert key not in self.known_keys
        self.known_keys.add(key)
        self.db_handle.insert_table(value, table_name=key, allow_overwrite=allow_overwrite)
        return key
    
    def remove(self, key: str) -> None:
        """
        Remove value from data space.

        :param key: key to remove
        """
        assert isinstance(key, str)
        self.known_keys.remove(key)
        self.db_handle.drop_table(key)
    
    def keys(self) -> Set[str]:
        """
        Return keys
        """
        return self.known_keys.copy()
    
    def retrieve(self, key: str, *, return_data_model: Optional[data_algebra.data_model.DataModel] = None):
        """
        Retrieve a table value from the DataSpace.

        :param key: key
        :param return_data_model: data model for return type
        :return: data value
        """
        assert isinstance(key, str)
        if return_data_model is None:
            return_data_model = data_algebra.pandas_model.default_data_model
        assert key in self.known_keys
        d = self.db_handle.read_table(key)
        d = return_data_model.data_frame(d)
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
        if not allow_overwrite:
            assert key not in self.known_keys
        else:
            if key in self.known_keys:
                self.db_handle.drop_table(key)
        self.known_keys.add(key)
        return self.db_handle.create_table(table_name=key, q=ops)

    def describe(self, key: str) -> data_algebra.data_ops.TableDescription:
        """
        Retrieve a table description from the DataSpace.

        :param key: key
        :return: data description
        """
        assert isinstance(key, str)
        assert key in self.known_keys
        descr = self.db_handle.describe_table(key)
        return descr

    def close(self) -> None:
        if self.drop_tables_on_close:
            for key in self.keys():
                self.remove(key)
        if self.close_handle:
            self.db_handle.close()
        self.db_handle = None
        self.known_keys = None
