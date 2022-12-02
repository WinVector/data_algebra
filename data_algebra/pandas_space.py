


from typing import Optional, Set
import data_algebra.data_model
import data_algebra.data_ops
import data_algebra.data_space
import data_algebra.pandas_model


class PandasSpace(data_algebra.data_space.DataSpace):
    """
    A data space as a map of mapped pandas data frames.
    """
    def __init__(
        self, 
        data_model: Optional[data_algebra.data_model.DataModel] = None,
        *,
        narrow: bool = False,
        ) -> None:
        """
        Build an isolated execution space. Good for enforcing different Pandas adaptors or alternatives.

        :param data_model: execution model
        :param narrow: if True strictly narrow columns.
        """
        super().__init__()
        assert isinstance(narrow, bool)
        if data_model is None:
            data_model = data_algebra.pandas_model.default_data_model
        assert isinstance(data_model, data_algebra.data_model.DataModel)
        self.data_model = data_model
        self.narrow = narrow
        self.data_map = dict()
        self.n_tmp = 0

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
        assert self.data_model.is_appropriate_data_instance(value)
        if not allow_overwrite:
            assert key not in self.data_map.keys()
        self.data_map[key] = value
        return key
    
    def remove(self, key: str) -> None:
        """
        Remove value from data space.

        :param key: key to remove
        """
        assert isinstance(key, str)
        del self.data_map[key]

    def keys(self) -> Set[str]:
        """
        Return keys
        """
        return set(self.data_map.keys())
    
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
        return return_data_model.data_frame(self.data_map[key])

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
            assert key not in self.data_map.keys()
        value = ops.eval(data_map=self.data_map, data_model=self.data_model, narrow=self.narrow)
        assert self.data_model.is_appropriate_data_instance(value)
        self.data_map[key] = value
        return data_algebra.data_ops.describe_table(value, table_name=key)

    def describe(self, key: str) -> data_algebra.data_ops.TableDescription:
        """
        Retrieve a table description from the DataSpace.

        :param key: key
        :return: data description
        """
        assert isinstance(key, str)
        d = self.data_map[key]
        return data_algebra.data_ops.describe_table(d, table_name=key)

