
import abc
from typing import Optional, Set
import data_algebra.data_model
import data_algebra.data_ops


class DataSpace(abc.ABC):
    """
    Class modeling a space of data keyed by strings, with specified semantics.
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def insert(self, *, key: Optional[str] = None, value, allow_overwrite: bool = True) -> data_algebra.data_ops.TableDescription:
        """
        Insert value into data space for key.

        :param key: key
        :param value: data
        :param allow_overwrite: if True, allow table replacement
        :return: table description
        """
    
    @abc.abstractmethod
    def remove(self, key: str) -> None:
        """
        Remove value from data space.

        :param key: key to remove
        """
    
    @abc.abstractmethod
    def keys(self) -> Set[str]:
        """
        Return keys
        """
    
    @abc.abstractmethod
    def retrieve(self, key: str):
        """
        Retrieve a table value from the DataSpace.

        :param key: key
        :return: data value
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def describe(self, key: str) -> data_algebra.data_ops.TableDescription:
        """
        Retrieve a table description from the DataSpace.

        :param key: key
        :return: data description
        """

    def close(self) -> None:
        pass

    # context management
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
