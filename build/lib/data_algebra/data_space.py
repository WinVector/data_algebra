
import abc
from typing import Optional, Set
import data_algebra.data_model
import data_algebra.data_ops
from data_algebra.shift_pipe_action import ShiftPipeAction


class DataSpace(ShiftPipeAction):
    """
    Class modeling a space of data keyed by strings, with specified semantics.
    """
    def __init__(self) -> None:
        ShiftPipeAction.__init__(self)

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
    
    def act_on(self, b, *, correct_ordered_first_call: bool = False):
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            return self.execute(b)
        if correct_ordered_first_call and isinstance(b, ShiftPipeAction):
            return b.act_on(self, correct_ordered_first_call=False)  # fall back
        raise TypeError(f"inappropriate type to DataSpace.act_on(): {type(b)}")

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
