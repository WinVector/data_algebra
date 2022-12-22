"""
Interface for realizing the data algebra as a sequence of steps over an object.
"""


import abc
import re
from typing import Any, Dict, Iterable, List


class DataModel(abc.ABC):
    """
    Interface for realizing the data algebra as a sequence of steps over Pandas like objects.
    """

    presentation_model_name: str
    module: Any

    def __init__(self, *, presentation_model_name: str, module):
        assert isinstance(presentation_model_name, str)
        self.presentation_model_name = presentation_model_name
        self.module = module

    # data frame helpers

    @abc.abstractmethod
    def data_frame(self, arg=None):
        """
        Build a new data frame.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """

    @abc.abstractmethod
    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """
    
    @abc.abstractmethod
    def clean_copy(self, df):
        """
        Copy of data frame without indices.
        """
    
    @abc.abstractmethod
    def to_pandas(self, df):
        """
        Convert to Pandas
        """
    
    @abc.abstractmethod
    def drop_indices(self, df) -> None:
        """
        Drop indices in place.
        """

    @abc.abstractmethod
    def bad_column_positions(self, x):
        """
        Return vector indicating which entries are bad (null or nan) (vectorized).
        """

    @abc.abstractmethod
    def table_is_keyed_by_columns(self, table, *, column_names: Iterable[str]) -> bool:
        """
        Check if a table is keyed by a given list of column names.

        :param table: DataFrame
        :param column_names: list of column names
        :return: True if rows are uniquely keyed by values in named columns
        """

    @abc.abstractmethod
    def concat_rows(self, frame_list: List):
        """
        Concatenate rows from frame_list
        """

    @abc.abstractmethod
    def concat_columns(self, frame_list):
        """
        Concatenate columns from frame_list
        """

    # evaluate

    @abc.abstractmethod
    def eval(self, op, *, data_map: Dict[str, Any]):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :return: data frame result
        """
    
    # cdata transform methods

    @abc.abstractmethod
    def blocks_to_rowrecs(self, data, *, blocks_in):
        """
        Convert a block record (record spanning multiple rows) into a rowrecord (record in a single row).

        :param data: data frame to be transformed
        :param blocks_in: cdata record specification
        :return: transformed data frame
        """
    
    @abc.abstractmethod
    def rowrecs_to_blocks(
        self,
        data,
        *,
        blocks_out,
    ):
        """
        Convert rowrecs (single row records) into block records (multiple row records).

        :param data: data frame to transform.
        :param blocks_out: cdata record specification.
        :return: transformed data frame
        """
    
    # expression helpers

    @abc.abstractmethod
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """
    
    @abc.abstractmethod
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param value: column name
        :return: converted result
        """
    
    @abc.abstractmethod
    def act_on_expression(self, *, arg, values: List, op: str):
        """
        Action for a column name.

        :param arg: item we are acting on
        :param values: op arguments already converted
        :param op: operator to apply
        :return: converted result
        """


# map type name strings to data models
data_model_type_map = dict()

def default_data_model() -> DataModel:
    """Get the default (Pandas) data model"""
    global data_model_type_map
    return data_model_type_map["default_data_model"]

pandas_regexp = re.compile(r".*[^a-zA-Z]pandas[^a-zA-Z].*[^a-zA-z]dataframe[^a-zA-Z].*")
assert pandas_regexp.match("<class 'pandas.core.frame.DataFrame'>".lower()) is not None
polars_regexp = re.compile(r".*[^a-zA-Z]polars[^a-zA-Z].*[^a-zA-z]dataframe[^a-zA-Z].*")
assert polars_regexp.match("<class 'polars.internals.dataframe.frame.DataFrame'>".lower()) is not None
polars_regexp_lazy = re.compile(r".*[^a-zA-Z]polars[^a-zA-Z].*[^a-zA-z]lazyframe[^a-zA-Z].*")
assert polars_regexp_lazy.match("<class 'polars.internals.lazyframe.frame.LazyFrame'>".lower()) is not None

def lookup_data_model_for_key(key: str) -> DataModel:
    global data_model_type_map
    assert isinstance(key, str)
    key_lower = key.lower()
    if (key == "default_data_model") or (key == "default_Pandas_model") or (pandas_regexp.match(key_lower) is not None):
        import data_algebra.pandas_model  # delayed import
        data_algebra.pandas_model.register_pandas_model(key)
    elif (key == "default_Polars_model") or (polars_regexp.match(key_lower) is not None) or (polars_regexp_lazy.match(key_lower) is not None):
        import data_algebra.polars_model  # delayed import
        data_algebra.polars_model.register_polars_model(key)
    return data_model_type_map[key]

def lookup_data_model_for_dataframe(d) -> DataModel:
    key = str(type(d))
    res = lookup_data_model_for_key(key)
    assert res.is_appropriate_data_instance(d)
    return res
