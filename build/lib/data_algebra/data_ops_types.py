"""
Type defs for data operations.
"""

import abc
from typing import Dict, Optional, Set, NamedTuple

import data_algebra.expr_rep
import data_algebra.cdata
import data_algebra.OrderedSet


class MethodUse(NamedTuple):
    """Carry description of a method use"""
    op_name: str
    is_project: bool = False
    is_windowed: bool = False
    is_ordered: bool = False


class OperatorPlatform(abc.ABC):
    """Abstract class representing ability to apply data_algebra operations."""

    node_name: str

    def __init__(self, *, node_name: str):
        assert isinstance(node_name, str)
        self.node_name = node_name

    @abc.abstractmethod
    def eval(self, data_map, *, data_model=None, narrow=True):
        """
        Evaluate operators with respect to Pandas data frames.

        :param data_map: map from table names to data frames
        :param data_model: adaptor to data dialect (Pandas for now)
        :param narrow: logical, if True don't copy unexpected columns
        :return: table result
        """

    # noinspection PyPep8Naming
    @abc.abstractmethod
    def transform(self, X, *, data_model=None, narrow: bool = True, check_incoming_data_constraints: bool = False):
        """
        apply self to data frame X, may or may not commute with composition

        :param X: input data frame
        :param data_model: implementation to use
        :param narrow: logical, if True don't copy unexpected columns
        :param check_incoming_data_constraints: logical, if True check incoming data meets constraints
        :return: transformed data frame
        """

    # noinspection PyPep8Naming
    def act_on(self, X, *, data_model=None):
        """
        apply self to data frame X, must commute with composition

        :param X: input data frame
        :param data_model implementation to use
        :return: transformed dataframe
        """
        return self.transform(X=X, data_model=data_model, narrow=False, check_incoming_data_constraints=True)

    @abc.abstractmethod
    def apply_to(self, a, *, target_table_key=None):
        """
        apply self to operator DAG a

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """

    def __rrshift__(self, other):  # override other >> self
        """
        override other >> self
        self.apply_to/act_on(other)

        :param other:
        :return:
        """
        if isinstance(other, OperatorPlatform):
            return self.apply_to(other)
        return self.act_on(other)

    def __rshift__(self, other):  # override self >> other
        """
        override self >> other
        other.apply_to(self)

        :param other:
        :return:
        """
        # can't use type >> type if only __rrshift__ is defined (must have __rshift__ in this case)
        if isinstance(other, OperatorPlatform):
            return other.apply_to(self)
        raise TypeError("unexpected type: " + str(type(other)))

    # composition
    def add(self, other):
        """
        other.apply_to(self)

        :param other:
        :return:
        """
        return other.apply_to(self)

    # imitate a method
    def use(self, user_function, *args, **kwargs):
        """
        Apply f as if it was a method on this chain.
        Defined as return f(self, *args, **kwargs).

        :param user_function: function to apply
        :param args: additional positional arguments
        :param kwargs: additional keyword arguments
        """
        return user_function(self, *args, **kwargs)

    # convenience

    @abc.abstractmethod
    def ex(self, *, data_model=None, narrow=True, allow_limited_tables=False):
        """
        Evaluate operators with respect to Pandas data frames already stored in the operator chain.

        :param data_model: adaptor to data dialect (Pandas for now)
        :param narrow: logical, if True don't copy unexpected columns
        :param allow_limited_tables: logical, if True allow execution on non-complete tables
        :return: table result
        """

    # characterization

    @abc.abstractmethod
    def get_tables(self):
        """
        Get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent.
        """

    # info

    @abc.abstractmethod
    def columns_produced(self):
        """
        Return list of columns produced by pipeline.
        """

    @abc.abstractmethod
    def methods_used(self) -> Set[MethodUse]:
        """
        Return set of methods used.
        """

    # query generation

    @abc.abstractmethod
    def to_near_sql_implementation_(self, db_model, *, using, temp_id_source):
        """
        Convert to NearSQL as a step in converting to a SQL string. Internal method.

        :param db_model: database model
        :param using: optional column restriction
        :param temp_id_source: temporary id source.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def extend_parsed_(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        """
        Add new derived columns, can replace existing columns for parsed operations. Internal method.

        :param parsed_ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification.
        :param order_by: optional window ordering specification.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def project_parsed_(self, parsed_ops=None, *, group_by=None):
        """
        Compute projection, or grouped calculation for parsed ops. Internal method.

        :param parsed_ops: dictionary of calculations to perform, can be empty.
        :param group_by: optional group key(s) specification.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def select_rows_parsed_(self, parsed_expr):
        """
        Select rows matching parsed expr criteria. Internal method.

        :param parsed_expr: logical expression specifying desired rows.
        :return: compose operator directed acyclic graph
        """

    # main API

    @abc.abstractmethod
    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        """
        Add new derived columns, can replace existing columns.

        :param ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification.
        :param order_by: optional window ordering specification.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def project(self, ops=None, *, group_by=None):
        """
        Compute projection, or grouped calculation.

        :param ops: dictionary of calculations to perform, can be empty.
        :param group_by: optional group key(s) specification.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def natural_join(self, b, *, by, jointype, check_all_common_keys_in_by=False):
        """
        Join self (left) results with b (right).

        :param b: second or right table to join to.
        :param by: list of join key column names.
        :param jointype: name of join type.
        :param check_all_common_keys_in_by: if True, raise if any non-key columns are common to tables.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        """
        Union or concatenate rows of self with rows of b.

        :param b: table with rows to add.
        :param id_column: optional name for new source identification column.
        :param a_name: source annotation to use for self/a.
        :param b_name: source annotation to use for b.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def select_rows(self, expr):
        """
        Select rows matching expr criteria.

        :param expr: logical expression specifying desired rows.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def drop_columns(self, column_deletions):
        """
        Remove columns from result.

        :param column_deletions: list of columns to remove.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def select_columns(self, columns):
        """
        Narrow to columns in result.

        :param columns: list of columns to keep.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def rename_columns(self, column_remapping: Dict[str, str]):
        """
        Rename columns.

        :param column_remapping: dictionary mapping new column names to old column sources (same
                                 direction as extend).
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def order_rows(self, columns, *, reverse=None, limit=None):
        """
        Order rows by column set.

        :param columns: columns to order by.
        :param reverse: optional columns to reverse order.
        :param limit: optional row limit to impose on result.
        :return: compose operator directed acyclic graph
        """

    @abc.abstractmethod
    def convert_records(self, record_map: data_algebra.cdata.RecordMap):
        """
        Apply a record mapping taking blocks_in to blocks_out structures.

        :param record_map: data_algebra.cdata.RecordMap transform specification
        :return: compose operator directed acyclic graph
        """

    def map_records(
            self,
            blocks_in: Optional[data_algebra.cdata.RecordSpecification] = None,
            blocks_out: Optional[data_algebra.cdata.RecordSpecification] = None):
        """
        Apply a record mapping taking blocks_in to blocks_out structures.

        :param blocks_in: Optional incoming record specification
        :param blocks_out: Optional incoming record specification
        :return: compose operator directed acyclic graph
        """
        if (blocks_in is None) and (blocks_out is None):
            return self  # NO-OP, return source ops
        return self.convert_records(
            data_algebra.cdata.RecordMap(blocks_in=blocks_in, blocks_out=blocks_out),
        )

    # sklearn step style interface

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        """sklearn interface, fit() is a noop"""
        pass

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        """sklearn interface, fit() is a noop"""
        self.fit(X, y=y)
        return self.transform(X)

    def get_feature_names(self, input_features=None):
        """sklearn interface, return columns"""
        cp = self.columns_produced()
        if input_features is not None:
            cp_set = set(cp)
            cp = cp + [f for f in input_features if f not in cp_set]
        return cp

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=False):
        """sklearn interface, noop"""
        return dict()

    def set_params(self, **params):
        """sklearn interface, noop"""
        pass

    # noinspection PyPep8Naming
    def inverse_transform(self, X):
        """sklearn interface, raise"""
        raise TypeError("data_algebra does not support inverse_transform")
