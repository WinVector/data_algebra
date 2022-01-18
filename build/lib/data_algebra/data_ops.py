"""
Realization of data operations.
"""


import abc
import collections
from typing import Iterable, Set, Dict, List, Optional, Tuple, Union
import numbers
import re

import numpy

import data_algebra
import data_algebra.expr_parse
import data_algebra.flow_text
import data_algebra.data_model
import data_algebra.pandas_model
import data_algebra.expr_rep
from data_algebra.data_ops_types import *
import data_algebra.data_ops_utils
import data_algebra.near_sql
from data_algebra.OrderedSet import (
    OrderedSet,
    ordered_intersect,
    ordered_union,
)
import data_algebra.util


_have_black = False
try:
    # noinspection PyUnresolvedReferences
    import black

    _have_black = True
except ImportError:
    pass


# noinspection PyBroadException
def pretty_format_python(python_str: str, *, black_mode=None) -> str:
    """
    Format Python code, using black.

    :param python_str: Python code
    :param black_mode: options for black
    :return: formatted Python code
    """
    assert isinstance(python_str, str)
    formatted_python = python_str
    if _have_black:
        try:
            if black_mode is None:
                black_mode = black.FileMode()
            formatted_python = black.format_str(python_str, mode=black_mode)
        except Exception:
            pass
    return formatted_python


def _assert_tables_defs_consistent(tm1: Dict, tm2: Dict):
    common_keys = set(tm1.keys()).intersection(tm2.keys())
    for k in common_keys:
        t1 = tm1[k]
        t2 = tm2[k]
        if not t1.same_table_description_(t2):
            raise ValueError("Table " + k + " has two incompatible representations")


def _work_col_group_arg(arg, *, arg_name: str, columns: Iterable[str]):
    """convert column list to standard form"""
    if arg is None:
        return []
    elif isinstance(arg, str):
        assert arg in set(columns)
        return [arg]
    elif isinstance(arg, Iterable):
        res = list(arg)
        assert len(res) == len(set(res))
        col_set = set(columns)
        assert numpy.all([col in col_set for col in arg])
        return res
    elif arg == 1:
        return 1
    assert ValueError(f"Need {arg_name} to be a list of strings or 1, got {arg}")


class ViewRepresentation(OperatorPlatform, abc.ABC):
    """Structure to represent the columns of a query or a table.
       Abstract base class."""

    column_names: Tuple[str, ...]
    sources: Tuple[
        "ViewRepresentation", ...
    ]  # https://www.python.org/dev/peps/pep-0484/#forward-references

    def __init__(
        self,
        column_names: Iterable[str],
        *,
        sources: Optional[Iterable["ViewRepresentation"]] = None,
        node_name: str,
    ):
        # don't let instances masquerade as iterables
        assert not isinstance(column_names, str)
        assert not isinstance(sources, OperatorPlatform)
        if not isinstance(column_names, tuple):
            column_names = tuple(column_names)
        assert len(column_names) > 0
        for v in column_names:
            assert isinstance(v, str)
        assert len(column_names) == len(set(column_names))
        self.column_names = column_names
        if sources is None:
            sources = ()
        else:
            if not isinstance(sources, tuple):
                sources = tuple(sources)
        for si in sources:
            assert isinstance(si, ViewRepresentation)
        self.sources = sources
        OperatorPlatform.__init__(self, node_name=node_name)

    def column_map(self) -> collections.OrderedDict:
        """
        Build a map of column names to ColumnReferences
        """
        res = collections.OrderedDict()
        for ci in self.column_names:
            res[ci] = data_algebra.expr_rep.ColumnReference(self, ci)
        return res

    def merged_rep_id(self) -> str:
        """
        String key for lookups.
        """
        return "ops+ " + str(id(self))

    # convenience

    def ex(self, *, data_model=None, narrow=True, allow_limited_tables=False):
        """
        Evaluate operators with respect to Pandas data frames already stored in the operator chain.

        :param data_model: adaptor to data dialect (Pandas for now)
        :param narrow: logical, if True don't copy unexpected columns
        :param allow_limited_tables: logical, if True allow execution on non-complete tables
        :return: table result
        """
        tables = self.get_tables()
        data_map = dict()
        for tv in tables.values():
            assert isinstance(tv, TableDescription)
            assert tv.head is not None
            if len(tables) > 1:
                assert tv.table_name_was_set_by_user
            if not allow_limited_tables:
                assert tv.nrows == tv.head.shape[0]
            data_map[tv.table_name] = tv.head
        return self.eval(data_map=data_map, data_model=data_model, narrow=narrow)

    # characterization

    def get_tables(self):
        """Get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent."""

        tables = dict()
        # eliminate recursions by stepping through sources
        visit_stack = list()
        visit_stack.append(self)
        while len(visit_stack) > 0:
            cursor = visit_stack.pop()
            if isinstance(cursor, TableDescription):
                k = cursor.key
                v = cursor
                if k in tables.keys():
                    if not v.same_table_description_(tables[k]):
                        raise ValueError(
                            "Table " + k + " has two incompatible representations"
                        )
                else:
                    tables[k] = v
            else:
                for s in cursor.sources:
                    visit_stack.append(s)
        return tables

    @abc.abstractmethod
    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """

    def methods_used(self) -> Set[MethodUse]:
        """
        Return set of methods used.
        """
        res: Set[MethodUse] = set()
        self.get_method_uses_(res)
        return res

    def get_method_uses_(self, methods_seen: Set[MethodUse]) -> None:
        """
        Implementation of get methods_used(), internal method.

        :params methods_seen: set to collect results in.
        :return: None
        """
        for s in self.sources:
            s.get_method_uses_(methods_seen)

    def columns_produced(self):
        """Return list of columns produced by operator dag."""
        return list(self.column_names)

    def columns_used_implementation_(self, *, using, columns_currently_using_records):
        """Implementation of columns used calculation, internal method."""
        self_merged_rep_id = self.merged_rep_id()
        try:
            crec = columns_currently_using_records[self_merged_rep_id]
        except KeyError:
            crec = set()
            columns_currently_using_records[self_merged_rep_id] = crec
        if using is None:
            crec.update(self.column_names)
        else:
            unknown = set(using) - set(self.column_names)
            if len(unknown) > 0:
                raise ValueError("asked for unknown columns: " + str(unknown))
            crec.update(using)
        cu_list = self.columns_used_from_sources(crec.copy())
        for i in range(len(self.sources)):
            self.sources[i].columns_used_implementation_(
                using=cu_list[i],
                columns_currently_using_records=columns_currently_using_records,
            )

    def columns_used(self, *, using=None):
        """Determine which columns are used from source tables."""

        tables = self.get_tables()
        columns_currently_using_records = {
            v.merged_rep_id(): set() for v in tables.values()
        }
        self.columns_used_implementation_(
            using=using, columns_currently_using_records=columns_currently_using_records
        )
        columns_used = dict()
        for k in tables.keys():
            ti = tables[k]
            vi = columns_currently_using_records[ti.merged_rep_id()]
            columns_used[k] = vi.copy()
        return columns_used

    def forbidden_columns(self, *, forbidden: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
        """
        Determine which columns should not be in source tables
        (were not in declared structure, and interfere with column production).

        :param forbidden: optional incoming forbids.
        :return: dictionary operator keys to forbidden sets.
        """
        if forbidden is None:
            forbidden = set()
        res: Dict[str, Set[str]] = dict()
        for source in self.sources:
            forbidden_i = source.forbidden_columns(forbidden=forbidden)
            for (tk, f) in forbidden_i.items():
                try:
                    have = res[tk]
                except KeyError:
                    have = set()
                    res[tk] = have
                have.update(f)
        return res

    # printing

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations. Internal method, allows skipping of sources.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        return "ViewRepresentation(" + self.column_names.__repr__() + ")"

    # noinspection PyBroadException
    def to_python(self, *, indent=0, strict=True, pretty=False, black_mode=None):
        """
        Return Python source code for operations.

        :param indent: extra indent.
        :param strict: if False allow eliding of columns names and other long structures.
        :param pretty: if True re-format result with black.
        :param black_mode: black formatter parameters.
        """
        self.columns_used()  # for table consistency check/raise
        if pretty:
            strict = True
        python_str = (
            "(\n"
            + self.to_python_src_(
                indent=indent, strict=strict, print_sources=True
            )
            + "\n)\n"
        )
        if pretty:
            python_str = pretty_format_python(python_str, black_mode=black_mode)
        return python_str

    def __repr__(self):
        return self.to_python(strict=True, pretty=True)

    def __str__(self):
        return self.to_python(strict=True, pretty=True)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    @abc.abstractmethod
    def _equiv_nodes(self, other):
        """Check if immediate ops structure is equivalent, does not check child nodes"""

    def __eq__(self, other):
        if not isinstance(other, ViewRepresentation):
            return False
        if not type(self) is type(other):
            return False
        if not type(other) is type(self):
            return False
        if self.node_name != other.node_name:
            return False
        if self.column_names != other.column_names:
            return False
        if len(self.sources) != len(other.sources):
            return False
        if not self._equiv_nodes(other):
            return False
        for i in range(len(self.sources)):
            if not self.sources[i].__eq__(other.sources[i]):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # query generation

    @abc.abstractmethod
    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """

    def to_sql(
        self, db_model, *, sql_format_options=None,
    ) -> str:
        """
        Convert operator dag to SQL.

        :param db_model: database model
        :param sql_format_options: options for sql formatting
        :return: string representation of SQL query
        """
        if sql_format_options is None:
            sql_format_options = db_model.default_SQL_format_options
        return db_model.to_sql(ops=self, sql_format_options=sql_format_options,)

    # Pandas realization

    def check_constraints(self, data_model, *, strict=True):
        """
        Check tables supplied meet data consistency constraints.

        data_model: dictionary of column name lists.
        """
        self.columns_used()  # for table consistency check/raise
        forbidden = self.forbidden_columns()
        tables = self.get_tables()
        missing_tables = set(tables.keys()) - set(data_model.keys())
        if len(missing_tables) > 0:
            raise ValueError("missing required tables: " + str(missing_tables))
        for k in tables.keys():
            have = set(data_model[k])
            td = tables[k]
            missing = set(td.column_names) - have
            if len(missing) > 0:
                raise ValueError(
                    "Table " + k + " missing required columns: " + str(missing)
                )
            if strict:
                cf = set(forbidden[k])
                excess = cf.intersection(have)
                if len(excess) > 0:
                    raise ValueError(
                        "Table " + k + " has forbidden columns: " + str(excess)
                    )

    def eval(self, data_map, *, data_model=None, narrow: bool = True, check_incoming_data_constraints: bool = False):
        """
         Evaluate operators with respect to Pandas data frames.

         :param data_map: map from table names to data frames
         :param data_model: adaptor to data dialect (Pandas for now)
         :param narrow: logical, if True don't copy unexpected columns
         :param check_incoming_data_constraints: logical, if True check incoming data meets constraints
         :return: table result
         """

        if data_map is not None:
            assert isinstance(data_map, dict)
        if data_model is None:
            data_model = data_algebra.default_data_model
        assert isinstance(data_model, data_algebra.data_model.DataModel)

        if check_incoming_data_constraints and (data_map is not None) and (len(data_map) > 0):
            self.columns_used()  # for table consistency check/raise
            tables = self.get_tables()
            self.check_constraints(
                {k: x.columns for (k, x) in data_map.items()}, strict=not narrow
            )
            for k in tables.keys():
                if k not in data_map.keys():
                    raise ValueError("Required table " + k + " not in data_map")
                else:
                    if not data_model.is_appropriate_data_instance(data_map[k]):
                        raise ValueError("data_map[" + k + "] was not a usable type")
        return data_model.eval(op=self, data_map=data_map, narrow=narrow)

    # noinspection PyPep8Naming
    def transform(self, X, *, data_model=None, narrow: bool = True, check_incoming_data_constraints: bool = False):
        """
        Apply data transform to a table

        :param X: tale to apply to
        :param data_model: data model for Pandas execution
        :param narrow: logical, if True narrow number of result columns to specification
        :param check_incoming_data_constraints: logical, if True check incoming data meets constraints
        :return: transformed data frame
        """
        if data_model is None:
            data_model = data_algebra.default_data_model
        assert isinstance(data_model, data_algebra.data_model.DataModel)
        self.columns_used()  # for table consistency check/raise
        tables = self.get_tables()
        if len(tables) != 1:
            raise ValueError(
                "transform(DataFrame) can only be applied to ops-dags with only one table def"
            )
        k = [k for k in tables.keys()][0]
        # noinspection PyUnresolvedReferences
        if not data_model.is_appropriate_data_instance(X):
            raise TypeError("can not apply transform() to type " + str(type(X)))
        data_map = {k: X}
        return self.eval(
            data_map=data_map,
            data_model=data_model,
            narrow=narrow,
            check_incoming_data_constraints=check_incoming_data_constraints)

    # composition (used to eliminate intermediate order nodes)

    def is_trivial_when_intermediate_(self) -> bool:
        """
        Return if True if operator can be eliminated from interior chain.
        """
        return False

    # return table representation of self
    def as_table_description(self, table_name: str, *, qualifiers=None):
        """
        Return representation of operator as a table description.

        :param table_name: table name to use.
        :param qualifiers: db qualifiers to annotate
        """
        return TableDescription(
            table_name=table_name,
            column_names=self.column_names,
            qualifiers=qualifiers,
        )

    # implement builders for all non-initial ops types on base class
    def extend_parsed_(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        """
        Add new derived columns, can replace existing columns for parsed operations. Internal method.

        :param parsed_ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification, or 1.
        :param order_by: optional window ordering specification, or 1.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph
        """
        if (parsed_ops is None) or (len(parsed_ops) < 1):
            return self
        partition_by = _work_col_group_arg(partition_by, arg_name='partition_by', columns=self.column_names)
        order_by = _work_col_group_arg(order_by, arg_name='order_by', columns=self.column_names)
        reverse = _work_col_group_arg(reverse, arg_name='reverse', columns=self.column_names)
        assert reverse != 1
        new_cols_produced_in_calc = set([k for k in parsed_ops.keys()])
        if (partition_by != 1) and (len(partition_by) > 0):
            if len(new_cols_produced_in_calc.intersection(partition_by)) > 0:
                raise ValueError("must not change partition_by columns")
            if (order_by != 1) and len(set(partition_by).intersection(set(order_by))) > 0:
                raise ValueError("order_by and partition_by columns must be disjoint")
        if len(new_cols_produced_in_calc.intersection(order_by)) > 0:
            raise ValueError("must not change partition_by columns")
        if len(set(reverse).difference(order_by)) > 0:
            raise ValueError("all columns in reverse must be in order_by")
        if self.is_trivial_when_intermediate_():
            return self.sources[0].extend_parsed_(
                parsed_ops=parsed_ops,
                partition_by=partition_by,
                order_by=order_by,
                reverse=reverse,
            )
        # see if we can combine nodes
        if isinstance(self, ExtendNode):
            compatible_partition = (partition_by == self.partition_by) or (
                ((partition_by == 1) or (len(partition_by) <= 0))
                and ((self.partition_by == 1) or (len(self.partition_by) <= 0))
            )
            same_windowing = (
                data_algebra.expr_rep.implies_windowed(parsed_ops)
                == self.windowed_situation
            )
            if (
                compatible_partition
                and same_windowing
                and (order_by == self.order_by)
                and (reverse == self.reverse)
            ):
                new_ops = data_algebra.data_ops_utils.try_to_merge_ops(
                    self.ops, parsed_ops
                )
                if new_ops is not None:
                    return ExtendNode(
                        source=self.sources[0],
                        parsed_ops=new_ops,
                        partition_by=partition_by,
                        order_by=order_by,
                        reverse=reverse,
                    )
        # new ops
        return ExtendNode(
            source=self,
            parsed_ops=parsed_ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse,
        )

    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        """
        Add new derived columns, can replace existing columns.

        :param ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification, or 1.
        :param order_by: optional window ordering specification, or 1.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph
        """
        parsed_ops = data_algebra.expr_parse.parse_assignments_in_context(
            ops=ops, view=self
        )
        return self.extend_parsed_(
            parsed_ops=parsed_ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse,
        )

    def project_parsed_(self, parsed_ops=None, *, group_by=None):
        """
        Compute projection, or grouped calculation for parsed ops. Internal method.

        :param parsed_ops: dictionary of calculations to perform, can be empty.
        :param group_by: optional group key(s) specification.
        :return: compose operator directed acyclic graph
        """

        group_by = _work_col_group_arg(group_by, arg_name='group_by', columns=self.column_names)
        assert group_by != 1
        if ((parsed_ops is None) or (len(parsed_ops) < 1)) and (len(group_by) < 1):
            raise ValueError("project must have ops or group_by")
        new_cols_produced_in_calc = set([k for k in parsed_ops.keys()])
        if len(new_cols_produced_in_calc.intersection(group_by)):
            raise ValueError("project can not alter grouping columns")
        if self.is_trivial_when_intermediate_():
            return self.sources[0].project_parsed_(parsed_ops, group_by=group_by)
        return ProjectNode(source=self, parsed_ops=parsed_ops, group_by=group_by)

    def project(self, ops=None, *, group_by=None):
        """
        Compute projection, or grouped calculation.

        :param ops: dictionary of calculations to perform, can be empty.
        :param group_by: optional group key(s) specification.
        :return: compose operator directed acyclic graph
        """
        parsed_ops = data_algebra.expr_parse.parse_assignments_in_context(
            ops=ops, view=self
        )
        return self.project_parsed_(parsed_ops=parsed_ops, group_by=group_by)

    def natural_join(self, b, *, by, jointype, check_all_common_keys_in_by=False):
        """
        Join self (left) results with b (right).

        :param b: second or right table to join to.
        :param by: list of join key column names.
        :param jointype: name of join type.
        :param check_all_common_keys_in_by: if True, raise if any non-key columns are common to tables.
        :return: compose operator directed acyclic graph
        """
        assert isinstance(b, ViewRepresentation)
        if isinstance(by, str):
            by = [by]
        assert isinstance(jointype, str)
        if self.is_trivial_when_intermediate_():
            return self.sources[0].natural_join(b, by=by, jointype=jointype)
        return NaturalJoinNode(
            a=self,
            b=b,
            by=by,
            jointype=jointype,
            check_all_common_keys_in_by=check_all_common_keys_in_by,
        )

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        """
        Union or concatenate rows of self with rows of b.

        :param b: table with rows to add.
        :param id_column: optional name for new source identification column.
        :param a_name: source annotation to use for self/a.
        :param b_name: source annotation to use for b.
        :return: compose operator directed acyclic graph
        """
        if b is None:
            return self
        assert isinstance(b, ViewRepresentation)
        assert isinstance(id_column, (str, type(None)))
        assert isinstance(a_name, str)
        assert isinstance(b_name, str)
        if self.is_trivial_when_intermediate_():
            return self.sources[0].concat_rows(
                b, id_column=id_column, a_name=a_name, b_name=b_name
            )
        return ConcatRowsNode(
            a=self, b=b, id_column=id_column, a_name=a_name, b_name=b_name
        )

    def select_rows_parsed_(self, parsed_expr):
        """
        Select rows matching parsed expr criteria. Internal method.

        :param parsed_expr: logical expression specifying desired rows.
        :return: compose operator directed acyclic graph
        """
        if parsed_expr is None:
            return self
        if self.is_trivial_when_intermediate_():
            return self.sources[0].select_rows_parsed_(parsed_expr=parsed_expr)
        return SelectRowsNode(source=self, ops=parsed_expr)

    def select_rows(self, expr):
        """
        Select rows matching expr criteria.

        :param expr: logical expression specifying desired rows.
        :return: compose operator directed acyclic graph
        """
        if expr is None:
            return self
        if isinstance(expr, (list, tuple)):
            # convert lists to and expressions
            assert all([isinstance(vi, str) for vi in expr])
            if len(expr) < 1:
                return self
            elif len(expr) == 1:
                expr = expr[0]
            else:
                expr = " & ".join(["(" + vi + ")" for vi in expr])
        assert isinstance(expr, (str, data_algebra.expr_rep.PreTerm))
        if self.is_trivial_when_intermediate_():
            return self.sources[0].select_rows(expr)
        ops = data_algebra.expr_parse.parse_assignments_in_context(
            ops={"expr": expr}, view=self
        )

        def r_walk_expr(opv):
            """recursively inspect expression types"""
            if not isinstance(opv, data_algebra.expr_rep.Expression):
                return
            for oi in opv.args:
                r_walk_expr(oi)

        for op in ops.values():
            r_walk_expr(op)
        return self.select_rows_parsed_(parsed_expr=ops)

    def drop_columns(self, column_deletions):
        """
        Remove columns from result.

        :param column_deletions: list of columns to remove.
        :return: compose operator directed acyclic graph
        """
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        if (column_deletions is None) or (len(column_deletions) < 1):
            return self
        if self.is_trivial_when_intermediate_():
            return self.sources[0].drop_columns(column_deletions)
        return DropColumnsNode(source=self, column_deletions=column_deletions)

    def select_columns(self, columns):
        """
        Narrow to columns in result.

        :param columns: list of columns to keep.
        :return: compose operator directed acyclic graph
        """
        if isinstance(columns, str):
            columns = [columns]
        if (columns is None) or (len(columns) < 1):
            raise ValueError("must select at least one column")
        if columns == self.column_names:
            return self
        if self.is_trivial_when_intermediate_():
            return self.sources[0].select_columns(columns)
        if isinstance(self, SelectColumnsNode):
            return self.sources[0].select_columns(columns)
        if isinstance(self, DropColumnsNode):
            return self.sources[0].select_columns(columns)
        return SelectColumnsNode(source=self, columns=columns)

    def rename_columns(self, column_remapping):
        """
        Rename columns.

        :param column_remapping: dictionary mapping new column names to old column sources (same
                                 direction as extend).
        :return: compose operator directed acyclic graph
        """
        if (column_remapping is None) or (len(column_remapping) < 1):
            return self
        assert isinstance(column_remapping, dict)
        if self.is_trivial_when_intermediate_():
            return self.sources[0].rename_columns(column_remapping)
        return RenameColumnsNode(source=self, column_remapping=column_remapping)

    def order_rows(self, columns, *, reverse=None, limit=None):
        """
        Order rows by column set.

        :param columns: columns to order by.
        :param reverse: optional columns to reverse order.
        :param limit: optional row limit to impose on result.
        :return: compose operator directed acyclic graph
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(reverse, str):
            reverse = [reverse]
        if ((columns is None) or (len(columns) < 1)) and (limit is None):
            return self
        if self.is_trivial_when_intermediate_():
            return self.sources[0].order_rows(columns, reverse=reverse, limit=limit)
        return OrderRowsNode(source=self, columns=columns, reverse=reverse, limit=limit)

    def convert_records(self, record_map):
        """
        Apply a record mapping taking blocks_in to blocks_out structures.

        :param record_map: data_algebra.cdata.RecordMap transform specification
        :return: compose operator directed acyclic graph
        """
        if record_map is None:
            return self
        if self.is_trivial_when_intermediate_():
            return self.sources[0].convert_records(record_map)
        return ConvertRecordsNode(source=self, record_map=record_map)


# Could also have general query as starting ops, but don't see a lot of point to
# it until somebody needs it.


class TableDescription(ViewRepresentation):
    """
        Describe columns, and qualifiers, of a table.

       Example:
           from data_algebra.data_ops import *
           d = TableDescription(table_name='d', column_names=['x', 'y'])
           print(d)
    """

    table_name: str
    column_names: Tuple[str, ...]
    qualifiers: Dict[str, str]
    key: str
    table_name_was_set_by_user: bool

    def __init__(
        self,
        *,
        table_name: Optional[str] = None,
        column_names: Iterable[str],
        qualifiers=None,
        sql_meta=None,
        head=None,
        limit_was: Optional[int] = None,
        nrows: Optional[int] = None,
    ):
        if isinstance(column_names, str):
            column_names = (column_names, )
        else:
            column_names = tuple(column_names)  # convert to tuple from other types such as series
        ViewRepresentation.__init__(
            self, column_names=column_names, node_name="TableDescription"
        )
        if table_name is None:
            self.table_name_was_set_by_user = False
            table_name = "data_frame"
        else:
            self.table_name_was_set_by_user = True
        assert isinstance(table_name, str)
        if head is not None:
            if set([c for c in head.columns]) != set(column_names):
                raise ValueError("head.columns != column_names")
        self.head = head
        self.limit_was = limit_was
        self.sql_meta = sql_meta
        self.table_name = table_name
        self.nrows = nrows
        self.column_names = column_names
        if qualifiers is None:
            qualifiers = {}
        assert isinstance(qualifiers, dict)
        self.qualifiers = qualifiers.copy()
        self.key = ""
        if self.table_name is not None:
            self.key = self.table_name

    def same_table_description_(self, other):
        """
        Return true if other is a description of the same table. Internal method, ingores data.
        """
        if not isinstance(other, data_algebra.data_ops.TableDescription):
            return False
        if self.table_name_was_set_by_user != other.table_name_was_set_by_user:
            return False
        if self.table_name != other.table_name:
            return False
        if self.key != other.key:
            return False
        if self.column_names != other.column_names:
            return False
        if self.qualifiers != other.qualifiers:
            return False
        # ignore head and limit_was, as they are just advisory
        return True

    def merged_rep_id(self) -> str:
        """
        String key for lookups.
        """
        return "table_" + str(self.key)

    def forbidden_columns(self, *, forbidden: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
        """
        Determine which columns should not be in source tables
        (were not in declared structure, and interfere with column production).

        :param forbidden: optional incoming forbids.
        :return: dictionary operator keys to forbidden sets.
        """
        if forbidden is None:
            forbidden = set()
        return {self.key: set(forbidden)}

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        if (target_table_key is None) or (target_table_key == self.key):
            # replace table with a
            return a
        # copy self
        r = TableDescription(
            table_name=self.table_name,
            column_names=self.column_names,
            qualifiers=self.qualifiers,
        )
        return r

    def _equiv_nodes(self, other):
        if not isinstance(other, TableDescription):
            return False
        if not self.table_name == other.table_name:
            return False
        if not self.column_names == other.column_names:
            return False
        if not self.qualifiers == other.qualifiers:
            return False
        if not self.key == other.key:
            return False
        return True

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        spacer = " "
        if indent >= 0:
            spacer = "\n " + " " * indent
        column_limit = 20
        truncated = (not strict) and (column_limit < len(self.column_names))
        if truncated:
            cols_to_print = [
                self.column_names[i].__repr__() for i in range(column_limit)
            ] + ["+ " + str(len(self.column_names)) + " more"]
        else:
            cols_to_print = [c.__repr__() for c in self.column_names]
        col_text = data_algebra.flow_text.flow_text(
            cols_to_print, align_right=70 - max(0, indent), sep_width=2
        )
        col_text = [", ".join(line) for line in col_text]
        col_text = (",  " + spacer).join(col_text)
        s = (
            "TableDescription("
            + spacer
            + "table_name="
            + self.table_name.__repr__()
            + ","
            + spacer
            + "column_names=["
            + spacer
            + "  "
            + col_text
            + "]"
        )
        if len(self.qualifiers) > 0:
            s = s + "," + spacer + "qualifiers=" + self.qualifiers.__repr__()
        s = s + ")"
        return s

    def get_tables(self):
        """get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent"""
        return {self.key: self}

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        return []  # no inputs to table description

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.table_def_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )

    def __str__(self):
        rep = ViewRepresentation.__str__(self)
        if self.head is not None:
            rep = rep + "\n#\t" + str(self.head).replace("\n", "\n#\t")
        return rep

    # comparable to other table descriptions
    def __eq__(self, other):
        if not isinstance(other, TableDescription):
            return False
        return self.key.__eq__(other.key)

    def __hash__(self):
        return self.key.__hash__()


def describe_table(
    d,
    table_name=None,
    *,
    qualifiers=None,
    sql_meta=None,
    row_limit: Optional[int] = 7,
    keep_sample=True,
    keep_all=False,
) -> TableDescription:
    """
    :param d: pandas table to describe
    :param table_name: name of table
    :param qualifiers: optional, able qualifiers
    :param sql_meta: optional, sql meta information map
    :param row_limit: how many rows to sample
    :param keep_sample: logical, if True retain head of table
    :param keep_all: logical, if True retain all of table
    :return: TableDescription
    """
    assert not isinstance(d, OperatorPlatform)
    assert not isinstance(d, ViewRepresentation)
    column_names = [c for c in d.columns]
    head = None
    nrows = d.shape[0]
    if keep_all or (row_limit is None):
        row_limit = None
        head = d.copy()
        head.reset_index(drop=True, inplace=True)
    elif keep_sample:
        if nrows > row_limit:
            head = d.iloc[range(row_limit), :].copy()
        else:
            head = d.copy()
        head.reset_index(drop=True, inplace=True)
    return TableDescription(
        table_name=table_name,
        column_names=column_names,
        qualifiers=qualifiers,
        sql_meta=sql_meta,
        head=head,
        limit_was=row_limit,
        nrows=nrows,
    )


def table(d, *, table_name=None):
    """
    Capture a table for later use

    :param d: Pandas data frame to capture
    :param table_name: name for this table
    :return: a table description, with values retained
    """
    return describe_table(
        d=d,
        table_name=table_name,
        qualifiers=None,
        sql_meta=None,
        row_limit=None,
        keep_sample=True,
        keep_all=True,
    )


def descr(**kwargs):
    """
    Capture a named partial table as a description.

    :param kwargs: exactly one named table of the form table_name=table_value
    :return: a table description (not all values retained)
    """
    assert len(kwargs) == 1
    table_name = [k for k in kwargs.keys()][0]
    d = kwargs[table_name]
    return describe_table(
        d=d,
        table_name=table_name,
        qualifiers=None,
        sql_meta=None,
        row_limit=7,
        keep_sample=True,
        keep_all=False,
    )


def data(*args, **kwargs):
    """
    Capture a full table for later use. Exactly one of args/kwags can be set.

    :param args: at most one unnamed table of the form table_name=table_value
    :param kwargs: at most one named table of the form table_name=table_value
    :return: a table description, with all values retained
    """
    assert (len(args) + len(kwargs)) == 1
    if len(kwargs) == 1:
        table_name = [k for k in kwargs.keys()][0]
        d = kwargs[table_name]
        return table(d=d, table_name=table_name)
    d = args[0]
    return table(d=d, table_name=None)


class ExtendNode(ViewRepresentation):
    """
    Class representation of .extend() method/step.
    """

    windowed_situation: bool
    ordered_windowed_situation: bool
    partition_by: List[str]

    def __init__(
        self, *, source, parsed_ops, partition_by=None, order_by=None, reverse=None
    ):
        windowed_situation = data_algebra.expr_rep.implies_windowed(parsed_ops)
        ordered_windowed_situation = False
        self.ops = parsed_ops
        if partition_by is None:
            partition_by = []
        if isinstance(partition_by, numbers.Number):
            partition_by = []
            windowed_situation = True
        if isinstance(partition_by, str):
            partition_by = [partition_by]
        if len(partition_by) > 0:
            windowed_situation = True
        self.partition_by = partition_by
        if order_by is None:
            order_by = []
        if isinstance(order_by, str):
            order_by = [order_by]
        if len(order_by) > 0:
            windowed_situation = True
            ordered_windowed_situation = True
        self.windowed_situation = windowed_situation
        self.order_by = order_by
        if reverse is None:
            reverse = []
        if isinstance(reverse, str):
            reverse = [reverse]
        self.reverse = reverse
        column_names = list(source.column_names)
        consumed_cols = set()
        for (k, o) in parsed_ops.items():
            o.get_column_names(consumed_cols)
        unknown_cols = consumed_cols - set(source.column_names)
        if len(unknown_cols) > 0:
            raise KeyError("referred to unknown columns: " + str(unknown_cols))
        known_cols = set(column_names)
        for ci in parsed_ops.keys():
            if ci not in known_cols:
                column_names.append(ci)
        if len(partition_by) != len(set(partition_by)):
            raise ValueError("Duplicate name(s) in partition_by")
        if len(order_by) != len(set(order_by)):
            raise ValueError("Duplicate name(s) in order_by")
        if len(reverse) != len(set(reverse)):
            raise ValueError("Duplicate name(s) in reverse")
        unknown = set(partition_by) - known_cols
        if len(unknown) > 0:
            raise ValueError("unknown partition_by columns: " + str(unknown))
        unknown = set(order_by) - known_cols
        if len(unknown) > 0:
            raise ValueError("unknown order_by columns: " + str(unknown))
        unknown = set(reverse) - set(order_by)
        if len(unknown) > 0:
            raise ValueError("reverse columns not in order_by: " + str(unknown))
        bad_overwrite = set(parsed_ops.keys()).intersection(
            set(partition_by).union(order_by, reverse)
        )
        if len(bad_overwrite) > 0:
            raise ValueError("tried to change: " + str(bad_overwrite))
        # check op arguments are very simple: all arguments are column names
        if windowed_situation:
            source_col_set = set(source.column_names)
            for (k, opk) in parsed_ops.items():
                if not isinstance(opk, data_algebra.expr_rep.Expression):
                    raise ValueError(
                        "non-aggregated expression in windowed/partitioned extend: "
                        + "'"
                        + k
                        + "': '"
                        + str(opk)
                        + "'"
                    )
                if len(opk.args) > 1:
                    for i in range(1, len(opk.args)):
                        if not isinstance(opk.args[i], data_algebra.expr_rep.Value):
                            raise ValueError(
                                "in windowed situations only simple operators are allowed, "
                                + "'"
                                + k
                                + "': '"
                                + str(opk)
                                + "' term is too complex an expression"
                            )
                if len(opk.args) > 0:
                    if isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                        value_name = opk.args[0].column_name
                        if value_name not in source_col_set:
                            raise ValueError(value_name + " not in source column set")
                    else:
                        if not isinstance(opk.args[0], data_algebra.expr_rep.Value):
                            raise ValueError(
                                "in windowed situations only simple operators are allowed, "
                                + "'"
                                + k
                                + "': '"
                                + str(opk)
                                + "' term is too complex an expression"
                            )
                if windowed_situation and (
                    opk.op
                    in data_algebra.expr_rep.fn_names_that_contradict_windowed_situation
                ):
                    raise ValueError(
                        str(opk) + "' is not allowed in a windowed situation"
                    )
                if ordered_windowed_situation and (
                    opk.op
                    in data_algebra.expr_rep.fn_names_that_contradict_ordered_windowed_situation
                ):
                    raise ValueError(
                        str(opk) + "' is not allowed in an ordered windowed situation"
                    )
                if (not ordered_windowed_situation) and (
                    opk.op
                    in data_algebra.expr_rep.fn_names_that_imply_ordered_windowed_situation
                ):
                    raise ValueError(
                        str(opk) + "' is not allowed in not-ordered windowed situation"
                    )
        self.ordered_windowed_situation = ordered_windowed_situation
        ViewRepresentation.__init__(
            self, column_names=column_names, sources=[source], node_name="ExtendNode"
        )

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composable API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].extend_parsed_(
            parsed_ops=self.ops,
            partition_by=self.partition_by,
            order_by=self.order_by,
            reverse=self.reverse,
        )

    def _equiv_nodes(self, other):
        if not isinstance(other, ExtendNode):
            return False
        if not self.windowed_situation == other.windowed_situation:
            return False
        if not self.partition_by == other.partition_by:
            return False
        if not self.order_by == other.order_by:
            return False
        if not self.reverse == other.reverse:
            return False
        if set(self.ops.keys()) != set(other.ops.keys()):
            return False
        for k in self.ops.keys():
            if not self.ops[k].is_equal(other.ops[k]):
                return False
        return True

    def get_method_uses_(self, methods_seen: Set[MethodUse]) -> None:
        """
        Implementation of get methods_used(), internal method.

        :params methods_seen: set to collect results in.
        :return: None
        """
        for s in self.sources:
            s.get_method_uses_(methods_seen)
        method_names_seen: Set[str] = set()
        for opk in self.ops.values():
            opk.get_method_names(method_names_seen)
        for k in method_names_seen:
            methods_seen.add(
                MethodUse(
                    k,
                    is_project=False,
                    is_windowed=self.windowed_situation,
                    is_ordered=self.ordered_windowed_situation))

    def check_extend_window_fns_(self):
        """
        Confirm extend functions are all compatible with windowing in Pandas. Internal function.
        """
        window_situation = (len(self.partition_by) > 0) or (len(self.order_by) > 0)
        if window_situation:
            # check these are forms we are prepared to work with
            for (k, opk) in self.ops.items():
                if len(opk.args) > 0:
                    if len(opk.args) > 1:
                        for i in range(1, len(opk.args)):
                            if not isinstance(opk.args[i], data_algebra.expr_rep.Value):
                                raise ValueError(
                                    "window function with more than one non-value argument"
                                )
                    for i in range(len(opk.args)):
                        if not (
                            isinstance(
                                opk.args[0], data_algebra.expr_rep.ColumnReference
                            )
                            or isinstance(opk.args[0], data_algebra.expr_rep.Value)
                        ):
                            raise ValueError(
                                "window expression argument must be a column or value: "
                                + str(k)
                                + ": "
                                + str(opk)
                            )

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        if using is None:
            return [OrderedSet(self.sources[0].column_names)]
        subops = {k: op for (k, op) in self.ops.items() if k in using}
        if len(subops) <= 0:
            return [OrderedSet(self.sources[0].column_names)]
        columns_we_take = using.union(self.partition_by, self.order_by, self.reverse)
        columns_we_take = columns_we_take - subops.keys()
        for (k, o) in subops.items():
            o.get_column_names(columns_we_take)
        return [
            OrderedSet(
                [v for v in self.sources[0].column_names if v in columns_we_take]
            )
        ]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        spacer = " "
        if indent >= 0:
            spacer = "\n   " + " " * indent
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + spacer
            )
        ops = [
            k.__repr__() + ": " + opi.to_python().__repr__()
            for (k, opi) in self.ops.items()
        ]
        flowed = ("," + spacer + " ").join(ops)
        s = s + (".extend({" + spacer + " " + flowed + "}")
        if self.windowed_situation:
            if len(self.partition_by) > 0:
                s = s + "," + spacer + "partition_by=" + self.partition_by.__repr__()
            else:
                s = s + "," + spacer + "partition_by=1"
        if len(self.order_by) > 0:
            s = s + "," + spacer + "order_by=" + self.order_by.__repr__()
        if len(self.reverse) > 0:
            s = s + "," + spacer + "reverse=" + self.reverse.__repr__()
        s = s + ")"
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.extend_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class ProjectNode(ViewRepresentation):
    """
    Class representation of .project() method/step.
    """
    # TODO: should project to take an optional order for last() style calculations?
    def __init__(self, *, source, parsed_ops, group_by=None):
        self.ops = parsed_ops
        if group_by is None:
            group_by = []
        if isinstance(group_by, str):
            group_by = [group_by]
        self.group_by = group_by
        column_names = group_by.copy()
        consumed_cols = set()
        for c in group_by:
            consumed_cols.add(c)
        for (k, o) in parsed_ops.items():
            o.get_column_names(consumed_cols)
        unknown_cols = consumed_cols - set(source.column_names)
        if len(unknown_cols) > 0:
            raise KeyError("referred to unknown columns: " + str(unknown_cols))
        known_cols = set(column_names)
        for ci in parsed_ops.keys():
            if ci not in known_cols:
                column_names.append(ci)
        if len(group_by) != len(set(group_by)):
            raise ValueError("Duplicate name in group_by")
        unknown = set(group_by) - known_cols
        if len(unknown) > 0:
            raise ValueError("unknown group_by columns: " + str(unknown))
        ViewRepresentation.__init__(
            self, column_names=column_names, sources=[source], node_name="ProjectNode"
        )
        for (k, opk) in self.ops.items():
            if isinstance(opk, data_algebra.expr_rep.Expression):
                if len(opk.args) > 1:
                    raise ValueError(
                        "non-trivial aggregation expression: "
                        + str(k)
                        + ": "
                        + str(opk)
                    )
                if len(opk.args) > 0:
                    if not (
                        isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference)
                        or isinstance(opk.args[0], data_algebra.expr_rep.Value)
                    ):
                        raise ValueError(
                            "windows expression argument must be a column or value: "
                            + str(k)
                            + ": "
                            + str(opk)
                        )
                if (
                    opk.op
                    in data_algebra.expr_rep.fn_names_that_imply_ordered_windowed_situation
                ):
                    raise ValueError(str(opk) + "' is not allowed in project")
                if opk.op in data_algebra.expr_rep.fn_names_not_allowed_in_project:
                    raise ValueError(str(opk) + "' is not allowed in project")
            else:
                raise ValueError(
                    "non-aggregated expression in project: " + str(k) + ": " + str(opk)
                )
            # TODO: check op is in list of aggregators
            # Note: non-aggregators making through will be caught by table shape check

    def forbidden_columns(self, *, forbidden: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
        """
        Determine which columns should not be in source tables
        (were not in declared structure, and interfere with column production).

        :param forbidden: optional incoming forbids.
        :return: dictionary operator keys to forbidden sets.
        """
        if forbidden is None:
            forbidden = set()
        forbidden = set(forbidden).intersection(self.column_names)
        return self.sources[0].forbidden_columns(forbidden=forbidden)

    def get_method_uses_(self, methods_seen: Set[MethodUse]) -> None:
        """
        Implementation of get methods_used(), internal method.

        :params methods_seen: set to collect results in.
        :return: None
        """
        for s in self.sources:
            s.get_method_uses_(methods_seen)
        method_names_seen: Set[str] = set()
        for opk in self.ops.values():
            opk.get_method_names(method_names_seen)
        for k in method_names_seen:
            methods_seen.add(MethodUse(k, is_project=True, is_windowed=False, is_ordered=False))

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].project_parsed_(
            parsed_ops=self.ops, group_by=self.group_by
        )

    def _equiv_nodes(self, other):
        if not isinstance(other, ProjectNode):
            return False
        if not self.group_by == other.group_by:
            return False
        if set(self.ops.keys()) != set(other.ops.keys()):
            return False
        for k in self.ops.keys():
            if not self.ops[k].is_equal(other.ops[k]):
                return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        if using is None:
            subops = self.ops
        else:
            subops = {k: op for (k, op) in self.ops.items() if k in using}
        columns_we_take = set(self.group_by)
        for (k, o) in subops.items():
            o.get_column_names(columns_we_take)
        return [columns_we_take]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        spacer = " "
        if indent >= 0:
            spacer = "\n   " + " " * indent
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (
            ".project({"
            + spacer
            + " "
            + ("," + spacer + " ").join(
                [
                    k.__repr__() + ": " + opi.to_python().__repr__()
                    for (k, opi) in self.ops.items()
                ]
            )
            + "}"
        )
        if len(self.group_by) > 0:
            s = s + "," + spacer + "group_by=" + self.group_by.__repr__()
        s = s + ")"
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.project_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class SelectRowsNode(ViewRepresentation):
    """
    Class representation of .select() method/step.
    """
    expr: data_algebra.expr_rep.Expression
    decision_columns: Set[str]

    def __init__(self, source, ops):
        if len(ops) < 1:
            raise ValueError("no ops")
        if len(ops) > 1:
            raise ValueError("too many ops")
        self.ops = ops
        self.expr = ops["expr"]
        self.decision_columns = set()
        self.expr.get_column_names(self.decision_columns)
        ViewRepresentation.__init__(
            self,
            column_names=source.column_names,
            sources=[source],
            node_name="SelectRowsNode",
        )

    def get_method_uses_(self, methods_seen: Set[MethodUse]) -> None:
        """
        Implementation of get methods_used(), internal method.

        :params methods_seen: set to collect results in.
        :return: None
        """
        for s in self.sources:
            s.get_method_uses_(methods_seen)
        method_names_seen: Set[str] = set()
        self.expr.get_method_names(method_names_seen)
        for k in method_names_seen:
            methods_seen.add(MethodUse(k, is_project=False, is_windowed=False, is_ordered=False))

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].select_rows_parsed_(parsed_ops=self.ops)

    def _equiv_nodes(self, other):
        if not isinstance(other, SelectRowsNode):
            return False
        if not self.expr.is_equal(other.expr):
            return False
        if len(self.ops) != len(other.ops):
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        columns_we_take = OrderedSet(self.sources[0].column_names)
        if using is None:
            return [columns_we_take]
        columns_we_take = ordered_intersect(columns_we_take, using)
        columns_we_take = ordered_union(columns_we_take, self.decision_columns)
        return [columns_we_take]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".select_rows(" + self.expr.to_python().__repr__() + ")")
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.select_rows_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class SelectColumnsNode(ViewRepresentation):
    """
    Class representation of .select_columns() method/step.
    """
    column_selection: List[str]

    def __init__(self, source, columns):
        if isinstance(columns, str):
            columns = [columns]
        column_selection = [c for c in columns]
        self.column_selection = column_selection
        if len(column_selection) < 1:
            raise ValueError("can not drop all columns")
        unknown = set(column_selection) - set(source.column_names)
        if len(unknown) > 0:
            raise KeyError("selecting unknown columns " + str(unknown))
        if isinstance(source, SelectColumnsNode):
            source = source.sources[0]
        ViewRepresentation.__init__(
            self,
            column_names=column_selection,
            sources=[source],
            node_name="SelectColumnsNode",
        )

    def forbidden_columns(self, *, forbidden: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
        """
        Determine which columns should not be in source tables
        (were not in declared structure, and interfere with column production).

        :param forbidden: optional incoming forbids.
        :return: dictionary operator keys to forbidden sets.
        """
        if forbidden is None:
            forbidden = set()
        forbidden = set(forbidden).intersection(self.column_selection)
        return self.sources[0].forbidden_columns(forbidden=forbidden)

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].select_columns(columns=self.column_selection)

    def _equiv_nodes(self, other):
        if not isinstance(other, SelectColumnsNode):
            return False
        if not self.column_selection == other.column_selection:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        cols = set(self.column_selection.copy())
        if using is None:
            return [cols]
        return [cols.intersection(using)]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".select_columns(" + self.column_selection.__repr__() + ")")
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.select_columns_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class DropColumnsNode(ViewRepresentation):
    """
    Class representation of .drop_columns() method/step.
    """
    column_deletions: List[str]

    def __init__(self, source, column_deletions):
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions
        unknown = set(column_deletions) - set(source.column_names)
        if len(unknown) > 0:
            raise KeyError("dropping unknown columns " + str(unknown))
        remaining_columns = [
            c for c in source.column_names if c not in column_deletions
        ]
        if len(remaining_columns) < 1:
            raise ValueError("can not drop all columns")
        ViewRepresentation.__init__(
            self,
            column_names=remaining_columns,
            sources=[source],
            node_name="DropColumnsNode",
        )

    def forbidden_columns(self, *, forbidden: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
        """
        Determine which columns should not be in source tables
        (were not in declared structure, and interfere with column production).

        :param forbidden: optional incoming forbids.
        :return: dictionary operator keys to forbidden sets.
        """
        if forbidden is None:
            forbidden = set()
        forbidden = set(forbidden) - set(self.column_deletions)
        return self.sources[0].forbidden_columns(forbidden=forbidden)

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].drop_columns(column_deletions=self.column_deletions)

    def _equiv_nodes(self, other):
        if not isinstance(other, DropColumnsNode):
            return False
        if not self.column_deletions == other.column_deletions:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        if using is None:
            using = set(self.sources[0].column_names)
        return [set([c for c in using if c not in self.column_deletions])]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".drop_columns(" + self.column_deletions.__repr__() + ")")
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.drop_columns_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class OrderRowsNode(ViewRepresentation):
    """
    Class representation of .order_rows() method/step.
    """
    order_columns: List[str]
    reverse: List[str]

    def __init__(self, source, columns, *, reverse=None, limit=None):
        if isinstance(columns, str):
            columns = [columns]
        self.order_columns = [c for c in columns]
        if reverse is None:
            reverse = []
        if isinstance(reverse, str):
            reverse = [reverse]
        self.reverse = [c for c in reverse]
        self.limit = limit
        have = source.column_names
        unknown = set(self.order_columns) - set(have)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        not_order = set(self.reverse) - set(self.order_columns)
        if len(not_order) > 0:
            raise ValueError("columns declared reverse, but not order: " + str(unknown))
        ViewRepresentation.__init__(
            self,
            column_names=source.column_names,
            sources=[source],
            node_name="OrderRowsNode",
        )

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].order_rows(
            columns=self.order_columns, reverse=self.reverse, limit=self.limit
        )

    def _equiv_nodes(self, other):
        if not isinstance(other, OrderRowsNode):
            return False
        if not self.order_columns == other.order_columns:
            return False
        if not self.reverse == other.reverse:
            return False
        if not self.limit == other.limit:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        cols = set(self.column_names)
        if using is None:
            return [cols]
        cols = cols.intersection(using).union(self.order_columns)
        return [cols]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".order_rows(" + self.order_columns.__repr__())
        if len(self.reverse) > 0:
            s = s + ", reverse=" + self.reverse.__repr__()
        if self.limit is not None:
            s = s + ", limit=" + self.limit.__repr__()
        s = s + ")"
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.order_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )

    # short-cut main interface

    def is_trivial_when_intermediate_(self) -> bool:
        """
        Return if True if operator can be eliminated from interior of chain.
        """
        return self.limit is None


class RenameColumnsNode(ViewRepresentation):
    """
    Class representation of .rename_columns() method/step.
    """
    column_remapping: Dict[str, str]
    reverse_mapping: Dict[str, str]
    mapped_columns: Set[str]

    def __init__(self, source, column_remapping):
        self.column_remapping = column_remapping.copy()
        self.reverse_mapping = {v: k for (k, v) in self.column_remapping.items()}
        self.mapped_columns = set(self.column_remapping.keys()).union(
            set(self.reverse_mapping.keys())
        )
        new_cols = [k for k in column_remapping.keys()]
        orig_cols = [k for k in column_remapping.values()]
        unknown = set(orig_cols) - set(source.column_names)
        if len(unknown) > 0:
            raise ValueError("Tried to rename unknown columns: " + str(unknown))
        collisions = (
            set(source.column_names) - set(new_cols).intersection(orig_cols)
        ).intersection(new_cols)
        if len(collisions) > 0:
            raise ValueError(
                "Mapping "
                + str(self.column_remapping)
                + " collides with existing columns "
                + str(collisions)
            )
        column_names = [
            (k if k not in self.reverse_mapping.keys() else self.reverse_mapping[k])
            for k in source.column_names
        ]
        self.new_columns = set(new_cols) - set(orig_cols)
        ViewRepresentation.__init__(
            self,
            column_names=column_names,
            sources=[source],
            node_name="RenameColumnsNode",
        )

    def forbidden_columns(self, *, forbidden: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
        """
        Determine which columns should not be in source tables
        (were not in declared structure, and interfere with column production).

        :param forbidden: optional incoming forbids.
        :return: dictionary operator keys to forbidden sets.
        """
        # this is where forbidden columns are introduced
        if forbidden is None:
            forbidden = set()
        new_forbidden = set(forbidden) - self.reverse_mapping.keys()
        new_forbidden.update(self.new_columns)
        return self.sources[0].forbidden_columns(forbidden=new_forbidden)

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].rename_columns(column_remapping=self.column_remapping)

    def _equiv_nodes(self, other):
        if not isinstance(other, RenameColumnsNode):
            return False
        if not self.column_remapping == other.column_remapping:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        if using is None:
            using_tuple = self.column_names
        else:
            using_tuple = tuple(using)
        cols = [
            (k if k not in self.column_remapping.keys() else self.column_remapping[k])
            for k in using_tuple
        ]
        return [OrderedSet(cols)]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".rename_columns(" + self.column_remapping.__repr__() + ")")
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.rename_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class NaturalJoinNode(ViewRepresentation):
    """
    Class representation of .natural_join() method/step.
    """
    by: List[str]
    jointype: str

    def __init__(self, a, b, *, by, jointype, check_all_common_keys_in_by=False):
        # check set of tables is consistent in both sub-dags
        a_tables = a.get_tables()
        b_tables = b.get_tables()
        _assert_tables_defs_consistent(a_tables, b_tables)
        if by is None:
            raise ValueError(
                "Must specify by in natural joins ([] for empty conditions)"
            )
        common_table_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_table_keys:
            if not a_tables[k].same_table_description_(b_tables[k]):
                raise ValueError(
                    "Different definition of table object on a/b for: " + k
                )
        # check columns
        column_names = list(a.column_names)
        columns_seen = set(column_names)
        for ci in b.column_names:
            if ci not in columns_seen:
                column_names.append(ci)
                columns_seen.add(ci)
        if isinstance(by, str):
            by = [by]
        by_set = set(by)
        if len(by) != len(by_set):
            raise ValueError("duplicate column names in by")
        missing_left = by_set - set(a.column_names)
        if len(missing_left) > 0:
            raise KeyError("left table missing join keys: " + str(missing_left))
        missing_right = by_set - set(b.column_names)
        if len(missing_right) > 0:
            raise KeyError("right table missing join keys: " + str(missing_right))
        if check_all_common_keys_in_by:
            missing_common = (
                set(a.column_names).intersection(set(b.column_names)) - by_set
            )
            if len(missing_common) > 0:
                raise KeyError(
                    "check_all_common_keys_in_by set, and the following common keys are are not in the by-clause: "
                    + str(missing_common)
                )
        # try to re-use column names if possible, saves space in deeply nested join trees.
        column_names = tuple(column_names)
        if isinstance(a.column_names, tuple) and (
            set(column_names) == set(a.column_names)
        ):
            column_names = a.column_names
        elif isinstance(b.column_names, tuple) and (
            set(column_names) == set(b.column_names)
        ):
            column_names = b.column_names
        ViewRepresentation.__init__(
            self,
            column_names=column_names,
            sources=[a, b],
            node_name="NaturalJoinNode",
        )
        self.by = by
        self.jointype = data_algebra.expr_rep.standardize_join_type(jointype)
        if (self.jointype == "CROSS") and (len(self.by) != 0):
            raise ValueError("CROSS joins must have an empty 'by' list")

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].natural_join(
            b=new_sources[1], by=self.by, jointype=self.jointype
        )

    def _equiv_nodes(self, other):
        if not isinstance(other, NaturalJoinNode):
            return False
        if not self.by == other.by:
            return False
        if not self.jointype == other.jointype:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        if using is None:
            return [OrderedSet(self.sources[i].column_names) for i in range(2)]
        using = using.union(self.by)
        return [
            ordered_intersect(self.sources[i].column_names, using) for i in range(2)
        ]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = "_0."
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".natural_join(b=\n" + " " * (indent + 6))
        if print_sources:
            s = s + (
                self.sources[1].to_python_src_(
                    indent=max(indent, 0) + 6, strict=strict
                )
                + ",\n"
                + " " * (max(indent, 0) + 6)
            )
        else:
            s = s + " _1, "
        s = s + (
            "by=" + self.by.__repr__() + ", jointype=" + self.jointype.__repr__() + ")"
        )
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.natural_join_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class ConcatRowsNode(ViewRepresentation):
    """
    Class representation of .concat_rows() method/step.
    """
    id_column: Union[str, None]

    def __init__(self, a, b, *, id_column="table_name", a_name="a", b_name="b"):
        # check set of tables is consistent in both sub-dags
        assert isinstance(a, ViewRepresentation)
        assert isinstance(b, ViewRepresentation)
        a_tables = a.get_tables()
        b_tables = b.get_tables()
        _assert_tables_defs_consistent(a_tables, b_tables)
        common_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_keys:
            if not a_tables[k].same_table_description_(b_tables[k]):
                raise ValueError(
                    "Different definition of table object on a/b for: " + k
                )
        sources = [a, b]
        # check columns
        if not set(sources[0].column_names) == set(sources[1].column_names):
            raise ValueError("a and b should have same set of column names")
        if id_column is not None and id_column in sources[0].column_names:
            raise ValueError("id_column should not be an input table column name")
        column_names = list(sources[0].column_names)
        if id_column is not None:
            assert id_column not in column_names
            column_names.append(id_column)
        ViewRepresentation.__init__(
            self, column_names=column_names, sources=sources, node_name="ConcatRowsNode"
        )
        self.id_column = id_column
        self.a_name = a_name
        self.b_name = b_name

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].concat_rows(
            b=new_sources[1],
            id_column=self.id_column,
            a_name=self.a_name,
            b_name=self.b_name,
        )

    def _equiv_nodes(self, other):
        if not isinstance(other, ConcatRowsNode):
            return False
        if not self.id_column == other.id_column:
            return False
        if not self.a_name == other.a_name:
            return False
        if not self.b_name == other.b_name:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        if using is None:
            return [OrderedSet(self.sources[i].column_names) for i in range(2)]
        return [
            ordered_intersect(self.sources[i].column_names, using) for i in range(2)
        ]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = "_0."
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + (".concat_rows(b=\n" + " " * (indent + 6))
        if print_sources:
            s = s + (
                self.sources[1].to_python_src_(
                    indent=max(indent, 0) + 6, strict=strict
                )
                + ",\n"
                + " " * (max(indent, 0) + 6)
            )
        else:
            s = s + " _1, "
        s = s + (
            "id_column="
            + self.id_column.__repr__()
            + ", a_name="
            + self.a_name.__repr__()
            + ", b_name="
            + self.b_name.__repr__()
            + ")"
        )
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        return db_model.concat_rows_to_near_sql(
            self,
            using=using,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )


class ConvertRecordsNode(ViewRepresentation):
    """
    Class representation of .convert_records() method/step.
    """
    def __init__(self, *, source, record_map):
        sources = [source]
        self.record_map = record_map
        unknown = set(self.record_map.columns_needed) - set(source.column_names)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        ViewRepresentation.__init__(
            self,
            column_names=record_map.columns_produced,
            sources=sources,
            node_name="ConvertRecordsNode",
        )

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].convert_records(record_map=self.record_map)

    def _equiv_nodes(self, other):
        if not isinstance(other, ConvertRecordsNode):
            return False
        if not self.record_map == other.record_map:
            return False
        return True

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        return [self.record_map.columns_needed]

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_src_(indent=indent, strict=strict)
                + "\n"
                + " " * (max(indent, 0) + 3)
            )
        rm_str = self.record_map.__repr__()
        rm_str = re.sub("\n", "\n   ", rm_str)
        s = s + ".convert_records(" + rm_str
        s = s + ")"
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        if temp_id_source is None:
            temp_id_source = [0]
        # TODO: narrow to what we are using
        # TODO: use nearsql instead of strings / lists of strings
        near_sql = self.sources[0].to_near_sql_implementation_(
            db_model=db_model,
            using=None,
            temp_id_source=temp_id_source,
            sql_format_options=sql_format_options,
        )
        assert isinstance(near_sql, data_algebra.near_sql.NearSQL)
        # claims to use all columns
        if self.record_map.blocks_in is not None:
            view_name = "convert_records_blocks_in_" + str(temp_id_source[0])
            temp_id_source[0] = temp_id_source[0] + 1
            pi, si = db_model.blocks_to_row_recs_query_str_list_pair(
                record_spec=self.record_map.blocks_in
            )
            near_sql = data_algebra.near_sql.NearSQLRawQStep(
                prefix=pi,
                query_name=view_name,
                quoted_query_name=db_model.quote_identifier(view_name),
                sub_sql=data_algebra.near_sql.NearSQLContainer(near_sql=near_sql),
                suffix=si,
                annotation="convert records blocks in",
            )
            assert isinstance(near_sql, data_algebra.near_sql.NearSQL)
        if self.record_map.blocks_out is not None:
            view_name = "convert_records_blocks_out_" + str(temp_id_source[0])
            temp_id_source[0] = temp_id_source[0] + 1
            pi, si = db_model.row_recs_to_blocks_query_str_list_pair(
                record_spec=self.record_map.blocks_out,
            )
            near_sql = data_algebra.near_sql.NearSQLRawQStep(
                prefix=pi,
                query_name=view_name,
                quoted_query_name=db_model.quote_identifier(view_name),
                sub_sql=data_algebra.near_sql.NearSQLContainer(near_sql=near_sql),
                suffix=si,
                annotation="convert records blocks out",
            )
            assert isinstance(near_sql, data_algebra.near_sql.NearSQL)
        return near_sql


class SQLNode(ViewRepresentation):
    """
    Class representation of user SQL step in pipeline. Can be used to start a pipeline instead of a TableDescription.
    """
    def __init__(
        self, *, sql: Union[str, List[str]], column_names: List[str], view_name: str
    ):
        if isinstance(sql, str):
            sql = sql.splitlines(keepends=False)
            sql = [v for v in sql if len(v.strip()) > 0]
        assert isinstance(sql, list)
        assert len(sql) > 0
        assert all([isinstance(v, str) for v in sql])
        assert isinstance(view_name, str)
        self.sql = sql.copy()
        self.view_name = view_name
        ViewRepresentation.__init__(
            self, column_names=column_names, node_name="SQLNode",
        )

    def apply_to(self, a, *, target_table_key=None):
        """
        Apply self to operator DAG a. Basic OperatorPlatform, composabile API.

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        return self

    def _equiv_nodes(self, other):
        if not isinstance(other, SQLNode):
            return False
        if self.view_name != other.view_name:
            return False
        if self.column_names != other.column_names:
            return False
        if self.sql != other.sql:
            return False
        return True

    def get_tables(self):
        """Get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent."""
        return dict()

    def columns_used_from_sources(self, using: Optional[set] = None) -> List:
        """
        Get columns used from sources. Internal method.

        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).
        """
        return []

    def to_python_src_(self, *, indent=0, strict=True, print_sources=True):
        """
        Return text representing operations.

        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.
        """
        s = (
            "SQLNode(sql="
            + str(self.sql)
            + ", column_names="
            + str(self.column_names)
            + ", view_name="
            + self.view_name.__repr__()
            + ")"
        )
        return s

    def to_near_sql_implementation_(
        self, db_model, *, using, temp_id_source, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert operator dag into NearSQL type for translation to SQL string.

        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL
        """
        quoted_query_name = db_model.quote_identifier(self.view_name)
        near_sql = data_algebra.near_sql.NearSQLRawQStep(
            prefix=self.sql,
            query_name=self.view_name,
            quoted_query_name=quoted_query_name,
            sub_sql=None,
            suffix=None,
            annotation="user supplied SQL",
            add_select=False,
        )
        return near_sql


def ex(d, *, data_model=None, narrow=True, allow_limited_tables=False):
    """
    Evaluate operators with respect to Pandas data frames already stored in the operator chain.

    :param d: data algebra pipeline or OpC container to evaluate.
    :param data_model: adaptor to data dialect (Pandas for now)
    :param narrow: logical, if True don't copy unexpected columns
    :param allow_limited_tables: logical, if True allow execution on non-complete tables
    :return: table result
    """
    return d.ex(
        data_model=data_model, narrow=narrow, allow_limited_tables=allow_limited_tables
    )
