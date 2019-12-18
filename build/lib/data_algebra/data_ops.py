from abc import ABC
from typing import Set, Dict, List, Union
import numbers
import collections
import re

import data_algebra
import data_algebra.flow_text
import data_algebra.data_model
import data_algebra.db_model
import data_algebra.pandas_model
import data_algebra.expr_rep
import data_algebra.env
from data_algebra.data_ops_types import *
import data_algebra.data_ops_utils
import data_algebra.near_sql

_have_black = False
try:
    # noinspection PyUnresolvedReferences
    import black

    _have_black = True
except ImportError:
    pass

_have_sqlparse = False
try:
    # noinspection PyUnresolvedReferences
    import sqlparse

    _have_sqlparse = True
except ImportError:
    pass


class ViewRepresentation(OperatorPlatform, ABC):
    """Structure to represent the columns of a query or a table.
       Abstract base class."""

    column_names: List[str]
    column_set: Set[str]
    column_map: data_algebra.env.SimpleNamespaceDict
    sources: List[
        "ViewRepresentation"
    ]  # https://www.python.org/dev/peps/pep-0484/#forward-references

    def __init__(self, column_names, *, sources=None, node_name):
        if isinstance(column_names, str):
            column_names = [column_names]
        self.column_names = [c for c in column_names]
        for ci in self.column_names:
            if not isinstance(ci, str):
                raise ValueError("non-string column name(s)")
        if len(self.column_names) < 1:
            raise ValueError("no column names")
        self.column_set = set(self.column_names)
        if not len(self.column_names) == len(self.column_set):
            raise ValueError("duplicate column name(s)")
        column_dict = {
            ci: data_algebra.expr_rep.ColumnReference(self, ci)
            for ci in self.column_names
        }
        self.column_map = data_algebra.env.SimpleNamespaceDict(**column_dict)
        if sources is None:
            sources = []
        for si in sources:
            if not isinstance(si, ViewRepresentation):
                raise ValueError("all sources must be of class ViewRepresentation")
        self.sources = [si for si in sources]
        self.columns_currently_used = set()
        OperatorPlatform.__init__(self, node_name=node_name)

    # adaptors

    def get_column_symbols(self):
        """Return a representation of this step as columns we can perform algebraic operations over.
        These objects capture the operations as an expression tree."""
        column_defs = self.column_map.__dict__
        nd = column_defs.copy()
        ns = data_algebra.env.SimpleNamespaceDict(**nd)
        return ns

    # characterization

    def get_tables(self, *, replacements=None):
        """Get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent."""
        tables = {}
        for i in range(len(self.sources)):
            s = self.sources[i]
            if isinstance(s, TableDescription):
                if replacements is not None and s.key in replacements:
                    orig_table = replacements[s.key]
                    if s.column_set != orig_table.column_set:
                        raise ValueError(
                            "table " + s.key + " has two incompatible definitions"
                        )
                    self.sources[i] = orig_table
                    s = orig_table
            ti = s.get_tables(replacements=replacements)
            for (k, v) in ti.items():
                if k in tables.keys():
                    if not tables[k] is v:
                        raise ValueError(
                            "Table " + k + " has two different representation objects"
                        )
                else:
                    tables[k] = v
        return tables

    def columns_used_from_sources(self, using=None):
        """Get column names used from direct source nodes when this node is exececuted
        with the using columns (None means all)."""
        raise NotImplementedError("base method called")

    def _clear_columns_currently_used(self):
        self.columns_currently_used = set()
        for si in self.sources:
            si._clear_columns_currently_used()

    def _columns_used_implementation(self, *, using=None):
        if using is None:
            self.columns_currently_used.update(self.column_names)
        else:
            unknown = set(using) - set(self.column_names)
            if len(unknown) > 0:
                raise ValueError("asked for unknown columns: " + str(unknown))
            self.columns_currently_used.update(using)
        cu_list = self.columns_used_from_sources(self.columns_currently_used.copy())
        for i in range(len(self.sources)):
            self.sources[i]._columns_used_implementation(using=cu_list[i])

    def columns_used(self, *, using=None):
        """Determine which columns are used from source tables.
        Sets nodes' columns_currently_used values as a side-effect."""
        self._clear_columns_currently_used()
        self._columns_used_implementation(using=using)
        tables = self.get_tables()
        # columns_used = {k: v.columns_currently_used.copy() for (k, v) in tables.items()}
        columns_used = dict()
        for k in tables.keys():
            ti = tables[k]
            vi = ti.columns_currently_used
            columns_used[k] = vi.copy()
        return columns_used

    # collect as simple structures for YAML I/O and other generic tasks

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        raise NotImplementedError("base method called")

    def collect_representation(self, *, pipeline=None, dialect="Python"):
        """Collect a representation of the operator DAG as simple serializable objects.
                   These objects can be saved/loaded in YAML format and also can rebuild the
                   pipeline via data_algebra.yaml.to_pipeline()."""
        self.columns_used()  # for table consistency check/raise
        return self.collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    # printing

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        return "ViewRepresentation(" + self.column_names.__repr__() + ")"

    # noinspection PyBroadException
    def to_python(self, *, indent=0, strict=True, pretty=False, black_mode=None):
        self.columns_used()  # for table consistency check/raise
        if pretty:
            strict = True
        python_str = self.to_python_implementation(
            indent=indent, strict=strict, print_sources=True
        )
        if pretty:
            if _have_black:
                try:
                    if black_mode is None:
                        black_mode = black.FileMode()
                    python_str = black.format_str(python_str, mode=black_mode)
                except Exception:
                    pass
        return python_str

    def __repr__(self):
        return self.to_python(strict=True)

    def __str__(self):
        return self.to_python(strict=True)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _equiv_nodes(self, other):
        """Check if immediate node structure is equivalent, assumes non
        recursive __eq__ checks are TRUE"""
        return True

    def __eq__(self, other):
        if not isinstance(other, ViewRepresentation):
            return False
        if not type(self) is type(other):
            return False
        if self.node_name != other.node_name:
            return False
        if self.column_names != other.column_names:
            return False
        if self.column_map != other.column_map:
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

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        raise NotImplementedError("base method called")

    # noinspection PyBroadException
    def to_sql(self, db_model, *, pretty=False, encoding=None, sqlparse_options=None):
        if sqlparse_options is None:
            sqlparse_options = {"reindent": True, "keyword_case": "upper"}
        if not isinstance(db_model, data_algebra.db_model.DBModel):
            raise TypeError(
                "Expected db_model to be derived from data_algebra.db_model.DBModel"
            )
        self.columns_used()  # for table consistency check/raise
        temp_id_source = [0]
        sql_str = self.to_sql_implementation(
            db_model=db_model, using=None, temp_id_source=temp_id_source
        )
        if isinstance(sql_str, str):
            print("break")
        sql_str = sql_str.to_sql(db_model=db_model, force_sql=True)
        if pretty and _have_sqlparse:
            try:
                sql_str = sqlparse.format(
                    sql_str, encoding=encoding, **sqlparse_options
                )
            except Exception:
                pass
        return sql_str

    # Pandas realization

    def eval_implementation(self, *, data_map, eval_env, data_model):
        raise NotImplementedError("base method called")

    def eval_pandas(self, data_map, *, eval_env=None, data_model=None):
        """
        Evaluate operators with respect to Pandas data frames.
        :param data_map: map from table names to data frames
        :param eval_env: environment to evaluate in
        :param data_model: adaptor to Pandas dialect
        :return:
        """

        if not isinstance(data_map, Dict):
            raise TypeError("data_map should be a dictionary")
        if len(data_map) < 1:
            raise ValueError("data_map should not be empty")
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        if not isinstance(data_model, data_algebra.data_model.DataModel):
            raise TypeError(
                "Expected data_model to derive from data_algebra.data_model.DataModel"
            )
        self.columns_used()  # for table consistency check/raise
        tables = self.get_tables()
        for k in tables.keys():
            if k not in data_map.keys():
                raise ValueError("Required table " + k + " not in data_map")
            else:
                data_model.assert_is_appropriate_data_instance(
                    data_map[k], "data_map[" + k + "]"
                )
        return self.eval_implementation(
            data_map=data_map, eval_env=eval_env, data_model=data_model
        )

    def eval(self, data_map, *, eval_env=None, data_model=None):
        if len(data_map) < 1:
            raise ValueError("Expected data_map to be non-empty")
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        if not isinstance(data_model, data_algebra.data_model.DataModel):
            raise TypeError(
                "Expected data_model to be derived from data_algebra.data_model.DataModel"
            )
        k = [k for k in data_map.keys()][0]
        x = data_map[k]
        # noinspection PyUnresolvedReferences
        if isinstance(x, data_model.pd.DataFrame):
            return self.eval_pandas(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        raise TypeError("can not apply eval() to type " + str(type(x)))

    # noinspection PyPep8Naming
    def transform(self, X, *, eval_env=None, data_model=None):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        if not isinstance(data_model, data_algebra.data_model.DataModel):
            raise TypeError(
                "Expected data_model to be derived from data_algebra.data_model.DataModel"
            )
        self.columns_used()  # for table consistency check/raise
        tables = self.get_tables()
        if len(tables) != 1:
            raise ValueError(
                "transfrom(DataFrame) can only be applied to ops-dags with only one table def"
            )
        k = [k for k in tables.keys()][0]
        # noinspection PyUnresolvedReferences
        if isinstance(X, data_model.pd.DataFrame):
            data_map = {k: X}
            return self.eval_pandas(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        raise TypeError("can not apply transform() to type " + str(type(X)))

    # composition (used to eliminate intermediate order nodes)

    def is_trivial_when_intermediate(self):
        return False

    # return table representation of self
    def as_table_description(
        self, table_name=None, *, qualifiers=None, column_types=None
    ):
        return TableDescription(
            table_name=table_name,
            column_names=self.column_names.copy(),
            qualifiers=qualifiers,
            column_types=column_types,
        )

    # implement builders for all non-initial node types on base class
    def extend_parsed(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        if (parsed_ops is None) or (len(parsed_ops) < 1):
            return self
        if partition_by is None:
            partition_by = []
        if order_by is None:
            order_by = []
        if reverse is None:
            reverse = []
        new_cols_produced_in_calc = set([k for k in parsed_ops.keys()])
        if (partition_by != 1) and (len(partition_by) > 0):
            if len(new_cols_produced_in_calc.intersection(partition_by)) > 0:
                raise ValueError("must not change partition_by columns")
        if len(new_cols_produced_in_calc.intersection(order_by)) > 0:
            raise ValueError("must not change partition_by columns")
        if len(set(reverse).difference(order_by)) > 0:
            raise ValueError("all columns in reverse must be in order_by")
        if self.is_trivial_when_intermediate():
            return self.sources[0].extend_parsed(
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
        # new node
        return ExtendNode(
            source=self,
            parsed_ops=parsed_ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse,
        )

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        if (ops is None) or (len(ops) < 1):
            return self
        parsed_ops = data_algebra.expr_rep.parse_assignments_in_context(
            ops, self, parse_env=parse_env
        )
        return self.extend_parsed(
            parsed_ops=parsed_ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse,
        )

    def project_parsed(self, parsed_ops=None, *, group_by=None):
        if group_by is None:
            group_by = []
        if ((parsed_ops is None) or (len(parsed_ops) < 1)) and (len(group_by) < 1):
            raise ValueError("must have ops or group_by")
        new_cols_produced_in_calc = set([k for k in parsed_ops.keys()])
        if len(new_cols_produced_in_calc.intersection(group_by)):
            raise ValueError("can not alter grouping columns")
        if self.is_trivial_when_intermediate():
            return self.sources[0].project_parsed(parsed_ops, group_by=group_by)
        return ProjectNode(source=self, parsed_ops=parsed_ops, group_by=group_by)

    def project(self, ops=None, *, group_by=None, parse_env=None):
        if group_by is None:
            group_by = []
        if ((ops is None) or (len(ops) < 1)) and (len(group_by) < 1):
            raise ValueError("must have ops or group_by")
        parsed_ops = data_algebra.expr_rep.parse_assignments_in_context(
            ops, self, parse_env=parse_env
        )
        return self.project_parsed(parsed_ops=parsed_ops, group_by=group_by)

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, ViewRepresentation):
            raise TypeError(
                "expected b to be a data_algebra.dat_ops.ViewRepresentation"
            )
        if self.is_trivial_when_intermediate():
            return self.sources[0].natural_join(b, by=by, jointype=jointype)
        return NaturalJoinNode(a=self, b=b, by=by, jointype=jointype)

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        if b is None:
            return self
        if not isinstance(b, ViewRepresentation):
            raise TypeError(
                "expected b to be a data_algebra.dat_ops.ViewRepresentation"
            )
        if self.is_trivial_when_intermediate():
            return self.sources[0].concat_rows(
                b, id_column=id_column, a_name=a_name, b_name=b_name
            )
        return ConcatRowsNode(
            a=self, b=b, id_column=id_column, a_name=a_name, b_name=b_name
        )

    def select_rows_parsed(self, parsed_expr):
        if parsed_expr is None:
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].select_rows_parsed(parsed_expr=parsed_expr)
        return SelectRowsNode(source=self, ops=parsed_expr)

    def select_rows(self, expr, *, parse_env=None):
        if expr is None:
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].select_rows(expr, parse_env=parse_env)
        ops = data_algebra.expr_rep.parse_assignments_in_context(
            {"expr": expr}, self, parse_env=parse_env
        )
        return self.select_rows_parsed(parsed_expr=ops)

    def drop_columns(self, column_deletions):
        if (column_deletions is None) or (len(column_deletions) < 1):
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].drop_columns(column_deletions)
        return DropColumnsNode(source=self, column_deletions=column_deletions)

    def select_columns(self, columns):
        if (columns is None) or (len(columns) < 1):
            raise ValueError("must select at least one column")
        if columns == self.column_names:
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].select_columns(columns)
        if isinstance(self, SelectColumnsNode):
            return self.sources[0].select_columns(columns)
        if isinstance(self, DropColumnsNode):
            return self.sources[0].select_columns(columns)
        return SelectColumnsNode(source=self, columns=columns)

    def rename_columns(self, column_remapping):
        if (column_remapping is None) or (len(column_remapping) < 1):
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].rename_columns(column_remapping)
        return RenameColumnsNode(source=self, column_remapping=column_remapping)

    def order_rows(self, columns, *, reverse=None, limit=None):
        if ((columns is None) or (len(columns) < 1)) and (limit is None):
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].order_rows(columns, reverse=reverse, limit=limit)
        return OrderRowsNode(source=self, columns=columns, reverse=reverse, limit=limit)

    def convert_records(self, record_map, *, blocks_out_table=None):
        if record_map is None:
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].convert_records(
                record_map, blocks_out_table=blocks_out_table
            )
        return ConvertRecordsNode(
            source=self, record_map=record_map, blocks_out_table=blocks_out_table
        )


# Could also have general query as starting node, but don't see a lot of point to
# it until somebody needs it.


class TableDescription(ViewRepresentation):
    """Describe columns, and qualifiers, of a table.

       If outer namespace is set user values are visible and
       _-side effects can be written back.

       Example:
           from data_algebra.data_ops import *
           import data_algebra.env
           with data_algebra.env.Env(globals()) as env:
               d = TableDescription('d', ['x', 'y'])
           print(_) # should be a SimpleNamespaceDict, not d/ViewRepresentation
           print(d)
    """

    table_name: str
    column_names: List[str]
    qualifiers: Dict[str, str]
    key: str

    def __init__(self, table_name, column_names, *, qualifiers=None, column_types=None):
        ViewRepresentation.__init__(
            self, column_names=column_names, node_name="TableDescription"
        )
        if table_name is None:
            table_name = ""
        if (table_name is not None) and (not isinstance(table_name, str)):
            raise TypeError("table_name must be a string")
        self.table_name = table_name
        if isinstance(column_names, str):
            column_names = [column_names]
        self.column_names = [c for c in column_names]
        self.column_types = None
        if column_types is not None:
            self.column_types = column_types.copy()
        if qualifiers is None:
            qualifiers = {}
        if not isinstance(qualifiers, dict):
            raise TypeError("qualifiers must be a dictionary")
        self.qualifiers = qualifiers.copy()
        key = ""
        if len(self.qualifiers) > 0:
            keys = [k for k in self.qualifiers.keys()]
            keys.sort()
            key = "{"
            for k in keys:
                key = key + "(" + k + ", " + str(self.qualifiers[k]) + ")"
            key = key + "}."
        if self.table_name is not None:
            key = key + self.table_name
        self.key = key

    def apply_to(self, a, *, target_table_key=None):
        if (target_table_key is None) or (target_table_key == self.key):
            # replace table with a
            return a
        # copy self
        r = TableDescription(
            table_name=self.table_name,
            column_names=self.column_names,
            qualifiers=self.qualifiers,
            column_types=self.column_types,
        )
        return r

    def _equiv_nodes(self, other):
        if not self.table_name == other.table_name:
            return False
        if not self.column_names == other.column_names:
            return False
        if not self.qualifiers == other.qualifiers:
            return False
        if not self.key == other.key:
            return False
        return True

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "TableDescription"
        od["table_name"] = self.table_name
        od["qualifiers"] = self.qualifiers.copy()
        od["column_names"] = self.column_names
        od["key"] = self.key
        pipeline.insert(0, od)
        return pipeline

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
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
            cols_to_print, align_right=70 - indent, sep_width=2
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

    def get_tables(self, *, replacements=None):
        """get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent"""
        if replacements is not None and self.key in replacements.keys():
            return {self.key: replacements[self.key]}
        return {self.key: self}

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.table_step(op=self, data_map=data_map, eval_env=eval_env)

    def columns_used_from_sources(self, using=None):
        return []  # no inputs to table description

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.table_def_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    # comparable to other table descriptions
    def __lt__(self, other):
        if not isinstance(other, TableDescription):
            return True
        return self.key.__lt__(other.key)

    def __eq__(self, other):
        if not isinstance(other, TableDescription):
            return False
        return self.key.__eq__(other.key)

    def __hash__(self):
        return self.key.__hash__()


def describe_table(d, table_name="data_frame", *, qualifiers=None, column_types=None):
    # Expect a pandas.DataFrame style object
    column_names = [c for c in d.columns]
    if column_types is None:
        if d.shape[0] > 0:
            column_types = {k: type(d.loc[0, k]) for k in column_names}
    return TableDescription(
        table_name, column_names, column_types=column_types, qualifiers=qualifiers
    )


class WrappedOperatorPlatform(OperatorPlatform):
    """Decorator class for OperatorPlatform."""

    def __init__(self, *, underlying, data_map):
        if not isinstance(underlying, ViewRepresentation):
            raise TypeError("Expected underlying to be of class OperatorPlatform")
        if isinstance(underlying, WrappedOperatorPlatform):
            raise TypeError("underlying should not be a WrappedOperatorPlatform")
        if not isinstance(data_map, dict):
            raise TypeError("Expected data_map to be a key to table dictionary")
        OperatorPlatform.__init__(self, node_name=underlying.node_name)
        self.data_map = data_map.copy()
        self.underlying = underlying

    # execution

    def ex(self):
        """Execute pipeline against internally stored values"""
        tables = self.underlying.get_tables()
        data_map = self.data_map
        missing = set(tables.keys()) - set(data_map.keys())
        if len(missing) > 0:
            raise ValueError("missing required tables: " + str(missing))
        res = self.underlying.eval(data_map=data_map)
        return res

    # general functions

    def __repr__(self):
        return (
            "["
            + self.underlying.__repr__()
            + "](\n "
            + set(self.data_map.keys()).__repr__()
            + ")"
        )

    def __str__(self):
        return (
            "["
            + self.underlying.__str__()
            + "](\n "
            + set(self.data_map.keys()).__repr__()
            + ")"
        )

    # overrides

    def _reach_in(self, other):
        data_map = self.data_map.copy()
        if isinstance(other, WrappedOperatorPlatform):
            data_map.update(other.data_map)
            other = other.underlying
        return data_map, other

    def apply_to(self, a, *, target_table_key=None):
        if not isinstance(a, WrappedOperatorPlatform):
            raise TypeError("didn't expect type: " + str(type(a)))
        data_map, a = self._reach_in(a)
        return WrappedOperatorPlatform(
            underlying=self.underlying.apply_to(a, target_table_key=target_table_key),
            data_map=data_map,
        )

    # noinspection PyPep8Naming
    def transform(self, X):
        data_map, X = self._reach_in(X)
        return WrappedOperatorPlatform(
            underlying=self.underlying.transform(X), data_map=data_map
        )

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __rshift__(self, other):  # override self >> other
        # can't use type >> type if only __rrshift__ is defined (must have __rshift__ in this case)
        data_map, other = self._reach_in(other)
        if isinstance(other, OperatorPlatform):
            return WrappedOperatorPlatform(
                underlying=other.transform(self.underlying), data_map=data_map
            )
        if isinstance(other, PipeStep):
            return WrappedOperatorPlatform(
                underlying=other.apply_to(self.underlying), data_map=data_map
            )
        raise TypeError("unexpected type: " + str(type(other)))

    # composition

    def add(self, other):
        """interface to what we used to call PipeStep nodes"""
        data_map, other = self._reach_in(other)
        return WrappedOperatorPlatform(
            underlying=other.apply_to(self.underlying), data_map=data_map
        )

    # sql

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return self.underlying.to_sql_implementation(
            db_model=db_model, using=using, temp_id_source=temp_id_source
        )

    # define builders for all non-initial node types on base class

    def extend_parsed(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        return WrappedOperatorPlatform(
            underlying=self.underlying.extend_parsed(
                parsed_ops=parsed_ops,
                partition_by=partition_by,
                order_by=order_by,
                reverse=reverse,
            ),
            data_map=self.data_map,
        )

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        return WrappedOperatorPlatform(
            underlying=self.underlying.extend(
                ops=ops,
                partition_by=partition_by,
                order_by=order_by,
                reverse=reverse,
                parse_env=parse_env,
            ),
            data_map=self.data_map,
        )

    def project_parsed(self, parsed_ops=None, *, group_by=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.project_parsed(
                parsed_ops=parsed_ops, group_by=group_by
            ),
            data_map=self.data_map,
        )

    def project(self, ops=None, *, group_by=None, parse_env=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.project(
                ops=ops, group_by=group_by, parse_env=parse_env
            ),
            data_map=self.data_map,
        )

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, WrappedOperatorPlatform):
            raise TypeError("expected b to be of type WrappedOperatorPlatform")
        data_map, b = self._reach_in(b)
        return WrappedOperatorPlatform(
            underlying=self.underlying.natural_join(b=b, by=by, jointype=jointype),
            data_map=data_map,
        )

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        if b is None:
            return self
        if not isinstance(b, WrappedOperatorPlatform):
            raise TypeError("expected b to be of type WrappedOperatorPlatform")
        data_map, b = self._reach_in(b)
        return WrappedOperatorPlatform(
            underlying=self.underlying.concat_rows(
                b=b, id_column=id_column, a_name=a_name, b_name=b_name
            ),
            data_map=data_map,
        )

    def select_rows_parsed(self, parsed_expr):
        return WrappedOperatorPlatform(
            underlying=self.underlying.select_rows_parsed(parsed_expr=parsed_expr),
            data_map=self.data_map,
        )

    def select_rows(self, expr, *, parse_env=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.select_rows(expr=expr, parse_env=parse_env),
            data_map=self.data_map,
        )

    def drop_columns(self, column_deletions):
        return WrappedOperatorPlatform(
            underlying=self.underlying.drop_columns(column_deletions=column_deletions),
            data_map=self.data_map,
        )

    def select_columns(self, columns):
        return WrappedOperatorPlatform(
            underlying=self.underlying.select_columns(columns=columns),
            data_map=self.data_map,
        )

    def rename_columns(self, column_remapping):
        return WrappedOperatorPlatform(
            underlying=self.underlying.rename_columns(
                column_remapping=column_remapping
            ),
            data_map=self.data_map,
        )

    def order_rows(self, columns, *, reverse=None, limit=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.order_rows(
                columns=columns, reverse=reverse, limit=limit
            ),
            data_map=self.data_map,
        )

    def convert_records(self, record_map, *, blocks_out_table=None):
        if record_map is None:
            return self
        data_map, blocks_out_table = self._reach_in(blocks_out_table)
        return WrappedOperatorPlatform(
            underlying=self.underlying.convert_records(
                record_map=record_map, blocks_out_table=blocks_out_table
            ),
            data_map=data_map,
        )


def wrap(d, *, table_name="data_frame"):
    # Expect a pandas.DataFrame style object
    column_names = [c for c in d.columns]
    column_types = None
    if d.shape[0] > 0:
        column_types = {k: type(d.loc[0, k]) for k in column_names}
    return WrappedOperatorPlatform(
        underlying=TableDescription(
            table_name, column_names, column_types=column_types
        ),
        data_map={table_name: d},
    )


class ExtendNode(ViewRepresentation):
    def __init__(
        self, *, source, parsed_ops, partition_by=None, order_by=None, reverse=None
    ):
        windowed_situation = data_algebra.expr_rep.implies_windowed(parsed_ops)
        self.ops = parsed_ops
        self.cols_used_in_calc = data_algebra.expr_rep.get_columns_used(parsed_ops)
        self.cols_produced_in_calc = [k for k in parsed_ops.keys()]
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
        self.windowed_situation = windowed_situation
        self.order_by = order_by
        if reverse is None:
            reverse = []
        if isinstance(reverse, str):
            reverse = [reverse]
        self.reverse = reverse
        column_names = source.column_names.copy()
        consumed_cols = set()
        for (k, o) in parsed_ops.items():
            o.get_column_names(consumed_cols)
        unknown_cols = consumed_cols - source.column_set
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
            for (k, opk) in parsed_ops.items():
                if not isinstance(opk, data_algebra.expr_rep.Expression):
                    raise ValueError(
                        "non-aggregated expression in windowed/partitoned extend: "
                        + "'"
                        + k
                        + "': '"
                        + opk.to_pandas()
                        + "'"
                    )
                if len(opk.args) > 1:
                    raise ValueError(
                        "in windowed situations only simple operators are allowed, "
                        + "'"
                        + k
                        + "': '"
                        + opk.to_pandas()
                        + "' term is too complex an expression"
                    )
                if len(opk.args) > 0:
                    value_name = opk.args[0].to_pandas()
                    if value_name not in source.column_set:
                        raise ValueError(
                            "in windowed situations only simple operators are allowed, "
                            + "'"
                            + k
                            + "': '"
                            + opk.to_pandas()
                            + "' term is too complex an expression"
                        )
        ViewRepresentation.__init__(
            self, column_names=column_names, sources=[source], node_name="ExtendNode"
        )

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].extend_parsed(
            parsed_ops=self.ops,
            partition_by=self.partition_by,
            order_by=self.order_by,
            reverse=self.reverse,
        )

    def _equiv_nodes(self, other):
        if not self.windowed_situation == other.windowed_situation:
            return False
        if not self.partition_by == other.partition_by:
            return False
        if not self.order_by == other.order_by:
            return False
        if not self.reverse == other.reverse:
            return False
        if not self.ops == other.ops:
            return False
        return True

    def check_extend_window_fns(self):
        window_situation = (len(self.partition_by) > 0) or (len(self.order_by) > 0)
        if window_situation:
            # check these are forms we are prepared to work with
            for (k, opk) in self.ops.items():
                if len(opk.args) > 0:
                    if len(opk.args) > 1:
                        raise ValueError("window function with more than one argument")
                    for i in range(len(opk.args)):
                        if not isinstance(
                            opk.args[0], data_algebra.expr_rep.ColumnReference
                        ):
                            raise ValueError(
                                "window expression argument must be a column: "
                                + str(k)
                                + ": "
                                + str(opk)
                            )

    def columns_used_from_sources(self, using=None):
        columns_we_take = self.sources[0].column_set.copy()
        if using is None:
            return [columns_we_take]
        subops = {k: op for (k, op) in self.ops.items() if k in using}
        if len(subops) <= 0:
            return [columns_we_take]
        columns_we_take = using.union(self.partition_by, self.order_by, self.reverse)
        columns_we_take = columns_we_take - subops.keys()
        for (k, o) in subops.items():
            o.get_column_names(columns_we_take)
        return [columns_we_take]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Extend"
        od["ops"] = {ci: vi.to_source(dialect=dialect) for (ci, vi) in self.ops.items()}
        od["partition_by"] = self.partition_by
        od["order_by"] = self.order_by
        od["reverse"] = self.reverse
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        spacer = "\n   " + " " * indent
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\"
                + spacer
            )
        ops = [
            k.__repr__() + ": " + opi.to_python().__repr__()
            for (k, opi) in self.ops.items()
        ]
        flowed = ("," + spacer + " ").join(ops)
        s = s + ("extend({" + spacer + " " + flowed + "}")
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

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.extend_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.extend_step(op=self, data_map=data_map, eval_env=eval_env)


class ProjectNode(ViewRepresentation):
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
        unknown_cols = consumed_cols - source.column_set
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
            if not isinstance(opk, data_algebra.expr_rep.Expression):
                raise ValueError(
                    "non-aggregated expression in project: " + str(k) + ": " + str(opk)
                )
            if len(opk.args) > 1:
                raise ValueError(
                    "non-trivial aggregation expression: " + str(k) + ": " + str(opk)
                )
            if len(opk.args) > 0:
                if not isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference):
                    raise ValueError(
                        "windows expression argument must be a column: "
                        + str(k)
                        + ": "
                        + str(opk)
                    )
            # TODO: check op is in list of aggregators
            # Note: non-aggregators making through will be caught by table shape check

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].project_parsed(
            parsed_ops=self.ops, group_by=self.group_by
        )

    def _equiv_nodes(self, other):
        if not self.group_by == other.group_by:
            return False
        if not self.ops == other.ops:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        if using is None:
            subops = self.ops
        else:
            subops = {k: op for (k, op) in self.ops.items() if k in using}
        columns_we_take = set(self.group_by)
        for (k, o) in subops.items():
            o.get_column_names(columns_we_take)
        return [columns_we_take]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Project"
        od["ops"] = {ci: vi.to_source(dialect=dialect) for (ci, vi) in self.ops.items()}
        od["group_by"] = self.group_by
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        spacer = "\n   " + " " * indent
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + (
            "project({"
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

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.project_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.project_step(op=self, data_map=data_map, eval_env=eval_env)


class SelectRowsNode(ViewRepresentation):
    expr: data_algebra.expr_rep.Expression
    decision_columns: Set[str]

    def __init__(self, source, ops):
        if len(ops) < 1:
            raise ValueError("no ops")
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

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].select_rows_parsed(parsed_ops=self.ops)

    def _equiv_nodes(self, other):
        if not self.expr == other.expr:
            return False
        if not self.decision_columns == other.decision_columns:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        columns_we_take = self.sources[0].column_set.copy()
        if using is None:
            return [columns_we_take]
        columns_we_take = columns_we_take.intersection(using)
        columns_we_take = columns_we_take.union(self.decision_columns)
        return [columns_we_take]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "SelectRows"
        od["expr"] = self.expr.to_source(dialect=dialect)
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("select_rows(" + self.expr.to_python().__repr__() + ")")
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.select_rows_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.select_rows_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class SelectColumnsNode(ViewRepresentation):
    column_selection: List[str]

    def __init__(self, source, columns):
        if isinstance(columns, str):
            columns = [columns]
        column_selection = [c for c in columns]
        self.column_selection = column_selection
        unknown = set(column_selection) - set(source.column_names)
        if len(unknown) > 0:
            raise ValueError("selecting unknown columns " + str(unknown))
        if isinstance(source, SelectColumnsNode):
            source = source.sources[0]
        ViewRepresentation.__init__(
            self,
            column_names=column_selection,
            sources=[source],
            node_name="SelectColumnsNode",
        )

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].select_columns(columns=self.column_selection)

    def _equiv_nodes(self, other):
        if not self.column_selection == other.column_selection:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        cols = set(self.column_selection.copy())
        if using is None:
            return [cols]
        return [cols.intersection(using)]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "SelectColumns"
        od["columns"] = self.column_selection
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("select_columns(" + self.column_selection.__repr__() + ")")
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.select_columns_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.select_columns_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class DropColumnsNode(ViewRepresentation):
    column_deletions: List[str]

    def __init__(self, source, column_deletions):
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions
        remaining_columns = [
            c for c in source.column_names if c not in column_deletions
        ]
        unknown = set(column_deletions) - set(source.column_names)
        if len(unknown) > 0:
            raise ValueError("dropping unknown columns " + str(unknown))
        ViewRepresentation.__init__(
            self,
            column_names=remaining_columns,
            sources=[source],
            node_name="DropColumnsNode",
        )

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].drop_columns(column_deletions=self.column_deletions)

    def _equiv_nodes(self, other):
        if not self.column_deletions == other.column_deletions:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        if using is None:
            using = set(self.sources[0].column_names)
        return [set([c for c in using if c not in self.column_deletions])]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "DropColumns"
        od["column_deletions"] = self.column_deletions
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("drop_columns(" + self.column_deletions.__repr__() + ")")
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.drop_columns_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.drop_columns_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class OrderRowsNode(ViewRepresentation):
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
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].order_rows(
            columns=self.order_columns, reverse=self.reverse, limit=self.limit
        )

    def _equiv_nodes(self, other):
        if not self.order_columns == other.order_columns:
            return False
        if not self.reverse == other.reverse:
            return False
        if not self.limit == other.limit:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        cols = set(self.column_names.copy())
        if using is None:
            return [cols]
        cols = cols.intersection(using).union(self.order_columns)
        return [cols]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Order"
        od["order_columns"] = self.order_columns
        od["reverse"] = self.reverse
        od["limit"] = self.limit
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("order_rows(" + self.order_columns.__repr__())
        if len(self.reverse) > 0:
            s = s + ", reverse=" + self.reverse.__repr__()
        if self.limit is not None:
            s = s + ", limit=" + self.limit.__repr__()
        s = s + ")"
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.order_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.order_rows_step(op=self, data_map=data_map, eval_env=eval_env)

    # short-cut main interface

    def is_trivial_when_intermediate(self):
        return self.limit is None


class RenameColumnsNode(ViewRepresentation):
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
        ViewRepresentation.__init__(
            self,
            column_names=column_names,
            sources=[source],
            node_name="RenameColumnsNode",
        )

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].rename(column_remapping=self.column_remapping)

    def _equiv_nodes(self, other):
        if not self.column_remapping == other.column_remapping:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        if using is None:
            using = self.column_names
        cols = [
            (k if k not in self.column_remapping.keys() else self.column_remapping[k])
            for k in using
        ]
        return [set(cols)]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Rename"
        od["column_remapping"] = self.column_remapping
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("rename_columns(" + self.column_remapping.__repr__() + ")")
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.rename_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.rename_columns_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class NaturalJoinNode(ViewRepresentation):
    by: List[str]
    jointype: str

    def __init__(self, a, b, *, by=None, jointype="INNER"):
        # check set of tables is consistent in both sub-dags
        a_tables = a.get_tables()
        b_tables = b.get_tables(replacements=a_tables)
        common_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_keys:
            if a_tables[k] is not b_tables[k]:
                raise ValueError(
                    "Different definition of table object on a/b for: " + k
                )
        sources = [a, b]
        # check columns
        column_names = sources[0].column_names.copy()
        for ci in sources[1].column_names:
            if ci not in sources[0].column_set:
                column_names.append(ci)
        if isinstance(by, str):
            by = [by]
        by_set = set(by)
        if len(by) != len(by_set):
            raise ValueError("duplicate column names in by")
        missing_left = by_set - a.column_set
        if len(missing_left) > 0:
            raise KeyError("left table missing join keys: " + str(missing_left))
        missing_right = by_set - b.column_set
        if len(missing_right) > 0:
            raise KeyError("right table missing join keys: " + str(missing_right))
        ViewRepresentation.__init__(
            self,
            column_names=column_names,
            sources=sources,
            node_name="NaturalJoinNode",
        )
        self.by = by
        self.jointype = data_algebra.expr_rep.standardize_join_type(jointype)
        self.get_tables()  # causes a throw if left and right table descriptions are inconsistent

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].natural_join(
            b=new_sources[1], by=self.by, jointype=self.jointype
        )

    def _equiv_nodes(self, other):
        if not self.by == other.by:
            return False
        if not self.jointype == other.jointype:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        if using is None:
            return [self.sources[i].column_set.copy() for i in range(2)]
        using = using.union(self.by)
        return [self.sources[i].column_set.intersection(using) for i in range(2)]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "NaturalJoin"
        od["by"] = self.by
        od["jointype"] = self.jointype
        od["b"] = self.sources[1].collect_representation_implementation(dialect=dialect)
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = "_0."
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("natural_join(b=\n" + " " * (indent + 6))
        if print_sources:
            s = s + (
                self.sources[1].to_python_implementation(
                    indent=indent + 6, strict=strict
                )
                + ",\n"
                + " " * (indent + 6)
            )
        else:
            s = s + " _1, "
        s = s + (
            "by=" + self.by.__repr__() + ", jointype=" + self.jointype.__repr__() + ")"
        )
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.natural_join_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.natural_join_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class ConcatRowsNode(ViewRepresentation):
    id_column: Union[str, None]

    def __init__(self, a, b, *, id_column="table_name", a_name="a", b_name="b"):
        # check set of tables is consistent in both sub-dags
        a_tables = a.get_tables()
        b_tables = b.get_tables(replacements=a_tables)
        common_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_keys:
            if a_tables[k] is not b_tables[k]:
                raise ValueError(
                    "Different definition of table object on a/b for: " + k
                )
        sources = [a, b]
        # check columns
        if not set(sources[0].column_names) == set(sources[1].column_names):
            raise ValueError("a and b should have same set of column names")
        if id_column is not None and id_column in sources[0].column_names:
            raise ValueError("id_column should not be an input table column name")
        column_names = sources[0].column_names.copy()
        if id_column is not None:
            column_names.append(id_column)
        ViewRepresentation.__init__(
            self, column_names=column_names, sources=sources, node_name="ConcatRowsNode"
        )
        self.id_column = id_column
        self.a_name = a_name
        self.b_name = b_name
        self.get_tables()  # causes a throw if left and right table descriptions are inconsistent

    def apply_to(self, a, *, target_table_key=None):
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
        if not self.id_column == other.id_column:
            return False
        if not self.a_name == other.a_name:
            return False
        if not self.b_name == other.b_name:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        if using is None:
            return [self.sources[i].column_set.copy() for i in range(2)]
        return [self.sources[i].column_set.intersection(using) for i in range(2)]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "ConcatRows"
        od["id_column"] = self.id_column
        od["b"] = self.sources[1].collect_representation_implementation(dialect=dialect)
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = "_0."
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + ("concat_rows(b=\n" + " " * (indent + 6))
        if print_sources:
            s = s + (
                self.sources[1].to_python_implementation(
                    indent=indent + 6, strict=strict
                )
                + ",\n"
                + " " * (indent + 6)
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

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.concat_rows_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.concat_rows_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class ConvertRecordsNode(ViewRepresentation):
    def __init__(self, source, record_map, *, blocks_out_table=None):
        sources = [source]
        if blocks_out_table is None and record_map.blocks_out.control_table is not None:
            blocks_out_table = TableDescription(
                "cdata_temp_record",
                [c for c in record_map.blocks_out.record_keys]
                + [c for c in record_map.blocks_out.control_table.columns],
            )
        if blocks_out_table is not None:
            # check blocks_out_table is a direct table
            if not isinstance(blocks_out_table, TableDescription):
                raise TypeError(
                    "expected blocks_out_table to be a data_algebra.data_ops.TableDescription"
                )
            # ensure table is the exact same definition object if already present
            a_tables = source.get_tables()
            if blocks_out_table.key in a_tables.keys():
                a_table = a_tables[blocks_out_table.key]
                if not a_table.column_set == blocks_out_table.column_set:
                    raise ValueError(
                        "blocks_out_table column definition does not match table already in op DAG"
                    )
                if blocks_out_table is not a_table:
                    blocks_out_table = a_table
            # check blocks_out_table is a direct table
            if not isinstance(blocks_out_table, TableDescription):
                raise TypeError(
                    "expected blocks_out_table to be a data_algebra.data_ops.TableDescription"
                )
            # check it has at least the columns we expect
            expect = [c for c in record_map.blocks_out.record_keys] + [
                c for c in record_map.blocks_out.control_table.columns
            ]
            unknown = set(expect) - set(blocks_out_table.column_names)
            if len(unknown) > 0:
                raise ValueError("blocks_out_table missing columns: " + str(unknown))
            sources = sources + [blocks_out_table]
        self.blocks_out_table = blocks_out_table
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
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].convert_records(
            record_map=self.record_map, blocks_out_table=self.blocks_out_table
        )

    def _equiv_nodes(self, other):
        if not self.blocks_out_table == other.blocks_out_table:
            return False
        if not self.record_map == other.record_map:
            return False
        return True

    def columns_used_from_sources(self, using=None):
        return [
            self.record_map.columns_needed,
            [c for c in self.record_map.blocks_out.record_keys]
            + [c for c in self.record_map.blocks_out.control_table.columns],
        ]

    def collect_representation_implementation(self, *, pipeline=None, dialect="Python"):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "ConvertRecords"
        od["record_map"] = self.record_map.to_simple_obj()
        od["blocks_out_table"] = None
        blocks_out_table = None
        if len(self.sources) > 1:
            blocks_out_table = self.sources[1]
        if blocks_out_table is not None:
            od["blocks_out_table"] = blocks_out_table.collect_representation(
                dialect=dialect
            )[0]
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(
            pipeline=pipeline, dialect=dialect
        )

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        rm_str = self.record_map.__repr__()
        rm_str = re.sub("\n", "\n   ", rm_str)
        s = s + "convert_records(" + rm_str
        if len(self.sources) > 1:
            s = s + (
                "\n,   blocks_out_table="
                + self.sources[1].to_python_implementation(
                    indent=indent + 3, strict=strict
                )
            )
        s = s + ")"
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        sub_query = self.sources[0].to_sql_implementation(
            db_model=db_model, using=using, temp_id_source=temp_id_source
        )
        query = sub_query.to_sql(columns=using, db_model=db_model)
        if self.record_map.blocks_in is not None:
            query = db_model.blocks_to_row_recs_query(
                query, record_spec=self.record_map.blocks_in
            )
        if self.record_map.blocks_out is not None:
            query = db_model.row_recs_to_blocks_query(
                query,
                record_spec=self.record_map.blocks_out,
                record_view=self.sources[1],
            )
        if temp_id_source is None:
            temp_id_source = [0]
        view_name = "convert_records_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLq(
            quoted_query_name=db_model.quote_identifier(view_name), query=query
        )
        return near_sql

    def eval_implementation(self, *, data_map, eval_env, data_model):
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        return data_model.convert_records_step(
            op=self, data_map=data_map, eval_env=eval_env
        )
