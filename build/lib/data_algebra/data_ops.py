from typing import Set, Any, Dict, List
import collections
import re

import pandas

import data_algebra
import data_algebra.data_model
import data_algebra.data_types
import data_algebra.db_model
import data_algebra.pandas_model
import data_algebra.dask_model
import data_algebra.datatable_model
import data_algebra.expr_rep
import data_algebra.pipe
import data_algebra.env

try:
    # noinspection PyUnresolvedReferences
    import black
except ImportError:
    pass

try:
    # noinspection PyUnresolvedReferences
    import sqlparse
except ImportError:
    pass

op_list = [
    "extend",
    "project",
    "natural_join",
    "select_rows",
    "drop_columns",
    "select_columns",
    "rename_columns",
    "order_rows",
    "convert_records",
]


class OperatorPlatform(data_algebra.pipe.PipeValue):
    """Abstract class representing ability to apply data_algebra operations."""

    def __init__(self):
        data_algebra.pipe.PipeValue.__init__(self)

    # Pandas interface

    # noinspection PyPep8Naming
    def transform(self, X):
        raise NotImplementedError("base class called")

    # composition
    def add(self, other):
        if not isinstance(other, data_algebra.pipe.PipeStep):
            raise TypeError("other should be of type data_algebra.pipe.PipeStep")
        return other.apply(self)

    # define builders for all non-initial node types on base class

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        raise NotImplementedError("base class called")

    def project(self, ops=None, *, group_by=None, parse_env=None):
        raise NotImplementedError("base class called")

    def natural_join(self, b, *, by=None, jointype="INNER"):
        raise NotImplementedError("base class called")

    def select_rows(self, expr, parse_env=None):
        raise NotImplementedError("base class called")

    def drop_columns(self, column_deletions):
        raise NotImplementedError("base class called")

    def select_columns(self, columns):
        raise NotImplementedError("base class called")

    def rename_columns(self, column_remapping):
        raise NotImplementedError("base class called")

    def order_rows(self, columns, *, reverse=None, limit=None):
        raise NotImplementedError("base class called")

    def convert_records(self, record_map, *, blocks_out_table=None):
        raise NotImplementedError("base class called")


class ViewRepresentation(OperatorPlatform):
    """Structure to represent the columns of a query or a table.
       Abstract base class."""

    column_names: List[str]
    column_set: Set[str]
    column_map: data_algebra.env.SimpleNamespaceDict
    sources: List[Any]  # actually ViewRepresentation
    columns_currently_used: Set[str]  # transient field, operations can update this

    def __init__(self, column_names, *, sources=None):
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
        OperatorPlatform.__init__(self)

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
        columns_used = {k: v.columns_currently_used.copy() for (k, v) in tables.items()}
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

    def to_python(self, *, indent=0, strict=True, pretty=False, black_mode=None):
        self.columns_used()  # for table consistency check/raise
        if pretty:
            strict = True
        python_str = self.to_python_implementation(
            indent=indent, strict=strict, print_sources=True
        )
        if pretty:
            if data_algebra.have_black:
                if black_mode is None:
                    black_mode = black.FileMode()
                python_str = black.format_str(python_str, mode=black_mode)
            python_str = re.sub(
                "[)].(" + "|".join(op_list) + ")[(]", "\n).\\1(\n    ", python_str
            )
            python_str = re.sub("\n+[ \t]*\n+", "\n", python_str)
        return python_str

    def __repr__(self):
        return self.to_python(strict=True)

    def __str__(self):
        return self.to_python(strict=True)

    # query generation

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        raise NotImplementedError("base method called")

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
        if pretty and data_algebra.have_sqlparse:
            sql_str = sqlparse.format(sql_str, encoding=encoding, **sqlparse_options)
        return sql_str

    # Pandas realization

    def eval_implementation(self, *, data_map, eval_env, data_model):
        raise NotImplementedError("base method called")

    def eval_pandas(self, data_map, *, eval_env=None, data_model=None):
        """
        Evaluate operators with respect to Pandas data frames.
        :param data_map: map from table names to data frames
        :param eval_env: environment to evaluate in
        :param data_model: adaptor to Pandas dialect (possibly dask)
        :return:
        """

        if not isinstance(data_map, Dict):
            raise TypeError("data_map should be a dictionary")
        if len(data_map) < 1:
            raise ValueError("data_map should not be empty")
        if eval_env is None:
            eval_env = data_algebra.env.outer_namespace()
        if eval_env is None:
            eval_env = globals()
        if data_model is None:
            data_model = data_algebra.pandas_model.PandasModel()
        if not isinstance(data_model, data_algebra.pandas_model.PandasModel):
            raise TypeError(
                "Expected data_model to derive from data_algebra.pandas_model.PandasModel"
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

    def eval_dask(self, data_map, *, eval_env=None, data_model=None):
        """
        Evaluate operators with respect to dask data frames.
        :param data_map: map from table names to data frames
        :param eval_env: environment to evaluate in
        :param data_model: adaptor to Pandas dialect (possibly dask)
        :return:
        """

        if not isinstance(data_map, Dict):
            raise TypeError("data_map should be a dictionary")
        if len(data_map) < 1:
            raise ValueError("data_map should not be empty")
        if eval_env is None:
            eval_env = data_algebra.env.outer_namespace()
        if eval_env is None:
            eval_env = globals()
        if data_model is None:
            data_model = data_algebra.dask_model.DaskModel()
        if not isinstance(data_model, data_algebra.dask_model.DaskModel):
            raise TypeError(
                "Expected data_model to derive from data_algebra.dask_model.DaskModel"
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

    def eval_datatable(self, data_map, *, eval_env=None, data_model=None):
        """
        Evaluate operators with respect to Python datatable data frames.
        :param data_map: map from table names to data frames
        :param eval_env: environment to evaluate in
        :param data_model: adaptor to Pandas dialect (possibly dask)
        :return:
        """

        if not isinstance(data_map, Dict):
            raise TypeError("data_map should be a dictionary")
        if len(data_map) < 1:
            raise ValueError("data_map should not be empty")
        if eval_env is None:
            eval_env = data_algebra.env.outer_namespace()
        if eval_env is None:
            eval_env = globals()
        if data_model is None:
            data_model = data_algebra.datatable_model.DataTableModel()
        if not isinstance(data_model, data_algebra.datatable_model.DataTableModel):
            raise TypeError(
                "Expected data_model to derive from data_algebra.datatable_model.DataTableModel"
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
        if data_model is not None:
            if not isinstance(data_model, data_algebra.data_model.DataModel):
                raise TypeError(
                    "Expected data_model to be derived from data_algebra.data_model.DataModel"
                )
        k = [k for k in data_map.keys()][0]
        x = data_map[k]
        if isinstance(x, pandas.DataFrame):
            return self.eval_pandas(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        if data_algebra.data_types.is_dask_data_frame(x):
            return self.eval_dask(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        if data_algebra.data_types.is_datatable_frame(x):
            return self.eval_datatable(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        raise TypeError("can not apply eval() to type " + str(type(x)))

    # implement builders for all non-initial node types on base class

    # noinspection PyPep8Naming
    def transform(self, X, *, eval_env=None, data_model=None):
        if data_model is not None:
            if not isinstance(data_model, data_algebra.data_model.DataModel):
                raise TypeError(
                    "Expected data_model to be derived from data_algebra.data_model.DataModel"
                )
        self.columns_used()  # for table consistency check/raise
        tables = self.get_tables()
        if len(tables) != 1:
            raise ValueError(
                "transfrom(pandas.DataFrame) can only be applied to ops-dags with only one table def"
            )
        k = [k for k in tables.keys()][0]
        data_map = {k: X}
        if isinstance(X, pandas.DataFrame):
            return self.eval_pandas(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        if data_algebra.data_types.is_dask_data_frame(X):
            return self.eval_dask(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        if data_algebra.data_types.is_datatable_frame(X):
            return self.eval_datatable(
                data_map=data_map, eval_env=eval_env, data_model=data_model
            )
        raise TypeError("can not apply transform() to type " + str(type(X)))

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    # nodes

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        return ExtendNode(
            source=self,
            ops=ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse,
            parse_env=parse_env,
        )

    def project(self, ops=None, *, group_by=None, parse_env=None):
        return ProjectNode(source=self, ops=ops, group_by=group_by, parse_env=parse_env)

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, ViewRepresentation):
            raise TypeError(
                "expected b to be a data_algebra.dat_ops.ViewRepresentation"
            )
        return NaturalJoinNode(a=self, b=b, by=by, jointype=jointype)

    def select_rows(self, expr, parse_env=None):
        return SelectRowsNode(source=self, expr=expr, parse_env=parse_env)

    def drop_columns(self, column_deletions):
        return DropColumnsNode(source=self, column_deletions=column_deletions)

    def select_columns(self, columns):
        return SelectColumnsNode(source=self, columns=columns)

    def rename_columns(self, column_remapping):
        return RenameColumnsNode(source=self, column_remapping=column_remapping)

    def order_rows(self, columns, *, reverse=None, limit=None):
        return OrderRowsNode(source=self, columns=columns, reverse=reverse, limit=limit)

    def convert_records(self, record_map, *, blocks_out_table=None):
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
    qualifiers: Dict[str, str]
    key: str

    def __init__(self, table_name, column_names, *, qualifiers=None):
        ViewRepresentation.__init__(self, column_names=column_names)
        if (table_name is not None) and (not isinstance(table_name, str)):
            raise TypeError("table_name must be a string")
        self.table_name = table_name
        if isinstance(column_names, str):
            column_names = [column_names]
        self.column_names = [c for c in column_names]
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
        self.key = key + self.table_name

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
        nc = min(len(self.column_names), 20)
        if (not strict) and (nc < len(self.column_names)):
            cols_str = (
                "["
                + ", ".join([self.column_names[i].__repr__() for i in range(nc)])
                + ", + "
                + str(len(self.column_names) - nc)
                + " more]"
            )
        else:
            cols_str = self.column_names.__repr__()
        s = (
            "TableDescription("
            + "table_name="
            + self.table_name.__repr__()
            + ", column_names="
            + cols_str
        )
        if len(self.qualifiers) > 0:
            s = s + ", qualifiers=" + self.qualifiers.__repr__()
        s = s + ")"
        return s

    def get_tables(self, *, replacements=None):
        """get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent"""
        if replacements is not None and self.key in replacements.keys():
            return {self.key: replacements[self.key]}
        return {self.key: self}

    def eval_implementation(self, *, data_map, eval_env, data_model):
        return data_model.table_step(op=self, data_map=data_map, eval_env=eval_env)

    def columns_used_from_sources(self, using=None):
        return []  # no inputs to table description

    def to_sql(self, db_model, *, pretty=False, encoding=None, sqlparse_options=None):
        if sqlparse_options is None:
            sqlparse_options = {"reindent": True, "keyword_case": "upper"}
        self.columns_used()  # for table consistency check/raise
        temp_id_source = [0]
        sql_str = self.to_sql_implementation(
            db_model=db_model, using=None, temp_id_source=temp_id_source, force_sql=True
        )
        if pretty and data_algebra.have_sqlparse:
            sql_str = sqlparse.format(sql_str, encoding=encoding, **sqlparse_options)
        return sql_str

    def to_sql_implementation(
        self, db_model, *, using, temp_id_source, force_sql=False
    ):
        return db_model.table_def_to_sql(self, using=using, force_sql=force_sql)

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


def describe_table(d, table_name="data_frame"):
    if isinstance(d, pandas.DataFrame):
        return TableDescription(table_name, [c for c in d.columns])
    if data_algebra.data_types.is_dask_data_frame(d):
        return TableDescription(table_name, [c for c in d.columns])
    if data_algebra.data_types.is_datatable_frame(d):
        return TableDescription(table_name, [c for c in d.names])
    raise TypeError("can't describe " + table_name + ": " + str(type(d)))


class ExtendNode(ViewRepresentation):
    def __init__(
        self,
        source,
        ops,
        *,
        partition_by=None,
        order_by=None,
        reverse=None,
        parse_env=None
    ):
        ops = data_algebra.expr_rep.parse_assignments_in_context(
            ops, source, parse_env=parse_env
        )
        if len(ops) < 1:
            raise ValueError("no ops")
        self.ops = ops
        if partition_by is None:
            partition_by = []
        if isinstance(partition_by, str):
            partition_by = [partition_by]
        self.partition_by = partition_by
        if order_by is None:
            order_by = []
        if isinstance(order_by, str):
            order_by = [order_by]
        self.order_by = order_by
        if reverse is None:
            reverse = []
        if isinstance(reverse, str):
            reverse = [reverse]
        self.reverse = reverse
        column_names = source.column_names.copy()
        consumed_cols = set()
        for (k, o) in ops.items():
            o.get_column_names(consumed_cols)
        unknown_cols = consumed_cols - source.column_set
        if len(unknown_cols) > 0:
            raise KeyError("referred to unknown columns: " + str(unknown_cols))
        known_cols = set(column_names)
        for ci in ops.keys():
            if ci not in known_cols:
                column_names.append(ci)
        if len(partition_by) != len(set(partition_by)):
            raise ValueError("Duplicate name in partition_by")
        if len(order_by) != len(set(order_by)):
            raise ValueError("Duplicate name in order_by")
        if len(reverse) != len(set(reverse)):
            raise ValueError("Duplicate name in reverse")
        unknown = set(partition_by) - known_cols
        if len(unknown) > 0:
            raise ValueError("unknown partition_by columns: " + str(unknown))
        unknown = set(order_by) - known_cols
        if len(unknown) > 0:
            raise ValueError("unknown order_by columns: " + str(unknown))
        unknown = set(reverse) - set(order_by)
        if len(unknown) > 0:
            raise ValueError("reverse columns not in order_by: " + str(unknown))
        bad_overwrite = set(ops.keys()).intersection(
            set(partition_by).union(order_by, reverse)
        )
        if len(bad_overwrite) > 0:
            raise ValueError("tried to change: " + str(bad_overwrite))
        ViewRepresentation.__init__(self, column_names=column_names, sources=[source])

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
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + (
            "extend({"
            + ", ".join(
                [
                    k.__repr__() + ": " + opi.to_python().__repr__()
                    for (k, opi) in self.ops.items()
                ]
            )
            + "}"
        )
        if len(self.partition_by) > 0:
            s = s + ", partition_by=" + self.partition_by.__repr__()
        if len(self.order_by) > 0:
            s = s + ", order_by=" + self.order_by.__repr__()
        if len(self.reverse) > 0:
            s = s + ", reverse=" + self.reverse.__repr__()
        s = s + ")"
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.extend_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, eval_env, data_model):
        return data_model.extend_step(op=self, data_map=data_map, eval_env=eval_env)


class ProjectNode(ViewRepresentation):
    def __init__(self, source, ops=None, *, group_by=None, parse_env=None):
        if ops is None:
            ops = {}
        ops = data_algebra.expr_rep.parse_assignments_in_context(
            ops, source, parse_env=parse_env
        )
        self.ops = ops
        if group_by is None:
            group_by = []
        if isinstance(group_by, str):
            group_by = [group_by]
        self.group_by = group_by
        column_names = group_by.copy()
        consumed_cols = set()
        for c in group_by:
            consumed_cols.add(c)
        for (k, o) in ops.items():
            o.get_column_names(consumed_cols)
        unknown_cols = consumed_cols - source.column_set
        if len(unknown_cols) > 0:
            raise KeyError("referred to unknown columns: " + str(unknown_cols))
        known_cols = set(column_names)
        for ci in ops.keys():
            if ci not in known_cols:
                column_names.append(ci)
        if len(group_by) != len(set(group_by)):
            raise ValueError("Duplicate name in group_by")
        unknown = set(group_by) - known_cols
        if len(unknown) > 0:
            raise ValueError("unknown partition_by columns: " + str(unknown))
        ViewRepresentation.__init__(self, column_names=column_names, sources=[source])

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
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (indent + 3)
            )
        s = s + (
            "project({"
            + ", ".join(
                [
                    k.__repr__() + ": " + opi.to_python().__repr__()
                    for (k, opi) in self.ops.items()
                ]
            )
            + "}"
        )
        if len(self.group_by) > 0:
            s = s + ", group_by=" + self.group_by.__repr__()
        s = s + ")"
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.project_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, eval_env, data_model):
        return data_model.project_step(op=self, data_map=data_map, eval_env=eval_env)


class SelectRowsNode(ViewRepresentation):
    expr: data_algebra.expr_rep.Expression
    decision_columns: Set[str]

    def __init__(self, source, expr, *, parse_env=None):
        ops = data_algebra.expr_rep.parse_assignments_in_context(
            {"expr": expr}, source, parse_env=parse_env
        )
        if len(ops) < 1:
            raise ValueError("no ops")
        self.expr = ops["expr"]
        self.decision_columns = set()
        self.expr.get_column_names(self.decision_columns)
        ViewRepresentation.__init__(
            self, column_names=source.column_names, sources=[source]
        )

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
            self, column_names=column_selection, sources=[source]
        )

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
            self, column_names=remaining_columns, sources=[source]
        )

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
            self, column_names=source.column_names, sources=[source]
        )

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
        return data_model.order_rows_step(op=self, data_map=data_map, eval_env=eval_env)


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
        ViewRepresentation.__init__(self, column_names=column_names, sources=[source])

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
        return data_model.rename_columns_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class NaturalJoinNode(ViewRepresentation):
    by: List[str]
    jointype: str

    def __init__(self, a, b, *, by=None, jointype="INNER"):
        a_tables = a.get_tables()
        b_tables = b.get_tables(replacements=a_tables)
        common_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_keys:
            if a_tables[k] is not b_tables[k]:
                raise ValueError(
                    "Different definition of table object on a/b for: " + k
                )
        sources = [a, b]
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
        self.by = by
        self.jointype = jointype
        ViewRepresentation.__init__(self, column_names=column_names, sources=sources)

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
        return data_model.natural_join_step(
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
        self.record_map = record_map
        unknown = set(self.record_map.columns_needed) - set(source.column_names)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        ViewRepresentation.__init__(
            self, column_names=record_map.columns_produced, sources=sources
        )

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
        s = s + "convert_record(" + rm_str
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
        res = self.sources[0].to_sql_implementation(
            db_model=db_model, using=using, temp_id_source=temp_id_source
        )
        if self.record_map.blocks_in is not None:
            res = db_model.blocks_to_row_recs_query(
                res, record_spec=self.record_map.blocks_in
            )
        if self.record_map.blocks_out is not None:
            res = db_model.row_recs_to_blocks_query(
                res, record_spec=self.record_map.blocks_out, record_view=self.sources[1]
            )
        return res

    def eval_implementation(self, *, data_map, eval_env, data_model):
        return data_model.convert_records_step(
            op=self, data_map=data_map, eval_env=eval_env
        )


class Extend(data_algebra.pipe.PipeStep):
    """Class to specify adding or altering columns.

       If outer namespace is set user values are visible and
       _-side effects can be written back.

       Example:
           from data_algebra.data_ops import *
           import data_algebra.env
           with data_algebra.env.Env(locals()) as env:
               q = 4
               x = 2
               var_name = 'y'

               print("first example")
               ops = (
                  TableDescription('d', ['x', 'y']) .
                     extend({'z':_.x + _[var_name]/q + _get('x')})
                )
                print(ops)
    """

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops, *, partition_by=None, order_by=None, reverse=None):
        if isinstance(partition_by, str):
            partition_by = [partition_by]
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(reverse, str):
            reverse = [reverse]
        if reverse is not None and len(reverse) > 0:
            if order_by is None:
                raise ValueError("set is None when order_by is not None")
            unknown = set(reverse) - set(order_by)
            if len(unknown) > 0:
                raise ValueError(
                    "columns in reverse that are not in order_by: " + str(unknown)
                )
        data_algebra.pipe.PipeStep.__init__(self, name="Extend")
        self._ops = ops
        self.partition_by = partition_by
        self.order_by = order_by
        self.reverse = reverse

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.extend(
            ops=self._ops,
            partition_by=self.partition_by,
            order_by=self.order_by,
            reverse=self.reverse,
            parse_env=parse_env,
        )

    def __repr__(self):
        return (
            "Extend("
            + self._ops.__repr__()
            + ", partition_by="
            + self.partition_by.__repr__()
            + ", order_by="
            + self.order_by.__repr__()
            + ", reverse="
            + self.reverse.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class Project(data_algebra.pipe.PipeStep):
    """Class to specify aggregating or summarizing columns."""

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops=None, *, group_by=None):
        if isinstance(group_by, str):
            group_by = [group_by]
        if ops is None:
            ops = {}
        data_algebra.pipe.PipeStep.__init__(self, name="Project")
        self._ops = ops
        self.group_by = group_by

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.project(ops=self._ops, group_by=self.group_by, parse_env=parse_env)

    def __repr__(self):
        return (
            "Project("
            + self._ops.__repr__()
            + ", group_by="
            + self.group_by.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class SelectRows(data_algebra.pipe.PipeStep):
    """Class to specify a choice of rows.
    """

    expr: data_algebra.expr_rep.Expression

    def __init__(self, expr):
        data_algebra.pipe.PipeStep.__init__(self, name="SelectRows")
        self.expr = expr

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        parse_env = kwargs.get("parse_env", None)
        return other.select_rows(expr=self.expr, parse_env=parse_env)

    def __repr__(self):
        return "SelectRows(" + self.expr.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class SelectColumns(data_algebra.pipe.PipeStep):
    """Class to specify a choice of columns.
    """

    column_selection: List[str]

    def __init__(self, columns):
        if isinstance(columns, str):
            columns = [columns]
        column_selection = [c for c in columns]
        self.column_selection = column_selection
        data_algebra.pipe.PipeStep.__init__(self, name="SelectColumns")

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.select_columns(self.column_selection)

    def __repr__(self):
        return "SelectColumns(" + self.column_selection.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class DropColumns(data_algebra.pipe.PipeStep):
    """Class to specify removal of columns.
    """

    column_deletions: List[str]

    def __init__(self, column_deletions):
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions
        data_algebra.pipe.PipeStep.__init__(self, name="DropColumns")

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.drop_columns(self.column_deletions)

    def __repr__(self):
        return "DropColumns(" + self.column_deletions.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class OrderRows(data_algebra.pipe.PipeStep):
    """Class to specify a columns to determine row order.
    """

    order_columns: List[str]
    reverse: List[str]

    def __init__(self, columns, *, reverse=None, limit=None):
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(reverse, str):
            reverse = [reverse]
        if reverse is not None and len(reverse) > 0:
            if columns is None:
                raise ValueError("set is None when order_by is not None")
            unknown = set(reverse) - set(columns)
            if len(unknown) > 0:
                raise ValueError(
                    "columns in reverse that are not in order_by: " + str(unknown)
                )
        self.order_columns = [c for c in columns]
        if reverse is None:
            reverse = []
        self.reverse = [c for c in reverse]
        self.limit = limit
        data_algebra.pipe.PipeStep.__init__(self, name="OrderRows")

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.order_rows(
            columns=self.order_columns, reverse=self.reverse, limit=self.limit
        )

    def __repr__(self):
        return (
            "OrderRows("
            + self.order_columns.__repr__()
            + ", reverse="
            + self.reverse.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class RenameColumns(data_algebra.pipe.PipeStep):
    """Class to rename columns.
    """

    column_remapping: Dict[str, str]

    def __init__(self, column_remapping):
        self.column_remapping = column_remapping.copy()
        data_algebra.pipe.PipeStep.__init__(self, name="RenameColumns")

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.rename_columns(column_remapping=self.column_remapping)

    def __repr__(self):
        return "RenameColumns(" + self.column_remapping.__repr__() + ")"

    def __str__(self):
        return self.__repr__()


class NaturalJoin(data_algebra.pipe.PipeStep):
    _by: List[str]
    _jointype: str
    _b: OperatorPlatform

    def __init__(self, *, b=None, by=None, jointype="INNER"):
        if not isinstance(b, ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        if isinstance(by, str):
            by = [by]
        self._by = by
        self._jointype = jointype
        self._b = b
        data_algebra.pipe.PipeStep.__init__(self, name="NaturalJoin")

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.natural_join(b=self._b, by=self._by, jointype=self._jointype)

    def __repr__(self):
        return (
            "NaturalJoin("
            + "b="
            + self._b.__repr__()
            + ", by="
            + self._by.__repr__()
            + ", jointype="
            + self._jointype.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class ConvertRecords(data_algebra.pipe.PipeStep):
    def __init__(self, record_map, *, blocks_out_table=None):
        self.record_map = record_map
        self.blocks_out_table = blocks_out_table
        data_algebra.pipe.PipeStep.__init__(self, name="ConvertRecords")

    def apply(self, other, **kwargs):
        if not isinstance(other, OperatorPlatform):
            raise TypeError(
                "expected other to be a data_algebra.data_ops.OperatorPlatform"
            )
        return other.convert_records(
            record_map=self.record_map, blocks_out_table=self.blocks_out_table
        )

    def __repr__(self):
        return (
            "ConvertRecords("
            + self.record_map.__repr__()
            + ", record_map="
            + self.record_map.__repr__()
            + ", blocks_out_table="
            + self.blocks_out_table.__repr__()
            + ")"
        )

    def __str__(self):
        return self.__repr__()


class Locum(OperatorPlatform):
    """Class to represent future opertions."""

    def __init__(self):
        OperatorPlatform.__init__(self)
        self.ops = []

    def apply_to(self, pipeline):
        if not isinstance(pipeline, OperatorPlatform):
            raise TypeError(
                "Expected othter to be a data_algebra.data_ops.OperatorPlatform"
            )
        for s in self.ops:
            # pipeline = pipeline >> s
            pipeline = s.apply(pipeline)
        return pipeline

    def append(self, other):
        if isinstance(other, Locum):
            for o in other.ops:
                self.ops.append(o)
        elif isinstance(other, data_algebra.pipe.PipeStep):
            self.ops.append(other)
        else:
            raise TypeError("unexpeted type for Locum + " + str(type(other)))
        return self

    def realize(self, x):
        pipeline = describe_table(x, table_name="x")
        return self.apply_to(pipeline)

    # noinspection PyPep8Naming
    def transform(self, X):
        if isinstance(X, OperatorPlatform):
            return self.apply_to(X)
        pipeline = self.realize(X)
        return pipeline.transform(X)

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __add__(self, other):  # override self + other
        res = Locum()
        res.append(self)
        res.append(other)
        return res

    def __radd__(self, other):  # override other + self
        return self.apply_to(other)

    # print

    def __repr__(self):
        return "[\n    " + "\n    ".join([str(o) + "," for o in self.ops]) + "\n]"

    def __str__(self):
        return "[\n    " + "\n    ".join([str(o) + "," for o in self.ops]) + "\n]"

    # implement method chaining collection of pending operations

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        if parse_env is not None:
            raise ValueError("Expected parse_env to be None")
        op = Extend(
            ops=ops, partition_by=partition_by, order_by=order_by, reverse=reverse
        )
        self.ops.append(op)
        return self

    def project(self, ops=None, *, group_by=None, parse_env=None):
        if parse_env is not None:
            raise ValueError("Expected parse_env to be None")
        if ops is None:
            ops = {}
        op = Project(ops=ops, group_by=group_by)
        self.ops.append(op)
        return self

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        op = NaturalJoin(by=by, jointype=jointype, b=b)
        self.ops.append(op)
        return self

    def select_rows(self, expr, parse_env=None):
        if parse_env is not None:
            raise ValueError("Expected parse_env to be None")
        op = SelectRows(expr=expr)
        self.ops.append(op)
        return self

    def drop_columns(self, column_deletions):
        op = DropColumns(column_deletions=column_deletions)
        self.ops.append(op)
        return self

    def select_columns(self, columns):
        op = SelectColumns(columns=columns)
        self.ops.append(op)
        return self

    def rename_columns(self, column_remapping):
        op = RenameColumns(column_remapping=column_remapping)
        self.ops.append(op)
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        op = OrderRows(columns=columns, reverse=reverse, limit=limit)
        self.ops.append(op)
        return self

    def convert_records(self, record_map, *, blocks_out_table=None):
        op = ConvertRecords(record_map=record_map, blocks_out_table=blocks_out_table)
        self.ops.append(op)
        return self


def wrap_pipeline(ops):
    return ops.apply(Locum())


def wrap_ops(*args):
    r = Locum()
    for s in args:
        s.apply(r)
    return r
