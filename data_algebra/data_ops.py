from typing import Set, Any, Dict, List
import collections
import re
import copy

import pandas

import data_algebra
import data_algebra.flow_text
import data_algebra.data_model
import data_algebra.data_types
import data_algebra.db_model
import data_algebra.pandas_model
import data_algebra.dask_model
import data_algebra.datatable_model
import data_algebra.expr_rep
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


class OperatorPlatform:
    """Abstract class representing ability to apply data_algebra operations."""

    def __init__(self):
        pass

    # noinspection PyPep8Naming
    def transform(self, X):
        raise NotImplementedError("base class called")

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __rshift__(self, other):  # override self >> other
        # can't use type >> type if only __rrshift__ is defined (must have __rshift__ in this case)
        if isinstance(other, OperatorPlatform):
            return other.transform(self)
        if isinstance(other, PipeStep):
            other.apply(self)
        raise TypeError("unexpected type: " + str(type(other)))

    # composition
    def add(self, other):
        """interface to what we used to call PipeStep nodes"""
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

    # noinspection PyBroadException
    def to_python(self, *, indent=0, strict=True, pretty=False, black_mode=None):
        self.columns_used()  # for table consistency check/raise
        if pretty:
            strict = True
        python_str = self.to_python_implementation(
            indent=indent, strict=strict, print_sources=True
        )
        if pretty:
            if data_algebra.have_black:
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
        if pretty and data_algebra.have_sqlparse:
            try:
                sql_str = sqlparse.format(sql_str, encoding=encoding, **sqlparse_options)
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

    def stand_in_for_table(self, ops, table_key):
        """re-write ops replacing any TableDescription with matching id with self"""
        if isinstance(ops, data_algebra.data_ops.TableDescription):
            if ops.key == table_key:
                return self
            else:
                return ops
        node = copy.copy(ops)
        node.sources = [self.stand_in_for_table(ops=s, table_key=table_key) for s in node.sources]
        return node

    # noinspection PyPep8Naming
    def transform(self, X, *, eval_env=None, data_model=None):
        if data_model is not None:
            if not isinstance(data_model, data_algebra.data_model.DataModel):
                raise TypeError(
                    "Expected data_model to be derived from data_algebra.data_model.DataModel"
                )
        cols_used = self.columns_used()  # for table consistency check/raise
        tables = self.get_tables()
        if len(tables) != 1:
            raise ValueError(
                "transfrom(pandas.DataFrame) can only be applied to ops-dags with only one table def"
            )
        k = [k for k in tables.keys()][0]
        if isinstance(X, ViewRepresentation):
            # replace self input table with X
            incoming_columns = cols_used[k]
            missing = set(incoming_columns) - set(X.column_names)
            if len(missing) > 0:
                raise ValueError("missing required columns: " + str(missing))
            excess = set(X.column_names) - set(incoming_columns)
            if len(excess):
                # insert a select columns node to get the match columns
                X = X.select_columns([c for c in incoming_columns])
            # check categorical arrow composition conditions
            if set(incoming_columns) != set(X.column_names):
                raise ValueError("arrow composition conditions not met (incoming column set doesn't match outgoing)")
            res = X.stand_in_for_table(ops=self, table_key=k)
            return res
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
    column_names: List[str]
    qualifiers: Dict[str, str]
    key: str

    def __init__(self, table_name, column_names,
                 *,
                 qualifiers=None,
                 column_types=None):
        ViewRepresentation.__init__(self, column_names=column_names)
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
        spacer = '\n ' + ' ' * indent
        column_limit = 20
        truncated = (not strict) and (column_limit < len(self.column_names))
        if truncated:
            cols_to_print = [self.column_names[i].__repr__() for i in range(column_limit)] + \
                            ['+ ' + str(len(self.column_names)) + ' more']
        else:
            cols_to_print = [c.__repr__() for c in self.column_names]
        col_text = data_algebra.flow_text.flow_text(cols_to_print,
                                                    align_right=70 - indent,
                                                    sep_width=2)
        col_text = [', '.join(line) for line in col_text]
        col_text = (',  ' + spacer).join(col_text)
        s = (
                "TableDescription("
                + spacer + "table_name=" + self.table_name.__repr__() + ","
                + spacer + "column_names=["
                + spacer + '  ' + col_text + "]"
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
        column_names = [c for c in d.columns]
        column_types = None
        if d.shape[0] > 0:
            column_types = {k: type(d.loc[0, k]) for k in column_names}
        return TableDescription(table_name, column_names,
                                column_types=column_types)
    if data_algebra.data_types.is_dask_data_frame(d):
        return TableDescription(table_name, [c for c in d.columns])
    if data_algebra.data_types.is_datatable_frame(d):
        return TableDescription(table_name, [c for c in d.names])
    raise TypeError("can't describe " + table_name + ": " + str(type(d)))


class WrappedOperatorPlatform(OperatorPlatform):
    """Decorator class for OperatorPlatform."""

    def __init__(self, *, underlying, data_map):
        OperatorPlatform.__init__(self)
        if not isinstance(underlying, OperatorPlatform):
            raise TypeError("Expected underlying to be of class OperatorPlatform")
        if isinstance(underlying, WrappedOperatorPlatform):
            raise TypeError("underlying should not be a WrappedOperatorPlatform")
        if not isinstance(data_map, dict):
            raise TypeError("Expected data_map to be a key to table dictionary")
        self.data_map = data_map.copy()
        self.underlying = underlying

    # execution

    def ex(self):
        """Execute pipeline against internally stored values"""
        tables = self.underlying.get_tables()
        data_map = self.data_map
        missing = set(tables.keys()) - set(data_map.keys())
        res = self.underlying.eval(data_map=data_map)
        return res

    # general functions

    def __repr__(self):
        return "[" + self.underlying.__repr__() + "](\n " + set(self.data_map.keys()).__repr__() + ")"

    def __str__(self):
        return "[" + self.underlying.__str__() + "](\n " + set(self.data_map.keys()).__repr__() + ")"

    # overrides

    def _reach_in(self, other):
        data_map = self.data_map.copy()
        if isinstance(other, WrappedOperatorPlatform):
            data_map.update(other.data_map)
            other = other.underlying
        return data_map, other

    # noinspection PyPep8Naming
    def transform(self, X):
        data_map, X = self._reach_in(X)
        return WrappedOperatorPlatform(
            underlying=self.underlying.transform(X),
            data_map=data_map
        )

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __rshift__(self, other):  # override self >> other
        # can't use type >> type if only __rrshift__ is defined (must have __rshift__ in this case)
        data_map, other = self._reach_in(other)
        if isinstance(other, OperatorPlatform):
            return WrappedOperatorPlatform(
                underlying=other.transform(self.underlying),
                data_map=data_map)
        if isinstance(other, PipeStep):
            return WrappedOperatorPlatform(
                underlying=other.apply(self.underlying),
                data_map=data_map)
        raise TypeError("unexpected type: " + str(type(other)))

    # composition
    def add(self, other):
        """interface to what we used to call PipeStep nodes"""
        data_map, other = self._reach_in(other)
        return WrappedOperatorPlatform(
            underlying=other.apply(self.underlying),
            data_map=data_map)

    # define builders for all non-initial node types on base class

    def extend(
            self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        return WrappedOperatorPlatform(
            underlying=self.underlying.extend(ops=ops,
                                   partition_by=partition_by,
                                   order_by=order_by,
                                   reverse=reverse,
                                   parse_env=parse_env),
            data_map=self.data_map)

    def project(self, ops=None, *, group_by=None, parse_env=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.project(ops=ops,
                                    group_by=group_by,
                                    parse_env=parse_env),
            data_map=self.data_map)

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, WrappedOperatorPlatform):
            raise TypeError("expected b to be of type WrappedOperatorPlatform")
        data_map, b = self._reach_in(b)
        return WrappedOperatorPlatform(
            underlying=self.underlying.natural_join(b=b,
                                         by=by,
                                         jointype=jointype),
            data_map=data_map)

    def select_rows(self, expr, parse_env=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.select_rows(expr=expr,
                                        parse_env=parse_env),
            data_map=self.data_map)

    def drop_columns(self, column_deletions):
        return WrappedOperatorPlatform(
            underlying=self.underlying.drop_columns(column_deletions=column_deletions),
            data_map=self.data_map)

    def select_columns(self, columns):
        return WrappedOperatorPlatform(
            underlying=self.underlying.select_columns(columns=columns),
            data_map=self.data_map)

    def rename_columns(self, column_remapping):
        return WrappedOperatorPlatform(
            underlying=self.underlying.rename_columns(column_remapping=column_remapping),
            data_map=self.data_map)

    def order_rows(self, columns, *, reverse=None, limit=None):
        return WrappedOperatorPlatform(
            underlying=self.underlying.order_rows(columns=columns,
                                       reverse=reverse,
                                       limit=limit),
            data_map=self.data_map)

    def convert_records(self, record_map, *, blocks_out_table=None):
        data_map, blocks_out_table = self._reach_in(blocks_out_table)
        return WrappedOperatorPlatform(
            underlying=self.underlying.convert_records(record_map=record_map,
                                            blocks_out_table=blocks_out_table),
            data_map=data_map)


def wrap(d, *, table_name="data_frame"):
    if isinstance(d, pandas.DataFrame):
        column_names = [c for c in d.columns]
        column_types = None
        if d.shape[0] > 0:
            column_types = {k: type(d.loc[0, k]) for k in column_names}
        return WrappedOperatorPlatform(
            underlying=TableDescription(table_name, column_names,
                                        column_types=column_types),
            data_map={table_name: d})
    if data_algebra.data_types.is_dask_data_frame(d):
        return WrappedOperatorPlatform(
            TableDescription(table_name, [c for c in d.columns]),
            data_map={table_name: d})
    if data_algebra.data_types.is_datatable_frame(d):
        return WrappedOperatorPlatform(
            TableDescription(table_name, [c for c in d.names]),
            data_map={table_name: d})
    raise TypeError("can't wrap " + table_name + ": " + str(type(d)))


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
        spacer = '\n   ' + ' ' * indent
        s = ""
        if print_sources:
            s = (
                    self.sources[0].to_python_implementation(indent=indent, strict=strict)
                    + " .\\"
                    + spacer
            )
        ops = [k.__repr__() + ": " + opi.to_python().__repr__() for (k, opi) in self.ops.items()]
        flowed = (',' + spacer + ' ').join(ops)
        s = s + (
                "extend({"
                + spacer + ' ' + flowed
                + "}"
        )
        if len(self.partition_by) > 0:
            s = s + "," + spacer + "partition_by=" + self.partition_by.__repr__()
        if len(self.order_by) > 0:
            s = s + "," + spacer + "order_by=" + self.order_by.__repr__()
        if len(self.reverse) > 0:
            s = s + "," + spacer + "reverse=" + self.reverse.__repr__()
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
        spacer = '\n   ' + ' ' * indent
        s = ""
        if print_sources:
            s = (
                    self.sources[0].to_python_implementation(indent=indent, strict=strict)
                    + " .\\\n"
                    + " " * (indent + 3)
            )
        s = s + (
                "project({" + spacer + ' '
                + ("," + spacer + ' ').join(
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
        self.jointype = data_algebra.expr_rep.standardize_join_type(jointype)
        ViewRepresentation.__init__(self, column_names=column_names, sources=sources)
        self.get_tables()  # causes a throw if left and right table descriptions are inconsistent

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
        self.blocks_out_table = blocks_out_table
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


class PipeStep:
    def __init__(self):
        pass

    def apply(self, other, **kwargs):
        raise NotImplementedError("base class called")


class Extend(PipeStep):
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
        PipeStep.__init__(self)
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


class Project(PipeStep):
    """Class to specify aggregating or summarizing columns."""

    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, ops=None, *, group_by=None):
        PipeStep.__init__(self)
        if isinstance(group_by, str):
            group_by = [group_by]
        if ops is None:
            ops = {}
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


class SelectRows(PipeStep):
    """Class to specify a choice of rows.
    """

    expr: data_algebra.expr_rep.Expression

    def __init__(self, expr):
        PipeStep.__init__(self)
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


class SelectColumns(PipeStep):
    """Class to specify a choice of columns.
    """

    column_selection: List[str]

    def __init__(self, columns):
        PipeStep.__init__(self)
        if isinstance(columns, str):
            columns = [columns]
        column_selection = [c for c in columns]
        self.column_selection = column_selection

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


class DropColumns(PipeStep):
    """Class to specify removal of columns.
    """

    column_deletions: List[str]

    def __init__(self, column_deletions):
        PipeStep.__init__(self)
        if isinstance(column_deletions, str):
            column_deletions = [column_deletions]
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions

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


class OrderRows(PipeStep):
    """Class to specify a columns to determine row order.
    """

    order_columns: List[str]
    reverse: List[str]

    def __init__(self, columns, *, reverse=None, limit=None):
        PipeStep.__init__(self)
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


class RenameColumns(PipeStep):
    """Class to rename columns.
    """

    column_remapping: Dict[str, str]

    def __init__(self, column_remapping):
        PipeStep.__init__(self)
        self.column_remapping = column_remapping.copy()

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


class NaturalJoin(PipeStep):
    _by: List[str]
    _jointype: str
    _b: OperatorPlatform

    def __init__(self, *, b=None, by=None, jointype="INNER"):
        PipeStep.__init__(self)
        if not isinstance(b, ViewRepresentation):
            raise TypeError("b must be a data_algebra.data_ops.ViewRepresentation")
        if isinstance(by, str):
            by = [by]
        self._by = by
        self._jointype = data_algebra.expr_rep.standardize_join_type(jointype)
        self._b = b

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


class ConvertRecords(PipeStep):
    def __init__(self, record_map, *, blocks_out_table=None):
        PipeStep.__init__(self)
        self.record_map = record_map
        self.blocks_out_table = blocks_out_table

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
