from typing import Set, Any, Dict, List
import collections

import data_algebra.expr_rep
import data_algebra.pipe
import data_algebra.env
import data_algebra.pending_eval

have_black = False
try:
    # noinspection PyUnresolvedReferences
    import black
    have_black = True
except ImportError:
    pass

have_sqlparse = False
try:
    # noinspection PyUnresolvedReferences
    import sqlparse
    have_sqlparse = True
except ImportError:
    pass


class ViewRepresentation(data_algebra.pipe.PipeValue):
    """Structure to represent the columns of a query or a table.
       Abstract base class."""

    column_names: List[str]
    column_set: Set[str]
    column_map: data_algebra.env.SimpleNamespaceDict
    sources: List[Any]  # actually ViewRepresentation

    def __init__(self, column_names, *, sources=None):
        self.column_names = [c for c in column_names]
        for ci in self.column_names:
            if not isinstance(ci, str):
                raise Exception("non-string column name(s)")
        if len(self.column_names) < 1:
            raise Exception("no column names")
        self.column_set = set(self.column_names)
        if not len(self.column_names) == len(self.column_set):
            raise Exception("duplicate column name(s)")
        column_dict = {
            ci: data_algebra.expr_rep.ColumnReference(self, ci)
            for ci in self.column_names
        }
        self.column_map = data_algebra.env.SimpleNamespaceDict(**column_dict)
        if sources is None:
            sources = []
        for si in sources:
            if not isinstance(si, ViewRepresentation):
                raise Exception("all sources must be of class ViewRepresentation")
        self.sources = [si for si in sources]
        data_algebra.pipe.PipeValue.__init__(self)

    # characterization

    def get_tables(self, tables=None):
        """get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent"""
        if tables is None:
            tables = {}
        for s in self.sources:
            tables = s.get_tables(tables)
        return tables

    def columns_used_from_sources(self, using=None):
        """Give column names used from source nodes when this node is exececuted
        with the using columns (None means all)."""
        raise Exception("base method called")

    def columns_used_implementation(self, *, columns_used, using=None):
        cu_list = self.columns_used_from_sources(using)
        for i in range(len(self.sources)):
            self.sources[i].columns_used_implementation(
                columns_used=columns_used, using=cu_list[i]
            )

    def columns_used(self):
        """Determine which columns are used from source tables."""
        tables = self.get_tables()
        columns_used = {ki: set() for ki in tables.keys()}
        self.columns_used_implementation(columns_used=columns_used, using=None)
        return columns_used

    # collect as simple structures for YAML I/O and other generic tasks

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        raise Exception("base method called")

    def collect_representation(self, *, pipeline=None, dialect='Python'):
        """Collect a representation of the operator DAG as simple serializable objects.
                   These objects can be saved/loaded in YAML format and also can rebuild the
                   pipeline via data_algebra.yaml.to_pipeline()."""
        self.get_tables()  # for table consistency check/raise
        return self.collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    # printing

    def to_python_implementation(self, *, indent=0, strict=True):
        return "ViewRepresentation(" + self.column_names.__repr__() + ")"

    def to_python(self, *, indent=0, strict=True, pretty=False, black_mode=None):
        global have_black
        self.get_tables()  # for table consistency check/raise
        if pretty:
            strict = True
        python_str = self.to_python_implementation(indent=indent, strict=strict)
        if pretty and have_black:
            if black_mode is None:
                black_mode = black.FileMode()
            python_str = black.format_str(python_str, mode=black_mode)
        return python_str

    def __repr__(self):
        return self.to_python(strict=True)

    def __str__(self):
        return self.to_python(strict=True)

    # query generation

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        raise Exception("base method called")

    def to_sql(self, db_model,
               *,
               pretty=False,
               encoding=None,
               sqlparse_options=None):
        global have_sqlparse
        if sqlparse_options is None:
            sqlparse_options = {'reindent': True,
                                'keyword_case': 'upper'}
        self.get_tables()  # for table consistency check/raise
        temp_id_source = [0]
        sql_str = self.to_sql_implementation(
            db_model=db_model, using=None, temp_id_source=temp_id_source
        )
        if pretty and have_sqlparse:
            sql_str = sqlparse.format(sql_str, encoding=encoding, **sqlparse_options)
        return sql_str

    # Pandas realization

    def eval_pandas(self, data_map):
        """
        Evaluate pipeline taking tables by name from data_map
        :param data_map: Dict[str, pandas.DataFrame]
        :return: pandas.DataFrame
        """
        raise Exception("base method called")

    # define builders for all non-leaf node types on base class

    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        return ExtendNode(
            source=self,
            ops=ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse,
        )

    def project(self, ops, *, group_by=None, order_by=None, reverse=None):
        raise Exception("not implmented yet")
        # return ProjectNode(
        #     source=self,
        #     ops=ops,
        #     group_by=group_by,
        #     order_by=order_by,
        #     reverse=reverse,
        # )

    def natural_join(self, b, *, by=None, jointype="INNER"):
        if not isinstance(b, ViewRepresentation):
            raise Exception(
                "expected b to be a data_algebra.dat_ops.ViewRepresentation"
            )
        return NaturalJoinNode(a=self, b=b, by=by, jointype=jointype)

    def select_rows(self, expr):
        return SelectRowsNode(source=self, expr=expr)

    def drop_columns(self, column_deletions):
        return DropColumnsNode(source=self, column_deletions=column_deletions)

    def select_columns(self, columns):
        return SelectColumnsNode(source=self, columns=columns)

    def rename_columns(self, column_remapping):
        return RenameColumnsNode(source=self, column_remapping=column_remapping)

    def order_rows(self, columns, *, reverse=None, limit=None):
        return OrderRowsNode(source=self, columns=columns, reverse=reverse, limit=limit)


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
            raise Exception("table_name must be a string")
        self.table_name = table_name
        self.column_names = column_names.copy()
        if qualifiers is None:
            qualifiers = {}
        if not isinstance(qualifiers, dict):
            raise Exception("qualifiers must be a dictionary")
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

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
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

    def to_python_implementation(self, *, indent=0, strict=True):
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

    def get_tables(self, tables=None):
        """get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent"""
        if tables is None:
            tables = {}
        if self.key in tables.keys():
            other = tables[self.key]
            if self.column_set != other.column_set:
                raise Exception(
                    "Two tables with key " + self.key + " have different column sets."
                )
        else:
            tables[self.key] = self
        return tables

    def eval_pandas(self, data_map):
        if len(self.qualifiers) > 0:
            raise Exception(
                "table descriptions used with eval_pandas() must not have qualifiers"
            )
        # make an index-free copy of the data to isolate side-effects and not deal with indices
        res = data_map[self.table_name]
        res = res.copy()
        res.reset_index(drop=True, inplace=True)
        return res

    def columns_used_from_sources(self, using=None):
        return []  # no inputs to table description

    def columns_used_implementation(self, *, columns_used, using=None):
        cset = columns_used[self.key]
        if using is None:
            using = self.column_set
        unexpected = using - self.column_set
        if len(unexpected) > 0:
            raise Exception("asked for undefined columns: " + str(unexpected))
        cset.update(using)

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.table_def_to_sql(self, using=using)

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


def describe_pandas_table(d, table_name):
    return TableDescription(table_name, [c for c in d.columns])


class ExtendNode(ViewRepresentation):
    ops: Dict[str, data_algebra.expr_rep.Expression]

    def __init__(self, source, ops, *, partition_by=None, order_by=None, reverse=None):
        ops = data_algebra.expr_rep.check_convert_op_dictionary(
            ops, source.column_map.__dict__
        )
        if len(ops) < 1:
            raise Exception("no ops")
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
            raise Exception("referred to unknown columns: " + str(unknown_cols))
        known_cols = set(column_names)
        for ci in ops.keys():
            if ci not in known_cols:
                column_names.append(ci)
        if len(partition_by) != len(set(partition_by)):
            raise Exception("Duplicate name in partition_by")
        if len(order_by) != len(set(order_by)):
            raise Exception("Duplicate name in order_by")
        if len(reverse) != len(set(reverse)):
            raise Exception("Duplicate name in reverse")
        unknown = set(partition_by) - known_cols
        if len(unknown) > 0:
            raise Exception("unknown partition_by columns: " + str(unknown))
        unknown = set(order_by) - known_cols
        if len(unknown) > 0:
            raise Exception("unknown order_by columns: " + str(unknown))
        unknown = set(reverse) - set(order_by)
        if len(unknown) > 0:
            raise Exception("reverse columns not in order_by: " + str(unknown))
        bad_overwrite = set(ops.keys()).intersection(
            set(partition_by).union(order_by, reverse)
        )
        if len(bad_overwrite) > 0:
            raise Exception("tried to change: " + str(bad_overwrite))
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

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Extend"
        od["ops"] = {ci: vi.to_source(dialect=dialect) for (ci, vi) in self.ops.items()}
        od["partition_by"] = self.partition_by
        od["order_by"] = self.order_by
        od["reverse"] = self.reverse
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        s = (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "extend({"
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

    def eval_pandas(self, data_map):
        window_situation = (len(self.partition_by) > 0) or (len(self.order_by) > 0)
        if window_situation:
            # check these are forms we are prepared to work with
            for (k, op) in self.ops.items():
                if len(op.args) > 1:
                    raise Exception(
                        "non-trivial windows expression: " + str(k) + ": " + str(op)
                    )
                if len(op.args) == 1:
                    if not isinstance(
                        op.args[0], data_algebra.expr_rep.ColumnReference
                    ):
                        raise Exception(
                            "windows expression argument must be a column: "
                            + str(k)
                            + ": "
                            + str(op)
                        )
        res = self.sources[0].eval_pandas(data_map)
        res.reset_index(inplace=True, drop=True)
        if not window_situation:
            for (k, op) in self.ops.items():
                res[k] = res.eval(op.to_pandas())
        else:
            for (k, op) in self.ops.items():
                # work on a slice of the data frame
                col_list = [c for c in set(self.partition_by)]
                for c in self.order_by:
                    if c not in col_list:
                        col_list = col_list + [c]
                value_name = None
                if len(op.args) > 0:
                    value_name = op.args[0].to_pandas()
                    if value_name not in set(col_list):
                        col_list = col_list + [value_name]
                ascending = [c not in set(self.reverse) for c in col_list]
                subframe = res[col_list].copy()
                subframe.reset_index(inplace=True, drop=True)
                subframe['_data_algebra_orig_index'] = [i for i in range(subframe.shape[0])]
                subframe.sort_values(by=col_list, ascending=ascending, inplace=True)
                subframe.reset_index(inplace=True, drop=True)
                if len(self.partition_by) > 0:
                    opframe = subframe.groupby(self.partition_by)
                    #  Groupby preserves the order of rows within each group.
                    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
                else:
                    opframe = subframe
                if len(op.args) == 0:
                    if op.op == "row_number":
                        subframe[k] = opframe.cumcount() + 1
                    else:  # TODO: more of these
                        raise Exception("not implemented: " + str(k) + ": " + str(op))
                else:
                    # len(op.args) == 1
                    subframe[k] = opframe[value_name].transform(op.op)
                subframe.reset_index(inplace=True, drop=True)
                subframe.sort_values(by=['_data_algebra_orig_index'], inplace=True)
                subframe.reset_index(inplace=True, drop=True)
                res[k] = subframe[k]
        return res


class SelectRowsNode(ViewRepresentation):
    expr: data_algebra.expr_rep.Expression
    decision_columns: Set[str]

    def __init__(self, source, expr):
        ops = data_algebra.expr_rep.check_convert_op_dictionary(
            {"expr": expr}, source.column_map.__dict__
        )
        if len(ops) < 1:
            raise Exception("no ops")
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

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "SelectRows"
        od["expr"] = self.expr.to_source(dialect=dialect)
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        s = (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "select_rows("
            + self.expr.to_python().__repr__()
            + ")"
        )
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.select_rows_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_pandas(self, data_map):
        res = self.sources[0].eval_pandas(data_map)
        res = res.query(self.expr.to_pandas())
        res.reset_index(inplace=True, drop=True)
        return res


class SelectColumnsNode(ViewRepresentation):
    column_selection: List[str]

    def __init__(self, source, columns):
        column_selection = [c for c in columns]
        self.column_selection = column_selection
        # TODO: check column conditions
        ViewRepresentation.__init__(
            self, column_names=column_selection, sources=[source]
        )

    def columns_used_from_sources(self, using=None):
        cols = set(self.column_selection.copy())
        if using is None:
            return [cols]
        return [cols.intersection(using)]

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "SelectColumns"
        od["columns"] = self.column_selection
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        s = (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "select_columns("
            + self.column_selection.__repr__()
            + ")"
        )
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.select_columns_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_pandas(self, data_map):
        res = self.sources[0].eval_pandas(data_map)
        return res[self.column_selection]


class DropColumnsNode(ViewRepresentation):
    column_deletions: List[str]

    def __init__(self, source, column_deletions):
        column_deletions = [c for c in column_deletions]
        self.column_deletions = column_deletions
        remaining_columns = [c for c in source.column_names if c not in column_deletions]
        # TODO: check column conditions
        ViewRepresentation.__init__(
            self, column_names=remaining_columns, sources=[source]
        )

    def columns_used_from_sources(self, using=None):
        if using is None:
            using = set(self.sources[0].column_names)
        return [set([c for c in using if c not in self.column_deletions])]

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "DropColumns"
        od["column_deletions"] = self.column_deletions
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        s = (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "drop_columns("
            + self.column_deletions.__repr__()
            + ")"
        )
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.drop_columns_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_pandas(self, data_map):
        res = self.sources[0].eval_pandas(data_map)
        column_selection = [c for c in res.columns if c not in self.column_deletions]
        return res[column_selection]


class OrderRowsNode(ViewRepresentation):
    order_columns: List[str]
    reverse: List[str]

    def __init__(self, source, columns, *, reverse=None, limit=None):
        self.order_columns = [c for c in columns]
        if reverse is None:
            reverse = []
        self.reverse = [c for c in reverse]
        self.limit = limit
        # TODO: check column conditions
        ViewRepresentation.__init__(
            self, column_names=source.column_names, sources=[source]
        )

    def columns_used_from_sources(self, using=None):
        cols = set(self.column_names.copy())
        if using is None:
            return [cols]
        cols = cols.intersection(using).union(self.order_columns)
        return [cols]

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Order"
        od["order_columns"] = self.order_columns
        od["reverse"] = self.reverse
        od["limit"] = self.limit
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        s = (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "order_rows("
            + self.order_columns.__repr__()
        )
        if len(self.reverse) > 0:
            s = s + ", reverse=" + self.reverse.__repr__()
        if self.limit is not None:
            s = s + ", limit=" + self.limit.__repr__()
        s = s + ")"
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.order_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_pandas(self, data_map):
        res = self.sources[0].eval_pandas(data_map)
        ascending = [False if ci in set(self.reverse) else True for ci in self.order_columns]
        res.sort_values(by=self.order_columns, ascending=ascending)
        return res


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
        column_names = [
            (k if k not in self.reverse_mapping.keys() else self.reverse_mapping[k])
            for k in source.column_names
        ]
        # TODO: check column conditions, don't allow name collisions
        ViewRepresentation.__init__(self, column_names=column_names, sources=[source])

    def columns_used_from_sources(self, using=None):
        if using is None:
            using = self.column_names
        cols = [
            (k if k not in self.column_remapping.keys() else self.column_remapping[k])
            for k in using
        ]
        return [set(cols)]

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "Rename"
        od["column_remapping"] = self.column_remapping
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        s = (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "rename_columns("
            + self.column_remapping.__repr__()
            + ")"
        )
        return s

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.rename_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_pandas(self, data_map):
        res = self.sources[0].eval_pandas(data_map)
        return res.rename(columns=self.reverse_mapping)


class NaturalJoinNode(ViewRepresentation):
    by: List[str]
    jointype: str

    def __init__(self, a, b, *, by=None, jointype="INNER"):
        sources = [a, b]
        column_names = sources[0].column_names.copy()
        for ci in sources[1].column_names:
            if ci not in sources[0].column_set:
                column_names.append(ci)
        if isinstance(by, str):
            by = [by]
        by_set = set(by)
        if len(by) != len(by_set):
            raise Exception("duplicate column names in by")
        missing_left = by_set - a.column_set
        if len(missing_left) > 0:
            raise Exception("left table missing join keys: " + str(missing_left))
        missing_right = by_set - b.column_set
        if len(missing_right) > 0:
            raise Exception("right table missing join keys: " + str(missing_right))
        self.by = by
        self.jointype = jointype
        ViewRepresentation.__init__(self, column_names=column_names, sources=sources)

    def columns_used_from_sources(self, using=None):
        if using is None:
            return [self.sources[i].column_set.copy() for i in range(2)]
        using = using.union(self.by)
        return [self.sources[i].column_set.intersection(using) for i in range(2)]

    def collect_representation_implementation(self, *, pipeline=None, dialect='Python'):
        if pipeline is None:
            pipeline = []
        od = collections.OrderedDict()
        od["op"] = "NaturalJoin"
        od["by"] = self.by
        od["jointype"] = self.jointype
        od["b"] = self.sources[1].collect_representation_implementation(dialect=dialect)
        pipeline.insert(0, od)
        return self.sources[0].collect_representation_implementation(pipeline=pipeline, dialect=dialect)

    def to_python_implementation(self, *, indent=0, strict=True):
        return (
            self.sources[0].to_python_implementation(indent=indent, strict=strict)
            + " .\\\n"
            + " " * (indent + 3)
            + "natural_join(b=\n"
            + " " * (indent + 6)
            + self.sources[1].to_python_implementation(indent=indent + 6, strict=strict)
            + ",\n"
            + " " * (indent + 6)
            + "by="
            + self.by.__repr__()
            + ", jointype="
            + self.jointype.__repr__()
            + ")"
        )

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.natural_join_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )
