from abc import ABC
from typing import Set, Dict, List, Union
import numbers
import re
import collections

import data_algebra
import data_algebra.expr_parse
import data_algebra.flow_text
import data_algebra.data_model
import data_algebra.db_model
import data_algebra.pandas_model
import data_algebra.expr_rep
from data_algebra.data_ops_types import *
import data_algebra.data_ops_utils
import data_algebra.near_sql
import data_algebra.OrderedSet


_have_black = False
try:
    # noinspection PyUnresolvedReferences
    import black

    _have_black = True
except ImportError:
    pass


# noinspection PyBroadException
def pretty_format_python(python_str, *, black_mode=None):
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


class ViewRepresentation(OperatorPlatform, ABC):
    """Structure to represent the columns of a query or a table.
       Abstract base class."""

    column_names: List[str]
    column_set: data_algebra.OrderedSet.OrderedSet
    column_map: collections.OrderedDict
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
        self.column_set = data_algebra.OrderedSet.OrderedSet()
        for c in self.column_names:
            self.column_set.add(c)
        if not len(self.column_names) == len(self.column_set):
            raise ValueError("duplicate column name(s)")
        column_dict = {
            ci: data_algebra.expr_rep.ColumnReference(self, ci)
            for ci in self.column_names
        }
        self.column_map = collections.OrderedDict(**column_dict)
        if sources is None:
            sources = []
        for si in sources:
            if not isinstance(si, ViewRepresentation):
                raise ValueError("all sources must be of class ViewRepresentation")
        self.sources = [si for si in sources]
        OperatorPlatform.__init__(self, node_name=node_name)

    def merged_rep_id(self):
        return "node+ " + str(id(self))

    # characterization

    def get_tables(self):
        """Get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent."""
        tables = {}
        for i in range(len(self.sources)):
            s = self.sources[i]
            ti = s.get_tables()
            for (k, v) in ti.items():
                if not isinstance(v, TableDescription):
                    raise TypeError(
                        "Expected v to be data_algebra.data_ops.TableDescription"
                    )
                if k in tables.keys():
                    if not v.same_table(tables[k]):
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

    def columns_produced(self):
        return self.column_names.copy()

    def _columns_used_implementation(self, *, using, columns_currenty_using_records):
        self_merged_rep_id = self.merged_rep_id()
        try:
            crec = columns_currenty_using_records[self_merged_rep_id]
        except KeyError:
            crec = set()
            columns_currenty_using_records[self_merged_rep_id] = crec
        if using is None:
            crec.update(self.column_names)
        else:
            unknown = set(using) - set(self.column_names)
            if len(unknown) > 0:
                raise ValueError("asked for unknown columns: " + str(unknown))
            crec.update(using)
        cu_list = self.columns_used_from_sources(crec.copy())
        for i in range(len(self.sources)):
            self.sources[i]._columns_used_implementation(
                using=cu_list[i],
                columns_currenty_using_records=columns_currenty_using_records,
            )

    def columns_used(self, *, using=None):
        """Determine which columns are used from source tables."""

        tables = self.get_tables()
        columns_currenty_using_records = {
            v.merged_rep_id(): set() for v in tables.values()
        }
        self._columns_used_implementation(
            using=using, columns_currenty_using_records=columns_currenty_using_records
        )
        columns_used = dict()
        for k in tables.keys():
            ti = tables[k]
            vi = columns_currenty_using_records[ti.merged_rep_id()]
            columns_used[k] = vi.copy()
        return columns_used

    def forbidden_columns(self, *, forbidden=None):
        """Determine which columns should not be in source tables"""
        if forbidden is None:
            forbidden = set()
        res = dict()
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
            python_str = pretty_format_python(python_str, black_mode=black_mode)
        return python_str

    def __repr__(self):
        return self.to_python(strict=True)

    def __str__(self):
        return self.to_python(strict=True)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _equiv_nodes(self, other):
        """Check if immediate node structure is equivalent, does not check child nodes"""
        raise NotImplementedError("base method called")

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

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        raise NotImplementedError("base method called")

    def to_sql(
        self,
        db_model,
        *,
        pretty=False,
        annotate=False,
        encoding=None,
        sqlparse_options=None,
        temp_tables=None,
        use_with=False
    ):
        if isinstance(db_model, data_algebra.db_model.DBHandle):
            db_model = db_model.db_model
        assert isinstance(db_model, data_algebra.db_model.DBModel)
        return db_model.to_sql(
            ops=self,
            pretty=pretty,
            annotate=annotate,
            encoding=encoding,
            sqlparse_options=sqlparse_options,
            temp_tables=temp_tables,
            use_with=use_with
        )

    # Pandas realization

    def eval_implementation(self, *, data_map, data_model, narrow):
        raise NotImplementedError("base method called")

    def check_constraints(self, data_model, *, strict=True):
        """
        Check tables supplied meet data consistency constraints.

        data_model: dictionairy of column name lists.
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

    def eval(self, data_map, *, data_model=None, narrow=True):
        """
         Evaluate operators with respect to Pandas data frames.
         :param data_map: map from table names to data frames
         :param data_model: adaptor to data dialect (Pandas for now)
         :param narrow logical, if True don't copy unexpected columns
         :return:
         """

        if not isinstance(data_map, dict):
            raise TypeError("data_map should be a dictionary")
        if len(data_map) < 1:
            raise ValueError("Expected data_map to be non-empty")
        if data_model is None:
            data_model = data_algebra.default_data_model
        if not isinstance(data_model, data_algebra.data_model.DataModel):
            raise TypeError(
                "Expected data_model to be derived from data_algebra.data_model.DataModel"
            )
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
        return self.eval_implementation(
            data_map=data_map, data_model=data_model, narrow=narrow
        )

    # noinspection PyPep8Naming
    def transform(self, X, *, data_model=None, narrow=True):
        if data_model is None:
            data_model = data_algebra.default_data_model
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
            return self.eval(
                data_map=data_map,
                data_model=data_model,
                narrow=narrow,
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
        parsed_ops = data_algebra.expr_parse.parse_assignments_in_context(
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
        parsed_ops = data_algebra.expr_parse.parse_assignments_in_context(
            ops, self, parse_env=parse_env
        )
        return self.project_parsed(parsed_ops=parsed_ops, group_by=group_by)

    def natural_join(self, b, *, by, jointype):
        """

        :param b: right table
        :param by: list of keys to join by
        :param jointype: one of 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'FULL', 'CROSS' (case insensitive)
        :return: ops describing join
        """
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
        ops = data_algebra.expr_parse.parse_assignments_in_context(
            {"expr": expr}, self, parse_env=parse_env
        )

        def r_walk_expr(opv):
            if not isinstance(opv, data_algebra.expr_rep.Expression):
                return
            for oi in opv.args:
                r_walk_expr(oi)

        for op in ops.values():
            r_walk_expr(op)
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

    def convert_records(self, record_map, *, temp_namer=None):
        if record_map is None:
            return self
        if self.is_trivial_when_intermediate():
            return self.sources[0].convert_records(record_map, temp_namer=temp_namer)
        return ConvertRecordsNode(source=self, record_map=record_map, temp_namer=temp_namer)


# Could also have general query as starting node, but don't see a lot of point to
# it until somebody needs it.


class TableDescription(ViewRepresentation):
    """Describe columns, and qualifiers, of a table.

       Example:
           from data_algebra.data_ops import *
           d = TableDescription('d', ['x', 'y'])
           print(d)
    """

    table_name: str
    column_names: List[str]
    qualifiers: Dict[str, str]
    key: str

    def __init__(
        self,
        table_name,
        column_names,
        *,
        qualifiers=None,
        sql_meta=None,
        column_types=None,
        head=None,
        limit_was=None
    ):
        ViewRepresentation.__init__(
            self, column_names=column_names, node_name="TableDescription"
        )
        if table_name is None:
            table_name = ""
        if (table_name is not None) and (not isinstance(table_name, str)):
            raise TypeError("table_name must be a string")
        if head is not None:
            if set([c for c in head.columns]) != set(column_names):
                raise ValueError("head.columns != column_names")
        self.head = head
        self.limit_was = limit_was
        self.sql_meta = sql_meta
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
        self.key = ""
        if self.table_name is not None:
            self.key = self.table_name

    def same_table(self, other):
        if not isinstance(other, data_algebra.data_ops.TableDescription):
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

    def merged_rep_id(self):
        return "table_" + str(self.key)

    def forbidden_columns(self, *, forbidden=None):
        if forbidden is None:
            forbidden = set()
        return {self.key: set(forbidden)}

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

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        spacer = ' '
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

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.table_step(
            op=self, data_map=data_map, narrow=narrow
        )

    def columns_used_from_sources(self, using=None):
        return []  # no inputs to table description

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.table_def_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def __str__(self):
        rep = ViewRepresentation.__str__(self)
        if self.head is not None:
            rep = rep + "\n#\t" + str(self.head).replace("\n", "\n#\t")
        return rep

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


def describe_table(d, table_name="data_frame",
                   *,
                   qualifiers=None,
                   sql_meta=None,
                   column_types=None,
                   row_limit=7):
    # Expect a pandas.DataFrame style object
    assert d is not None
    assert not isinstance(d, OperatorPlatform)
    assert not isinstance(d, ViewRepresentation)
    column_names = [c for c in d.columns]
    if column_types is None:
        if d.shape[0] > 0:
            column_types = {k: type(d.loc[0, k]) for k in column_names}
    if d.shape[0] > row_limit:
        head = d.iloc[range(row_limit), :].copy()
    else:
        head = d.copy()
    head.reset_index(drop=True, inplace=True)
    return TableDescription(
        table_name,
        column_names,
        column_types=column_types,
        qualifiers=qualifiers,
        sql_meta=sql_meta,
        head=head,
        limit_was=row_limit,
    )


class ExtendNode(ViewRepresentation):
    def __init__(
        self, *, source, parsed_ops, partition_by=None, order_by=None, reverse=None
    ):
        windowed_situation = data_algebra.expr_rep.implies_windowed(parsed_ops)
        ordered_windowed_situation = False
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
            ordered_windowed_situation = True
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
                        "non-aggregated expression in windowed/partitioned extend: "
                        + "'"
                        + k
                        + "': '"
                        + str(opk)
                        + "'"
                    )
                if len(opk.args) > 1:
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
                        if value_name not in source.column_set:
                            raise ValueError(
                                value_name + " not in source column set"
                            )
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
                if (ordered_windowed_situation and
                        (opk.op in data_algebra.expr_rep.fn_names_that_contradict_ordered_windowed_situation)):
                    raise ValueError(
                        str(opk)
                        + "' is not allowed in an ordered windowed situation"
                    )
                if ((not ordered_windowed_situation) and
                        (opk.op in data_algebra.expr_rep.fn_names_that_imply_ordered_windowed_situation)):
                    raise ValueError(
                        str(opk)
                        + "' is not allowed in not-ordered windowed situation"
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

    def check_extend_window_fns(self):
        window_situation = (len(self.partition_by) > 0) or (len(self.order_by) > 0)
        if window_situation:
            # check these are forms we are prepared to work with
            for (k, opk) in self.ops.items():
                if len(opk.args) > 0:
                    if len(opk.args) > 1:
                        raise ValueError("window function with more than one argument")
                    for i in range(len(opk.args)):
                        if not (isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference)
                                or isinstance(opk.args[0], data_algebra.expr_rep.Value)):
                            raise ValueError(
                                "window expression argument must be a column or value: "
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

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        spacer = ' '
        if indent >= 0:
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

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.extend_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.extend_step(
            op=self, data_map=data_map, narrow=narrow
        )


class ProjectNode(ViewRepresentation):
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
            if isinstance(opk, data_algebra.expr_rep.Expression):
                if len(opk.args) > 1:
                    raise ValueError(
                        "non-trivial aggregation expression: "
                        + str(k)
                        + ": "
                        + str(opk)
                    )
                if len(opk.args) > 0:
                    if not (isinstance(opk.args[0], data_algebra.expr_rep.ColumnReference)
                            or isinstance(opk.args[0], data_algebra.expr_rep.Value)):
                        raise ValueError(
                            "windows expression argument must be a column or value: "
                            + str(k)
                            + ": "
                            + str(opk)
                        )
                if opk.op in data_algebra.expr_rep.fn_names_that_imply_ordered_windowed_situation:
                    raise ValueError(
                        str(opk)
                        + "' is not allowed in project"
                    )
            else:
                raise ValueError(
                    "non-aggregated expression in project: "
                    + str(k)
                    + ": "
                    + str(opk)
                )
            # TODO: check op is in list of aggregators
            # Note: non-aggregators making through will be caught by table shape check

    def forbidden_columns(self, *, forbidden=None):
        if forbidden is None:
            forbidden = set()
        forbidden = set(forbidden).intersection(self.column_names)
        return self.sources[0].forbidden_columns(forbidden=forbidden)

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].project_parsed(
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

    def columns_used_from_sources(self, using=None):
        if using is None:
            subops = self.ops
        else:
            subops = {k: op for (k, op) in self.ops.items() if k in using}
        columns_we_take = set(self.group_by)
        for (k, o) in subops.items():
            o.get_column_names(columns_we_take)
        return [columns_we_take]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        spacer = ' '
        if indent >= 0:
            spacer = "\n   " + " " * indent
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
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

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.project_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.project_step(
            op=self, data_map=data_map, narrow=narrow
        )


class SelectRowsNode(ViewRepresentation):
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

    def apply_to(self, a, *, target_table_key=None):
        new_sources = [
            s.apply_to(a, target_table_key=target_table_key) for s in self.sources
        ]
        return new_sources[0].select_rows_parsed(parsed_ops=self.ops)

    def _equiv_nodes(self, other):
        if not isinstance(other, SelectRowsNode):
            return False
        if not self.expr.is_equal(other.expr):
            return False
        if len(self.ops) != len(other.ops):
            return False
        return True

    def columns_used_from_sources(self, using=None):
        columns_we_take = self.sources[0].column_set.copy()
        if using is None:
            return [columns_we_take]
        columns_we_take = columns_we_take.intersection(using)
        columns_we_take = columns_we_take.union(self.decision_columns)
        return [columns_we_take]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("select_rows(" + self.expr.to_python().__repr__() + ")")
        return s

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.select_rows_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.select_rows_step(
            op=self, data_map=data_map, narrow=narrow
        )


class SelectColumnsNode(ViewRepresentation):
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

    def forbidden_columns(self, *, forbidden=None):
        if forbidden is None:
            forbidden = set()
        forbidden = set(forbidden).intersection(self.column_selection)
        return self.sources[0].forbidden_columns(forbidden=forbidden)

    def apply_to(self, a, *, target_table_key=None):
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

    def columns_used_from_sources(self, using=None):
        cols = set(self.column_selection.copy())
        if using is None:
            return [cols]
        return [cols.intersection(using)]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("select_columns(" + self.column_selection.__repr__() + ")")
        return s

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.select_columns_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.select_columns_step(
            op=self, data_map=data_map, narrow=narrow
        )


class DropColumnsNode(ViewRepresentation):
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

    def forbidden_columns(self, *, forbidden=None):
        if forbidden is None:
            forbidden = set()
        forbidden = set(forbidden) - set(self.column_deletions)
        return self.sources[0].forbidden_columns(forbidden=forbidden)

    def apply_to(self, a, *, target_table_key=None):
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

    def columns_used_from_sources(self, using=None):
        if using is None:
            using = set(self.sources[0].column_names)
        return [set([c for c in using if c not in self.column_deletions])]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("drop_columns(" + self.column_deletions.__repr__() + ")")
        return s

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.drop_columns_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.drop_columns_step(
            op=self, data_map=data_map, narrow=narrow
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
        if not isinstance(other, OrderRowsNode):
            return False
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

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("order_rows(" + self.order_columns.__repr__())
        if len(self.reverse) > 0:
            s = s + ", reverse=" + self.reverse.__repr__()
        if self.limit is not None:
            s = s + ", limit=" + self.limit.__repr__()
        s = s + ")"
        return s

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.order_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.order_rows_step(
            op=self, data_map=data_map, narrow=narrow
        )

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
        self.new_columns = set(new_cols) - set(orig_cols)
        ViewRepresentation.__init__(
            self,
            column_names=column_names,
            sources=[source],
            node_name="RenameColumnsNode",
        )

    def forbidden_columns(self, *, forbidden=None):
        # this is where forbidden columns are introduced
        if forbidden is None:
            forbidden = set()
        new_forbidden = set(forbidden) - self.reverse_mapping.keys()
        new_forbidden.update(self.new_columns)
        return self.sources[0].forbidden_columns(forbidden=new_forbidden)

    def apply_to(self, a, *, target_table_key=None):
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

    def columns_used_from_sources(self, using=None):
        if using is None:
            using = self.column_names
        cols = [
            (k if k not in self.column_remapping.keys() else self.column_remapping[k])
            for k in using
        ]
        return [set(cols)]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("rename_columns(" + self.column_remapping.__repr__() + ")")
        return s

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.rename_to_sql(self, using=using, temp_id_source=temp_id_source)

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.rename_columns_step(
            op=self, data_map=data_map, narrow=narrow
        )


class NaturalJoinNode(ViewRepresentation):
    by: List[str]
    jointype: str

    def __init__(self, a, b, *, by, jointype):
        # check set of tables is consistent in both sub-dags
        a_tables = a.get_tables()
        b_tables = b.get_tables()
        if by is None:
            raise ValueError("Must specify by in natural joins ([] for empty conditions)")
        common_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_keys:
            if not a_tables[k].same_table(b_tables[k]):
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
        if (self.jointype == 'CROSS') and (len(self.by) != 0):
            raise ValueError("CROSS joins must have an empty 'by' list")
        self.get_tables()  # causes a throw if left and right table descriptions are inconsistent

    def apply_to(self, a, *, target_table_key=None):
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

    def columns_used_from_sources(self, using=None):
        if using is None:
            return [self.sources[i].column_set.copy() for i in range(2)]
        using = using.union(self.by)
        return [self.sources[i].column_set.intersection(using) for i in range(2)]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = "_0."
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("natural_join(b=\n" + " " * (indent + 6))
        if print_sources:
            s = s + (
                self.sources[1].to_python_implementation(
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

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.natural_join_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.natural_join_step(
            op=self, data_map=data_map, narrow=narrow
        )


class ConcatRowsNode(ViewRepresentation):
    id_column: Union[str, None]

    def __init__(self, a, b, *, id_column="table_name", a_name="a", b_name="b"):
        # check set of tables is consistent in both sub-dags
        a_tables = a.get_tables()
        b_tables = b.get_tables()
        common_keys = set(a_tables.keys()).intersection(b_tables.keys())
        for k in common_keys:
            if not a_tables[k].same_table(b_tables[k]):
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
        if not isinstance(other, ConcatRowsNode):
            return False
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

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = "_0."
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        s = s + ("concat_rows(b=\n" + " " * (indent + 6))
        if print_sources:
            s = s + (
                self.sources[1].to_python_implementation(
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

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return db_model.concat_rows_to_sql(
            self, using=using, temp_id_source=temp_id_source
        )

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.concat_rows_step(
            op=self, data_map=data_map, narrow=narrow
        )


class ConvertRecordsNode(ViewRepresentation):
    def __init__(self, *, source, record_map, temp_namer=None):
        sources = [source]
        self.record_map = record_map
        self.temp_namer = temp_namer
        unknown = set(self.record_map.columns_needed) - set(source.column_names)
        if len(unknown) > 0:
            raise ValueError("missing required columns: " + str(unknown))
        ViewRepresentation.__init__(
            self,
            column_names=record_map.columns_produced,
            sources=sources,
            node_name="ConvertRecordsNode",
        )

    def control_out_table(self, *, temp_id_source):
        if self.temp_namer is None:
            view_name = "cdata_temp_record_" + str(temp_id_source[0])
        else:
            view_name = self.temp_namer(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        res = TableDescription(
            table_name=view_name,
            column_names=self.record_map.blocks_out.control_table.columns
        )
        return res

    def apply_to(self, a, *, target_table_key=None):
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

    def columns_used_from_sources(self, using=None):
        return [self.record_map.columns_needed]

    def to_python_implementation(self, *, indent=0, strict=True, print_sources=True):
        s = ""
        if print_sources:
            s = (
                self.sources[0].to_python_implementation(indent=indent, strict=strict)
                + " .\\\n"
                + " " * (max(indent, 0) + 3)
            )
        rm_str = self.record_map.__repr__()
        rm_str = re.sub("\n", "\n   ", rm_str)
        s = s + "convert_records(" + rm_str
        s = s + ")"
        return s

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        if temp_id_source is None:
            temp_id_source = [0]
        # TODO: narrow to what we are using
        sub_query = self.sources[0].to_near_sql_implementation(
            db_model=db_model, using=None, temp_id_source=temp_id_source
        )
        # claims to use all columns
        query = sub_query.to_sql(
            columns=self.columns_used_from_sources()[0], db_model=db_model, force_sql=True
        )
        control_out_table = None
        if self.record_map.blocks_in is not None:
            query = db_model.blocks_to_row_recs_query(
                query, record_spec=self.record_map.blocks_in
            )
        if self.record_map.blocks_out is not None:
            control_out_table = self.control_out_table(temp_id_source=temp_id_source)
            query = db_model.row_recs_to_blocks_query(
                query,
                record_spec=self.record_map.blocks_out,
                control_view=control_out_table,
            )
        view_name = "convert_records_in_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        prev_view_name = "convert_records_out_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = {k: None for k in self.record_map.columns_produced}
        temp_tables = sub_query.temp_tables.copy()
        if control_out_table is not None:
            if control_out_table.key in temp_tables.keys():
                raise ValueError(
                    "key collision in temp_tables construction: " + control_out_table.key
                )
            temp_tables[control_out_table.key] = self.record_map.blocks_out.control_table
        near_sql = data_algebra.near_sql.NearSQLq(
            quoted_query_name=db_model.quote_identifier(view_name),
            prev_quoted_query_name=db_model.quote_identifier(prev_view_name),
            query=query,
            terms=terms,
            temp_tables=temp_tables,
            annotation='convert records'
        )
        return near_sql

    def eval_implementation(self, *, data_map, data_model, narrow):
        if data_model is None:
            raise ValueError("Expected data_model to not be None")
        return data_model.convert_records_step(
            op=self, data_map=data_map, narrow=narrow
        )
