
import math
import re
import io
from types import SimpleNamespace
from collections import OrderedDict

import pandas.io.sql

import data_algebra

import data_algebra.near_sql
import data_algebra.expr_rep
import data_algebra.util
import data_algebra.data_ops_types
import data_algebra.eval_model
import data_algebra.data_ops


_have_sqlparse = False
try:
    # noinspection PyUnresolvedReferences
    import sqlparse

    _have_sqlparse = True
except ImportError:
    pass


# noinspection PyBroadException
def pretty_format_sql(sql, *, encoding=None, sqlparse_options=None):
    assert isinstance(sql, str)
    assert isinstance(encoding, (str, type(None)))
    assert isinstance(encoding, (dict, type(None)))
    if sqlparse_options is None:
        sqlparse_options = {"reindent": True, "keyword_case": "upper"}
    formatted_sql = sql
    if _have_sqlparse:
        try:
            formatted_sql = sqlparse.format(
                sql, encoding=encoding, **sqlparse_options
            )
        except Exception:
            pass
    return formatted_sql


def _clean_annotation(annotation):
    assert isinstance(annotation, (str, type(None)))
    if annotation is None:
        return annotation
    annotation = annotation.strip()
    annotation = re.sub(r'\s+', ' ', annotation)
    return annotation


# map from op-name to special SQL formatting code


def _db_mean_expr(dbmodel, expression):
    return (
        "AVG(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


def _db_size_expr(dbmodel, expression):
    return "SUM(1)"


def _db_is_null_expr(dbmodel, expression):
    return (
        "("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " IS NULL)"
    )


def _db_is_bad_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return (
        "("
        + subexpr
        + " IS NULL OR "
        + subexpr
        + " >= "
        + dbmodel.quote_literal("+infinity")
        + " OR "
        + subexpr
        + " <= "
        + dbmodel.quote_literal("-infinity")
        + " OR ("
        + subexpr
        + " != 0 AND "
        + subexpr
        + " = -"
        + subexpr
        + "))"
    )


def _db_if_else_expr(dbmodel, expression):
    if_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    x_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[2], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + if_expr
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + if_expr
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_maximum_expr(dbmodel, expression):
    x_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + "(" + x_expr + ") >= (" + y_expr + ")"
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + "(" + x_expr + ") >= (" + y_expr + ")"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_minimum_expr(dbmodel, expression):
    x_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + "(" + x_expr + ") <= (" + y_expr + ")"
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + "(" + x_expr + ") <= (" + y_expr + ")"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_is_in_expr(dbmodel, expression):
    is_in_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    x_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "("
        + is_in_expr
        + " IN ("
        + ', '.join([repr(v) for v in x_expr])
        + "))"
    )


# noinspection PyUnusedLocal
def _db_count_expr(dbmodel, expression):
    return 'SUM(1)'


def _db_concat_expr(dbmodel, expression):
    return (
        "("  # TODO: cast each to char on way in
        + " || ".join([dbmodel.expr_to_sql(ai, want_inline_parens=True) for ai in expression.args])
        + ")"
    )


def _db_coalesce_expr(dbmodel, expression):
    return (
        "COALESCE("
        + ", ".join([dbmodel.expr_to_sql(ai, want_inline_parens=False) for ai in expression.args])
        + ")"
    )


def _db_round_expr(dbmodel, expression):
    return (
        "ROUND("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _db_floor_expr(dbmodel, expression):
    return (
        "FLOOR("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _db_ceil_expr(dbmodel, expression):
    return (
        "CEILING("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _db_pow_expr(dbmodel, expression):
    return (
        "POWER("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ")"
    )


def _db_nunique_expr(dbmodel, expression):
    return (
        "COUNT(DISTINCT (" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + "))"
    )


# fns that had been in bigquery_user_fns


def _as_int64(dbmodel, expression):
    return (
        'CAST('
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ' AS INT64)'
    )


def _as_str(dbmodel, expression):
    return (
            'CAST('
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ' AS ' + dbmodel.string_type + ')'
    )


def _trimstr(dbmodel, expression):
    return (
            'SUBSTR('
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ', 1 + ' + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[2], want_inline_parens=False)
            + ')'
    )


def _datetime_to_date(dbmodel, expression):
    return (
            'DATE('
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _parse_datetime(dbmodel, expression):
    return (
            'PARSE_DATETIME('
            + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _parse_date(dbmodel, expression):
    return (
            'PARSE_DATE('
            + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _format_datetime(dbmodel, expression):
    return (
            'FORMAT_DATETIME('
            + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _format_date(dbmodel, expression):
    return (
            'FORMAT_DATE('
            + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _dayofweek(dbmodel, expression):
    return (
            'EXTRACT(DAYOFWEEK FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _dayofyear(dbmodel, expression):
    return (
            'EXTRACT(DAYOFYEAR FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _dayofmonth(dbmodel, expression):
    return (
            'EXTRACT(DAY FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _weekofyear(dbmodel, expression):
    return (
            'EXTRACT(WEEK FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _month(dbmodel, expression):
    return (
            'EXTRACT(MONTH FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _quarter(dbmodel, expression):
    return (
            'EXTRACT(QUARTER FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _year(dbmodel, expression):
    return (
            'EXTRACT(YEAR FROM '
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ')'
    )


def _timestamp_diff(dbmodel, expression):
    return (
            'TIMESTAMP_DIFF('
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', SECOND)'
    )


def _date_diff(dbmodel, expression):
    return (
            'TIMESTAMP_DIFF('
            + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
            + ', ' + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
            + ', DAY)'
    )


def _base_Sunday(dbmodel, expression):
    subex = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    return f'DATE_SUB({subex}, INTERVAL (EXTRACT(DAYOFWEEK FROM {subex}) - 1) DAY)'


db_expr_formatters = {
    "is_null": _db_is_null_expr,
    "is_bad": _db_is_bad_expr,
    "mean": _db_mean_expr,
    "size": _db_size_expr,
    "if_else": _db_if_else_expr,
    "is_in": _db_is_in_expr,
    "maximum": _db_maximum_expr,
    "minimum": _db_minimum_expr,
    'count': _db_count_expr,
    'concat': _db_concat_expr,
    'coalesce': _db_coalesce_expr,
    'round': _db_round_expr,
    'floor': _db_floor_expr,
    'ceil': _db_ceil_expr,
    '**': _db_pow_expr,
    "nunique": _db_nunique_expr,
    # fns that had been in bigquery_user_fns
    'as_int64': _as_int64,
    'as_str': _as_str,
    'trimstr': _trimstr,
    'datetime_to_date': _datetime_to_date,
    'parse_datetime': _parse_datetime,
    'parse_date': _parse_date,
    'format_datetime': _format_datetime,
    'format_date': _format_date,
    'dayofweek': _dayofweek,
    'dayofyear': _dayofyear,
    'dayofmonth': _dayofmonth,
    'weekofyear': _weekofyear,
    'month': _month,
    'quarter': _quarter,
    'year': _year,
    'timestamp_diff': _timestamp_diff,
    'date_diff': _date_diff,
    'base_Sunday': _base_Sunday,
}


db_default_op_replacements = {
    "==": "=",
    "cumsum": "sum",
    "&": "AND",
    "&&": "AND",
    "|": "OR",
    "||": "OR",
    "!": "NOT",
    "~": "NOT",
    }


class DBModel:
    """A model of how SQL should be generated for a given database.
       """

    identifier_quote: str
    string_quote: str
    on_start: str
    on_end: str
    on_joiner: str
    drop_text: str
    string_type: str
    join_name_map: dict
    supports_with: bool

    def __init__(
        self,
        *,
        identifier_quote='"',
        string_quote="'",
        sql_formatters=None,
        op_replacements=None,
        local_data_model=None,
        on_start='',
        on_end='',
        on_joiner=', ',
        drop_text='DROP TABLE',
        string_type='VARCHAR',
        join_name_map=None,
        supports_with=True,
    ):
        if local_data_model is None:
            local_data_model = data_algebra.default_data_model
        self.local_data_model = local_data_model
        if sql_formatters is None:
            sql_formatters = {}
        assert identifier_quote != string_quote
        self.identifier_quote = identifier_quote
        self.string_quote = string_quote
        self.sql_formatters = sql_formatters.copy()
        for k in db_expr_formatters.keys():
            if k not in self.sql_formatters.keys():
                self.sql_formatters[k] = db_expr_formatters[k]
        if op_replacements is None:
            op_replacements = db_default_op_replacements
        self.op_replacements = op_replacements.copy()
        self.on_start = on_start
        self.on_end = on_end
        self.on_joiner = on_joiner
        self.drop_text = drop_text
        self.string_type = string_type
        if join_name_map is None:
            join_name_map = {}
        self.join_name_map = join_name_map.copy()
        self.supports_with = supports_with

    def db_handle(self, conn):
        return DBHandle(db_model=self, conn=conn)

    def prepare_connection(self, conn):
        pass

    # database helpers

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        """

        :param conn: database connectionex
        :param q: sql query
        """
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        pandas.io.sql.execute(q, conn)

    def read_query(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        :return: query results as table
        """
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            temp_tables = dict()
            q = q.to_sql(db_model=self, temp_tables=temp_tables)
            if len(temp_tables) > 1:
                raise ValueError("ops require management of temp tables, please collect them via to_sql(temp_tables)")
        else:
            q = str(q)
        r = pandas.io.sql.read_sql(q, conn)
        r = self.local_data_model.pd.DataFrame(r)
        r = r.reset_index(drop=True)
        return r

    def table_exists(self, conn, table_name):
        q_table_name = self.quote_table_name(table_name)
        # noinspection PyBroadException
        table_exists = True
        try:
            self.read_query(conn, "SELECT * FROM " + q_table_name + " LIMIT 1")
        except Exception:
            table_exists = False
        return table_exists

    def drop_table(self, conn, table_name, *, check=True):
        if (not check) or self.table_exists(conn, table_name):
            q_table_name = self.quote_table_name(table_name)
            self.execute(conn, self.drop_text + ' ' + q_table_name)

    # noinspection PyMethodMayBeStatic,SqlNoDataSourceInspection
    def insert_table(
        self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False
    ):
        """
        Insert a table.

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :param qualifiers: schema and such
        :param allow_overwrite logical, if True drop previous table
        """

        if qualifiers is not None:
            raise ValueError("non-empty qualifiers not yet supported on insert")
        if self.table_exists(conn, table_name):
            if not allow_overwrite:
                raise ValueError("table " + table_name + " already exists")
            else:
                self.drop_table(conn, table_name, check=False)
        # Note: the Pandas to_sql() method appears to have SQLite hard-wired into it
        # it refers to sqlite_master
        d.to_sql(name=table_name, con=conn, index=False)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def read_table(self, conn, table_name, *, qualifiers=None, limit=None):
        if not isinstance(table_name, str):
            raise TypeError("Expect table_name to be a str")
        q_table_name = self.quote_table_name(
            table_name
        )
        sql = "SELECT * FROM " + q_table_name
        if limit is not None:
            sql = sql + " LIMIT " + limit.__repr__()
        return self.read_query(conn, sql)

    def read(self, conn, table):
        if table.node_name != "TableDescription":
            raise TypeError(
                "Expect table to be a data_algebra.data_ops.TableDescription"
            )
        return self.read_table(
            conn=conn, table_name=table.table_name, qualifiers=table.qualifiers
        )

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise TypeError("expected identifier to be a str")
        if self.identifier_quote in identifier:
            raise ValueError('did not expect ' + self.identifier_quote + ' in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def quote_table_name(self, table_description):
        if not isinstance(table_description, str):
            try:
                if table_description.node_name == "TableDescription":
                    table_description = table_description.table_name
                else:
                    raise TypeError(
                        "Expected table_description to be a string or data_algebra.data_ops.TableDescription)"
                    )
            except KeyError:
                raise TypeError(
                    "Expected table_description to be a string or data_algebra.data_ops.TableDescription)"
                )
        return self.quote_identifier(table_description)

    def quote_string(self, string):
        if not isinstance(string, str):
            raise TypeError("expected string to be a str")
        # replace all string with doubled string quotes
        return (
            self.string_quote
            + re.sub(self.string_quote, self.string_quote + self.string_quote, string)
            + self.string_quote
        )

    def quote_literal(self, val):
        return self.quote_string(str(val))

    def value_to_sql(self, v):
        if v is None:
            return "NULL"
        if isinstance(v, data_algebra.expr_rep.ListTerm):
            return [self.value_to_sql(vi) for vi in v.value]
        if isinstance(v, data_algebra.expr_rep.Value):
            return v.value
        if isinstance(v, str):
            return self.quote_string(v)
        if isinstance(v, bool):
            if v:
                return "TRUE"
            else:
                return "FALSE"
        if isinstance(v, float):
            if math.isnan(v):
                return "NULL"  # Pandas confuses NaN and None, so we use NULL
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, list) or isinstance(v, tuple):
            return [vi for vi in v]
        return str(v)

    def expr_to_sql(self, expression, *, want_inline_parens=False):
        if isinstance(expression, str):
            return expression
        if not isinstance(expression, data_algebra.expr_rep.PreTerm):
            raise TypeError(
                "expression should be of class data_algebra.table_rep.PreTerm"
            )
        if isinstance(expression, data_algebra.expr_rep.Value):
            return self.value_to_sql(expression.value)
        if isinstance(expression, data_algebra.expr_rep.ColumnReference):
            return self.quote_identifier(expression.column_name)
        if isinstance(expression, data_algebra.expr_rep.Expression):
            op = expression.op
            if op in self.op_replacements.keys():
                op = self.op_replacements[op]
            if op in self.sql_formatters.keys():
                return self.sql_formatters[op](self, expression)
            subs = [
                self.expr_to_sql(ai, want_inline_parens=True) for ai in expression.args
            ]
            if len(subs) == 2 and expression.inline:
                if want_inline_parens:
                    return "(" + subs[0] + " " + op.upper() + " " + subs[1] + ")"
                else:
                    # SQL window functions don't like parens
                    return subs[0] + " " + op.upper() + " " + subs[1]
            return op.upper() + "(" + ", ".join(subs) + ")"
        if isinstance(expression, data_algebra.expr_rep.ListTerm):
            return self.value_to_sql(expression.value)
        raise TypeError("unexpected type: " + str(type(expression)))

    def table_def_to_sql(self, table_def, *, using=None, temp_id_source=None):
        if table_def.node_name != "TableDescription":
            raise TypeError(
                "Expected table_def to be a data_algebra.data_ops.TableDescription)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = table_def.column_set
        missing = using - table_def.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        cols_using = [c for c in table_def.column_names if c in using]

        subsql = data_algebra.near_sql.NearSQLTable(
            terms={k: self.quote_identifier(k) for k in cols_using},
            quoted_query_name=self.quote_table_name(table_def),
            quoted_table_name=self.quote_table_name(table_def),
        )
        near_sql = subsql
        if (len(using) > 0) and (not (set([k for k in using]) == set([k for k in table_def.column_names]))):
            # need a non-trivial select here
            terms = OrderedDict()
            for k in using:
                terms[k] = k
            view_name = "table_reference_" + str(temp_id_source[0])
            temp_id_source[0] = temp_id_source[0] + 1
            near_sql = data_algebra.near_sql.NearSQLUnaryStep(
                terms=terms,  # TODO: implement pruning
                quoted_query_name=self.quote_identifier(view_name),
                sub_sql=subsql.to_bound_near_sql(columns=using),
                temp_tables=subsql.temp_tables.copy(),
            )
        return near_sql

    def extend_to_sql(self, extend_node, *, using=None, temp_id_source=None):
        if extend_node.node_name != "ExtendNode":
            raise TypeError(
                "Expected extend_node to be a data_algebra.data_ops.ExtendNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = extend_node.column_set
        using = using.union(
            extend_node.partition_by, extend_node.order_by, extend_node.reverse
        )
        subops = OrderedDict()
        for (k, op) in extend_node.ops.items():
            if k in using:
                subops[k] = op
        if len(subops) <= 0:
            # know using was not None is this case as len(extend_node.ops)>0 and all keys are in extend_node.column_set
            return extend_node.sources[0].to_near_sql_implementation(
                db_model=self, using=using, temp_id_source=temp_id_source
            )
        if len(using) < 1:
            raise ValueError("must produce at least one column")
        missing = using - extend_node.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        # get set of columns we need from subquery
        subusing = extend_node.columns_used_from_sources(using=using)[0]
        subsql = extend_node.sources[0].to_near_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        window_term = ""
        if (
            extend_node.windowed_situation
            or (len(extend_node.partition_by) > 0)
            or (len(extend_node.order_by) > 0)
        ):
            window_term = " OVER ( "
            if len(extend_node.partition_by) > 0:
                pt = [self.quote_identifier(ci) for ci in extend_node.partition_by]
                window_term = window_term + "PARTITION BY " + ", ".join(pt) + " "
            if len(extend_node.order_by) > 0:
                revs = set(extend_node.reverse)
                rt = [
                    self.quote_identifier(ci) + (" DESC" if ci in revs else "")
                    for ci in extend_node.order_by
                ]
                window_term = window_term + "ORDER BY " + ", ".join(rt) + " "
            window_term = window_term + " ) "
        terms = OrderedDict()
        origcols = [k for k in using if k not in subops.keys()]
        for k in origcols:
            terms[k] = None
        for (ci, oi) in subops.items():
            terms[ci] = self.expr_to_sql(oi) + window_term
        view_name = "extend_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            temp_tables=subsql.temp_tables.copy(),
            annotation=extend_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def project_to_sql(self, project_node, *, using=None, temp_id_source=None):
        if project_node.node_name != "ProjectNode":
            raise TypeError(
                "Expected project_node to be a data_algebra.data_ops.ProjectNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = project_node.column_set
        subops = {k: op for (k, op) in project_node.ops.items() if k in using}
        subusing = project_node.columns_used_from_sources(using=using)[0]
        terms = {ci: self.expr_to_sql(oi) for (ci, oi) in subops.items()}
        terms.update({g: None for g in project_node.group_by})
        subsql = project_node.sources[0].to_near_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "project_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        suffix = None
        if len(project_node.group_by) > 0:
            group_terms = [self.quote_identifier(c) for c in project_node.group_by]
            suffix = "GROUP BY " + ", ".join(group_terms)
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            temp_tables=subsql.temp_tables.copy(),
            annotation=project_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def select_rows_to_sql(self, select_rows_node, *, using=None, temp_id_source=None):
        if select_rows_node.node_name != "SelectRowsNode":
            raise TypeError(
                "Expected select_rows_node to be a data_algebra.data_ops.SelectRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = select_rows_node.column_set
        subusing = select_rows_node.columns_used_from_sources(using=using)[0]
        subsql = select_rows_node.sources[0].to_near_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "select_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = {ci: None for ci in using}
        suffix = " WHERE " + self.expr_to_sql(select_rows_node.expr)
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            temp_tables=subsql.temp_tables.copy(),
            annotation=select_rows_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def select_columns_to_sql(
        self, select_columns_node, *, using=None, temp_id_source=None
    ):
        if select_columns_node.node_name != "SelectColumnsNode":
            raise TypeError(
                "Expected select_columns_to_sql to be a data_algebra.data_ops.SelectColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = select_columns_node.column_set
        subusing = select_columns_node.columns_used_from_sources(using=using)[0]
        subusing = [
            c for c in select_columns_node.column_selection if c in subusing
        ]  # fix order
        subsql = select_columns_node.sources[0].to_near_sql_implementation(
            db_model=self, using=set(subusing), temp_id_source=temp_id_source
        )
        # order/limit columns
        subsql.terms = {
            k: subsql.terms[k]
            for k in select_columns_node.column_selection
            if k in subusing
        }
        return subsql

    def drop_columns_to_sql(
        self, drop_columns_node, *, using=None, temp_id_source=None
    ):
        if drop_columns_node.node_name != "DropColumnsNode":
            raise TypeError(
                "Expected drop_columns_node to be a data_algebra.data_ops.DropColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = drop_columns_node.column_set
        subusing = drop_columns_node.columns_used_from_sources(using=using)[0]
        subsql = drop_columns_node.sources[0].to_near_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        # /limit columns
        subsql.terms = {
            k: v for k, v in subsql.terms.items() if k not in drop_columns_node.column_deletions
        }
        return subsql

    def order_to_sql(self, order_node, *, using=None, temp_id_source=None):
        if order_node.node_name != "OrderRowsNode":
            raise TypeError(
                "Expected order_node to be a data_algebra.data_ops.OrderRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        using_was_None = False
        if using is None:
            using = order_node.column_set
            using_was_None = True
        subusing = order_node.columns_used_from_sources(using=using)[0]
        subusing = [c for c in order_node.column_names if c in subusing]  # fix order
        subsql = order_node.sources[0].to_near_sql_implementation(
            db_model=self, using=set(subusing), temp_id_source=temp_id_source
        )
        view_name = "order_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = None
        if not using_was_None:
            terms = {ci: None for ci in subusing}
        suffix = ""
        if len(order_node.order_columns) > 0:
            suffix = (
                suffix
                + " ORDER BY "
                + ", ".join(
                    [
                        self.quote_identifier(ci)
                        + (" DESC" if ci in set(order_node.reverse) else "")
                        for ci in order_node.order_columns
                    ]
                )
            )
        if order_node.limit is not None:
            suffix = suffix + " LIMIT " + order_node.limit.__repr__()
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            temp_tables=subsql.temp_tables.copy(),
            annotation=order_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def rename_to_sql(self, rename_node, *, using=None, temp_id_source=None):
        if rename_node.node_name != "RenameColumnsNode":
            raise TypeError(
                "Expected rename_node to be a data_algebra.data_ops.RenameColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = rename_node.column_set
        subusing = rename_node.columns_used_from_sources(using=using)[0]
        subsql = rename_node.sources[0].to_near_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "rename_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        unchanged_columns = subusing - set(rename_node.column_remapping.values()).union(
            rename_node.column_remapping.keys()
        )
        terms = {
            ki: self.quote_identifier(vi)
            for (ki, vi) in rename_node.column_remapping.items()
        }
        terms.update({vi: None for vi in unchanged_columns})
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            temp_tables=subsql.temp_tables.copy(),
            annotation=rename_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def _coalesce_terms(self, *, sub_view_name_left, sub_view_name_right, cols):
        coalesce_formatter = self.sql_formatters['coalesce']

        class PseudoExpression:
            def __init__(self, args):
                self.args = args.copy()

        terms = {ci: coalesce_formatter(
                        self,
                        PseudoExpression([
                            sub_view_name_left + "." + self.quote_identifier(ci),
                            sub_view_name_right + "." + self.quote_identifier(ci)
                    ])) for ci in cols}
        return terms

    def natural_join_to_sql(self, join_node, *, using=None, temp_id_source=None):
        if join_node.node_name != "NaturalJoinNode":
            raise TypeError(
                "Expected join_node to be a data_algebra.data_ops.NaturalJoinNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = join_node.column_set
        by_set = set(join_node.by)
        if len(using) < 1:
            raise ValueError("must select at least one column")
        missing = using - join_node.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        subusing = join_node.columns_used_from_sources(using=using.union(by_set))
        using_left = subusing[0]
        using_right = subusing[1]
        sql_left = join_node.sources[0].to_near_sql_implementation(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        sub_view_name_left = sql_left.quoted_query_name
        sql_right = join_node.sources[1].to_near_sql_implementation(
            db_model=self, using=using_right, temp_id_source=temp_id_source
        )
        sub_view_name_right = sql_right.quoted_query_name
        view_name = "natural_join_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        common = using_left.intersection(using_right)
        terms = self._coalesce_terms(
            sub_view_name_left=sub_view_name_left,
            sub_view_name_right=sub_view_name_right,
            cols=[ci for ci in common if ci in using])
        terms.update({ci: None for ci in using_left - common})
        terms.update({ci: None for ci in using_right - common})
        on_terms = ""
        if len(join_node.by) > 0:
            on_terms = (
                " ON "
                + self.on_start
                + self.on_joiner.join(
                    [
                        sub_view_name_left
                        + "."
                        + self.quote_identifier(c)
                        + " = "
                        + sub_view_name_right
                        + "."
                        + self.quote_identifier(c)
                        for c in join_node.by
                    ]
                )
                + self.on_end
                + " "
            )
        confused_temps = set(sql_left.temp_tables.keys()).intersection(
            sql_right.temp_tables.keys()
        )
        if len(confused_temps) > 0:
            raise ValueError("name collisions on temp_tables: " + str(confused_temps))
        temp_tables = sql_left.temp_tables.copy()
        temp_tables.update(sql_right.temp_tables)
        jointype = join_node.jointype
        try:
            jointype = self.join_name_map[jointype]  # TODO: maybe move this mapping earlier
        except KeyError:
            pass
        near_sql = data_algebra.near_sql.NearSQLBinaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_bound_near_sql(columns=using_left, force_sql=False),
            joiner=jointype + " JOIN",
            sub_sql2=sql_right.to_bound_near_sql(columns=using_right, force_sql=False),
            suffix=on_terms,
            temp_tables=temp_tables,
            annotation=join_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def concat_rows_to_sql(self, concat_node, *, using=None, temp_id_source=None):
        if concat_node.node_name != "ConcatRowsNode":
            raise TypeError(
                "Expected join_node to be a data_algebra.data_ops.ConcatRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = concat_node.column_set
        if len(using) < 1:
            raise ValueError("must select at least one column")
        missing = using - concat_node.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        subusing = concat_node.columns_used_from_sources(using=using)
        using_left = subusing[0]
        using_right = subusing[1]
        if set(using_left) != set(using_right):
            raise ValueError("left/right usings did not match")
        sql_left = concat_node.sources[0].to_near_sql_implementation(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        sql_right = concat_node.sources[1].to_near_sql_implementation(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        view_name = "concat_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = {ci: None for ci in using_left}
        constants_left = None
        constants_right = None
        if concat_node.id_column is not None:
            constants_left = {
                concat_node.id_column: self.quote_string(concat_node.a_name)
            }
            constants_right = {
                concat_node.id_column: self.quote_string(concat_node.b_name)
            }
            terms.update({concat_node.id_column: None})
        confused_temps = set(sql_left.temp_tables.keys()).intersection(
            sql_right.temp_tables.keys()
        )
        if len(confused_temps) > 0:
            raise ValueError("name collisions on temp_tables: " + str(confused_temps))
        temp_tables = sql_left.temp_tables.copy()
        temp_tables.update(sql_right.temp_tables)
        near_sql = data_algebra.near_sql.NearSQLBinaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_bound_near_sql(
                columns=using_left,
                force_sql=True,
                constants=constants_left,
            ),
            joiner="UNION ALL",
            sub_sql2=sql_right.to_bound_near_sql(
                columns=using_right,
                force_sql=True,
                constants=constants_right,
            ),
            temp_tables=temp_tables,
            annotation=concat_node.to_python_implementation(print_sources=False, indent=-1)
        )
        return near_sql

    def to_sql(
        self,
        ops,
        *,
        pretty=False,
        annotate=False,
        encoding=None,
        sqlparse_options=None,
        temp_tables=None,
        use_with=False
    ):
        assert isinstance(self, DBModel)
        assert isinstance(ops, data_algebra.data_ops.ViewRepresentation)
        if sqlparse_options is None:
            sqlparse_options = {"reindent": True, "keyword_case": "upper"}
        ops.columns_used()  # for table consistency check/raise
        temp_id_source = [0]
        near_sql = ops.to_near_sql_implementation(
            db_model=self, using=None, temp_id_source=temp_id_source
        )
        assert isinstance(near_sql, data_algebra.near_sql.NearSQL)
        if (near_sql.temp_tables is not None) and (len(near_sql.temp_tables) > 0):
            if temp_tables is None:
                raise ValueError(
                    "need temp_tables to be a dictionary to copy back found temporary table values"
                )
            temp_tables.update(near_sql.temp_tables)
        sql_str = None
        if use_with and self.supports_with:
            sequence = near_sql.to_with_form()
            len_sequence = len(sequence)
            # can fall back to the non-with path
            if len(sequence) >= 2:
                sql_sequence = [None] * (len_sequence - 1)
                for i in range(len_sequence - 1):
                    nmi = sequence[i][0]  # already quoted
                    sqli = sequence[i][1].to_sql(db_model=self, annotate=annotate)
                    sql_sequence[i] = f' {nmi} AS (\n {sqli} \n)'
                sql_last = sequence[len_sequence - 1].to_sql(db_model=self, force_sql=True, annotate=annotate)
                sql_str = 'WITH\n' + ',\n'.join(sql_sequence) + '\n' + sql_last
        if sql_str is None:
            # non-with path
            sql_str = near_sql.to_sql(db_model=self, force_sql=True, annotate=annotate)
        if pretty:
            sql_str = pretty_format_sql(sql_str, encoding=encoding, sqlparse_options=sqlparse_options)
        if annotate:
            model_descr = re.sub(r'\s+', ' ', str(self))
            self.string_quote
            self.identifier_quote
            sql_str = (
                f'-- data_algebra SQL https://github.com/WinVector/data_algebra\n'
                + f'--  dialect: {model_descr}\n'
                + f'--       string quote: {self.string_quote}\n'
                + f'--   identifier quote: {self.identifier_quote}\n'
                + sql_str
            )
        return sql_str

    def row_recs_to_blocks_query(
        self, source_sql, record_spec, control_view, *, using=None, temp_id_source=None
    ):
        if temp_id_source is None:
            temp_id_source = [0]
        # if not isinstance(record_spec, data_algebra.cdata.RecordSpecification):
        #     raise TypeError(
        #         "record_spec should be a data_algebra.cdata.RecordSpecification"
        #     )
        if not isinstance(control_view, data_algebra.data_ops_types.OperatorPlatform):
            raise TypeError(
                "control_view should be a data_algebra.data_ops_types.OperatorPlatform"
            )
        control_value_cols = [
            c
            for c in record_spec.control_table.columns
            if c not in record_spec.control_table_keys
        ]
        control_cols = [
            "a." + self.quote_identifier(c) for c in record_spec.record_keys
        ] + [
            "b." + self.quote_identifier(key_col)
            for key_col in record_spec.control_table_keys
        ]
        col_stmts = []
        for c in record_spec.record_keys:
            col_stmts.append(
                " a." + self.quote_identifier(c) + " AS " + self.quote_identifier(c)
            )
        for key_col in record_spec.control_table_keys:
            col_stmts.append(
                " b."
                + self.quote_identifier(key_col)
                + " AS "
                + self.quote_identifier(key_col)
            )
        seen = set()
        for result_col in control_value_cols:
            if result_col in seen:
                continue
            seen.add(result_col)
            cstmt = " CASE\n"
            col = record_spec.control_table[result_col]
            isnull = col.isnull()
            for i in range(len(col)):
                if not (isnull[i]):
                    source_col = col[i]
                    col_sql = (
                        "  WHEN CAST(b."
                        + self.quote_identifier(result_col)
                        + " AS " + self.string_type + ") = "
                        + self.quote_string(str(source_col))
                        + " THEN a."
                        + self.quote_identifier(source_col)
                        + "\n"
                    )
                    cstmt = cstmt + col_sql
            cstmt = cstmt + "  ELSE NULL END AS " + self.quote_identifier(result_col)
            col_stmts.append(cstmt)
        sql = (
            "SELECT\n"
            + ",\n".join(col_stmts)
            + "\n"
            + "FROM (\n  "
            + source_sql
            + " ) a\n"
            + "CROSS JOIN (\n  "
            + control_view.to_near_sql_implementation(
                self, using=using, temp_id_source=temp_id_source
            ).to_sql(db_model=self, columns=using, force_sql=True)
            + " ) b\n"
            + " ORDER BY "
            + ", ".join(control_cols)
        )
        return sql

    # noinspection PyUnusedLocal
    def blocks_to_row_recs_query(
        self, source_sql, record_spec, *, using=None, temp_id_source=None
    ):
        # if not isinstance(record_spec, data_algebra.cdata.RecordSpecification):
        #     raise TypeError(
        #         "record_spec should be a data_algebra.cdata.RecordSpecification"
        #     )
        control_value_cols = [
            c
            for c in record_spec.control_table.columns
            if c not in record_spec.control_table_keys
        ]
        control_cols = [self.quote_identifier(c) for c in record_spec.record_keys]
        col_stmts = []
        for c in record_spec.record_keys:
            col_stmts.append(
                " " + self.quote_identifier(c) + " AS " + self.quote_identifier(c)
            )
        seen = set()
        for i in range(record_spec.control_table.shape[0]):
            for vc in control_value_cols:
                col = record_spec.control_table[vc]
                isnull = col.isnull()
                if col[i] not in seen and not isnull[i]:
                    seen.add(col[i])
                    clauses = []
                    for cc in record_spec.control_table_keys:
                        clauses.append(
                            " ( CAST("
                            + self.quote_identifier(cc)
                            + " AS " + self.string_type + ") = "
                            + self.quote_string(str(record_spec.control_table[cc][i]))
                            + " ) "
                        )
                    cstmt = (
                        " MAX(CASE WHEN "
                        + " AND ".join(clauses)
                        + " THEN "
                        + self.quote_identifier(vc)
                        + " ELSE NULL END) AS "
                        + self.quote_identifier(col[i])
                    )
                    col_stmts.append(cstmt)
        sql = (
            "SELECT\n"
            + ",\n".join(col_stmts)
            + "\n"
            + "FROM (\n  "
            + source_sql
            + "\n"
            + " ) a\n"
            + " GROUP BY "
            + ", ".join(control_cols)
            + "\n"
            + " ORDER BY "
            + ", ".join(control_cols)
        )
        return sql

    # encode and name a term for use in a SQL expression
    def enc_term_(self, k, *, terms):
        v = None
        try:
            v = terms[k]
        except KeyError:
            pass
        if (v is None) or (v == k):
            return self.quote_identifier(k)
        return v + " AS " + self.quote_identifier(k)

    def convert_nearsql_container_subsql_(self, nearsql_container, *, annotate=False):
        assert isinstance(nearsql_container, data_algebra.near_sql.NearSQLContainer)
        if isinstance(nearsql_container.near_sql, data_algebra.near_sql.NearSQLTable):
            sql = (
                    " "
                    + nearsql_container.to_sql(self, annotate=annotate)
                    + " "
            )
            if nearsql_container.near_sql.quoted_query_name != nearsql_container.near_sql.quoted_table_name:
                sql = sql + (
                        nearsql_container.near_sql.quoted_query_name
                        + " "
                )
        elif isinstance(nearsql_container.near_sql, data_algebra.near_sql.NearSQLCommonTableExpression):
            sql = (
                    " "
                    + nearsql_container.to_sql(self, annotate=annotate)
                    + " "
            )
        else:
            sql = (
                    " ( "
                    + nearsql_container.to_sql(self, annotate=annotate)
                    + " ) "
                    + nearsql_container.near_sql.quoted_query_name
                    + " "
            )
        return sql

    def nearsqlcte_to_sql_(self, near_sql, *, columns=None, force_sql=False, constants=None, annotate=False):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLCommonTableExpression)
        if force_sql:
            return "SELECT * FROM " + near_sql.quoted_query_name
        return near_sql.quoted_query_name

    def nearsqltable_to_sql_(self, near_sql, *, columns=None, force_sql=False, constants=None, annotate=False):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLTable)
        if columns is None:
            columns = [k for k in near_sql.terms.keys()]
        if len(columns) <= 0:
            force_sql = False
        have_constants = (constants is not None) and (len(constants) > 0)
        if force_sql or have_constants:
            terms_strs = [self.quote_identifier(k) for k in columns]
            if have_constants:
                terms_strs = terms_strs + [
                    v + " AS " + self.quote_identifier(k)
                    for (k, v) in constants.items()
                ]
            if len(terms_strs) < 1:
                terms_strs = [f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}']
            return "SELECT " + ", ".join(terms_strs) + " FROM " + near_sql.quoted_table_name
        return near_sql.quoted_table_name

    def nearsqlunary_to_sql_(self, near_sql, *, columns=None, force_sql=False, constants=None, annotate=False):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLUnaryStep)
        terms_strs = ['*']  # allow * notation if nothing is specified
        terms = near_sql.terms
        if terms is not None:
            if columns is None:
                columns = [k for k in terms.keys()]
            terms_strs = [self.enc_term_(k, terms=terms) for k in columns]
            if (constants is not None) and (len(constants) > 0):
                terms_strs = terms_strs + [
                    v + " AS " + self.quote_identifier(k)
                    for (k, v) in constants.items()
                ]
            if len(terms_strs) < 1:
                terms_strs = [f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}']
        sql = "SELECT "
        if annotate and (near_sql.annotation is not None) and (len(near_sql.annotation) > 0):
            sql = sql + " -- " + _clean_annotation(near_sql.annotation) + "\n "
        sql = sql + ", ".join(terms_strs) + " FROM " + near_sql.sub_sql.convert_subsql(db_model=self, annotate=annotate)
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + " " + near_sql.suffix
        return sql

    def nearsqlbinary_to_sql_(self, near_sql, *,
                              columns=None, force_sql=False, constants=None, annotate=False, quoted_query_name=None):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLBinaryStep)
        if columns is None:
            columns = [k for k in near_sql.terms.keys()]
        terms = near_sql.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)
        terms_strs = [self.enc_term_(k, terms=terms) for k in columns]
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}']
        is_union = 'union' in near_sql.joiner.lower()
        sql = "SELECT "
        if annotate and (near_sql.annotation is not None) and (len(near_sql.annotation) > 0):
            sql = sql + " -- " + _clean_annotation(near_sql.annotation) + "\n "
        if is_union:
            substr_1 = near_sql.sub_sql1.to_sql(db_model=self, annotate=annotate)
            substr_2 = near_sql.sub_sql2.to_sql(db_model=self, annotate=annotate)
        else:
            substr_1 = near_sql.sub_sql1.convert_subsql(db_model=self, annotate=annotate)
            substr_2 = near_sql.sub_sql2.convert_subsql(db_model=self, annotate=annotate)
        sql = sql + (
                ", ".join(terms_strs) + " FROM " + " ( "
                + substr_1
                + " " + near_sql.joiner + " "
                + substr_2
                )
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + " " + near_sql.suffix
        sql = sql + " ) "
        if is_union and (quoted_query_name is not None):
            sql = sql + quoted_query_name + " "
        return sql

    def nearsqlq_to_sql_(self, near_sql, *, columns=None, force_sql=False, constants=None, annotate=False):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLq)
        if columns is None:
            columns = [k for k in near_sql.terms.keys()]
        terms = near_sql.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)

        def enc_term(k):
            v = terms[k]
            if v is None:
                return self.quote_identifier(k)
            return v + " AS " + self.quote_identifier(k)

        terms_strs = [enc_term(k) for k in columns]
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}']
        sql = "SELECT "
        if annotate and (near_sql.annotation is not None) and (len(near_sql.annotation) > 0):
            sql = sql + " -- " + _clean_annotation(near_sql.annotation) + "\n "
        return sql + (
            ", ".join(terms_strs)
            + " FROM ( "
            + near_sql.query
            + " ) "
            + near_sql.prev_quoted_query_name
        )

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        return self.__str__()


class DBHandle(data_algebra.eval_model.EvalModel):
    def __init__(self, *, db_model, conn):
        if not isinstance(db_model, DBModel):
            raise TypeError(
                "expected db_model to be of class data_algebra.db_model.DBHandle"
            )
        data_algebra.eval_model.EvalModel.__init__(self)
        self.db_model = db_model
        self.conn = conn

    def __enter__(self):
        return self

    # noinspection PyShadowingBuiltins
    def __exit__(self, type, value, traceback):
        self.close()

    def read_query(self, q):
        return self.db_model.read_query(conn=self.conn, q=q)

    def describe_table(self, table_name, *, qualifiers=None, row_limit=7):
        head = self.read_query(
            q="SELECT * FROM "
              + self.db_model.quote_table_name(table_name)
              + " LIMIT "
              + str(row_limit),
        )
        return data_algebra.data_ops.describe_table(
            head,
            table_name=table_name,
            qualifiers=qualifiers,
            row_limit=row_limit
        )

    def to_pandas(self, handle, *, data_map=None):
        if isinstance(handle, data_algebra.data_ops.TableDescription):
            handle = handle.table_name
        if not isinstance(handle, str):
            raise TypeError(
                "Expect handle to be a data_algebra.data_ops.TableDescription or str"
            )
        if data_map is not None:
            if handle not in data_map:
                return ValueError("Expected handle to be a data_map key " + handle)
            if not isinstance(data_map[handle], data_algebra.data_ops.TableDescription):
                raise ValueError(
                    "Expect data_map["
                    + handle
                    + "] to be class data_algebra.data_ops.TableDescription"
                )
            if data_map[handle].table_name != handle:
                raise ValueError(
                    "data_map["
                    + handle
                    + "].table_name == "
                    + data_map[handle].table_name
                    + ", not "
                    + handle
                )
        return self.db_model.read_table(self.conn, handle)

    def execute(self, q):
        self.db_model.execute(conn=self.conn, q=q)

    def drop_table(self, table_name):
        self.db_model.drop_table(self.conn, table_name)

    def insert_table(self, d, *, table_name, allow_overwrite=False):
        self.db_model.insert_table(
            conn=self.conn, d=d, table_name=table_name, allow_overwrite=allow_overwrite
        )
        return self.describe_table(table_name)

    def to_sql(self, ops,
               *,
               pretty=False,
               annotate=False,
               encoding=None,
               sqlparse_options=None,
               temp_tables=None,
               use_with=False):
        return self.db_model.to_sql(ops=ops,
                          pretty=pretty,
                          annotate=annotate,
                          encoding=encoding,
                          sqlparse_options=sqlparse_options,
                          temp_tables=temp_tables,
                          use_with=use_with)

    def query_to_csv(self, q, *, res_name):
        d = self.read_query(q)
        d.to_csv(res_name, index=False)

    def managed_eval(self, ops, *, data_map=None, result_name=None, narrow=True):
        query = ops.to_sql(self.db_model)
        if result_name is None:
            result_name = self.mk_tmp_name(data_map)
        tables_needed = [k for k in ops.get_tables().keys()]
        if data_map is not None:
            missing_tables = set(tables_needed) - set(data_map.keys())
            if len(missing_tables) > 0:
                raise ValueError("missing required tables: " + str(missing_tables))
            for k in tables_needed:
                if not isinstance(data_map[k], data_algebra.data_ops.TableDescription):
                    raise ValueError(
                        "Expect data_map["
                        + k
                        + "] to be class data_algebra.data_ops.TableDescription"
                    )
                if data_map[k].table_name != k:
                    raise ValueError(
                        "data_map["
                        + k
                        + "].table_name == "
                        + data_map[k].table_name
                        + ", not "
                        + k
                    )
        if result_name in tables_needed:
            raise ValueError("Can not write over an input table")
        q_table_name = self.db_model.quote_table_name(result_name)
        drop_query = self.db_model.drop_text + ' ' + q_table_name
        create_query = "CREATE TABLE " + q_table_name + " AS " + query
        # noinspection PyBroadException
        try:
            self.db_model.execute(self.conn, drop_query)
        except Exception:
            pass
        self.db_model.execute(self.conn, create_query)
        res = self.describe_table(result_name)
        if data_map is not None:
            data_map[result_name] = res
        return result_name

    def __str__(self):
        return (
            str(type(self).__name__) +
            "("
            + "db_model="
            + str(self.db_model)
            + ", conn="
            + str(self.conn)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
