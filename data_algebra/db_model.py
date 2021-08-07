import math
import re
from collections import OrderedDict
from types import SimpleNamespace

import pandas.io.sql

import data_algebra

import data_algebra.near_sql
import data_algebra.expr_rep
import data_algebra.util
import data_algebra.data_ops_types
import data_algebra.eval_model
import data_algebra.data_ops


class SQLFormatOptions(SimpleNamespace):
    def __init__(self,
                 use_with=True,
                 annotate=True,
                 sql_indent=' ',
                 initial_commas=False,
                 ):
        SimpleNamespace.__init__(
            self,
            use_with=use_with,
            annotate=annotate,
            sql_indent=sql_indent,
            initial_commas=initial_commas)


def _str_join_expecting_list(joiner, str_list):
    assert isinstance(joiner, str)
    assert isinstance(str_list, list)
    assert all([isinstance(vi, str) for vi in str_list])
    return joiner.join(str_list)


def _clean_annotation(annotation):
    assert isinstance(annotation, (str, type(None)))
    if annotation is None:
        return annotation
    annotation = annotation.strip()
    annotation = re.sub(r"\s+", " ", annotation)
    return annotation


# map from op-name to special SQL formatting code


def _db_lag_expr(dbmodel, expression):
    arg_0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    if len(expression.args) == 1:
        return f'LAG({arg_0})'
    elif len(expression.args) == 2:
        periods = expression.args[1].value
        if periods > 0:
            return f'LAG({arg_0}, {periods})'
        elif periods < 0:
            return f'LEAD({arg_0}, {-periods})'
        else:
            return f'{arg_0}'
    else:
        raise ValueError("too many arguments to SQL LAG/LEAD")


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
        + "("
        + x_expr
        + ") >= ("
        + y_expr
        + ")"
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + "("
        + x_expr
        + ") >= ("
        + y_expr
        + ")"
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
        + "("
        + x_expr
        + ") <= ("
        + y_expr
        + ")"
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + "("
        + x_expr
        + ") <= ("
        + y_expr
        + ")"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_is_in_expr(dbmodel, expression):
    is_in_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    x_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return "(" + is_in_expr + " IN " + x_expr + ")"


# noinspection PyUnusedLocal
def _db_count_expr(dbmodel, expression):
    return "SUM(1)"


def _db_concat_expr(dbmodel, expression):
    return (
        "("  # TODO: cast each to char on way in
        + " || ".join(
            [dbmodel.expr_to_sql(ai, want_inline_parens=True) for ai in expression.args]
        )
        + ")"
    )


def _db_coalesce_expr(dbmodel, expression):
    return (
        "COALESCE("
        + ", ".join(
            [
                dbmodel.expr_to_sql(ai, want_inline_parens=False)
                for ai in expression.args
            ]
        )
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
        "COUNT(DISTINCT ("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + "))"
    )


# fns that had been in bigquery_user_fns


def _as_int64(dbmodel, expression):
    return (
        "CAST("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " AS INT64)"
    )


def _as_str(dbmodel, expression):
    return (
        "CAST("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " AS "
        + dbmodel.string_type
        + ")"
    )


def _trimstr(dbmodel, expression):
    return (
        "SUBSTR("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", 1 + "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[2], want_inline_parens=False)
        + ")"
    )


def _datetime_to_date(dbmodel, expression):
    return (
        "DATE("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _parse_datetime(dbmodel, expression):
    return (
        "PARSE_DATETIME("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _parse_date(dbmodel, expression):
    return (
        "PARSE_DATE("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _format_datetime(dbmodel, expression):
    return (
        "FORMAT_DATETIME("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _format_date(dbmodel, expression):
    return (
        "FORMAT_DATE("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _dayofweek(dbmodel, expression):
    return (
        "EXTRACT(DAYOFWEEK FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _dayofyear(dbmodel, expression):
    return (
        "EXTRACT(DAYOFYEAR FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _dayofmonth(dbmodel, expression):
    return (
        "EXTRACT(DAY FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _weekofyear(dbmodel, expression):
    return (
        "EXTRACT(WEEK FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _month(dbmodel, expression):
    return (
        "EXTRACT(MONTH FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _quarter(dbmodel, expression):
    return (
        "EXTRACT(QUARTER FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _year(dbmodel, expression):
    return (
        "EXTRACT(YEAR FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _timestamp_diff(dbmodel, expression):
    return (
        "TIMESTAMP_DIFF("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", SECOND)"
    )


def _date_diff(dbmodel, expression):
    return (
        "TIMESTAMP_DIFF("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", DAY)"
    )


def _base_Sunday(dbmodel, expression):
    subex = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    return f"DATE_SUB({subex}, INTERVAL (EXTRACT(DAYOFWEEK FROM {subex}) - 1) DAY)"


db_expr_formatters = {
    "shift": _db_lag_expr,
    "is_null": _db_is_null_expr,
    "is_bad": _db_is_bad_expr,
    "mean": _db_mean_expr,
    "size": _db_size_expr,
    "if_else": _db_if_else_expr,
    "is_in": _db_is_in_expr,
    "maximum": _db_maximum_expr,
    "minimum": _db_minimum_expr,
    "count": _db_count_expr,
    "concat": _db_concat_expr,
    "coalesce": _db_coalesce_expr,
    "round": _db_round_expr,
    "floor": _db_floor_expr,
    "ceil": _db_ceil_expr,
    "**": _db_pow_expr,
    "nunique": _db_nunique_expr,
    # fns that had been in bigquery_user_fns
    "as_int64": _as_int64,
    "as_str": _as_str,
    "trimstr": _trimstr,
    "datetime_to_date": _datetime_to_date,
    "parse_datetime": _parse_datetime,
    "parse_date": _parse_date,
    "format_datetime": _format_datetime,
    "format_date": _format_date,
    "dayofweek": _dayofweek,
    "dayofyear": _dayofyear,
    "dayofmonth": _dayofmonth,
    "weekofyear": _weekofyear,
    "month": _month,
    "quarter": _quarter,
    "year": _year,
    "timestamp_diff": _timestamp_diff,
    "date_diff": _date_diff,
    "base_Sunday": _base_Sunday,
}


db_default_op_replacements = {
    "==": "=",
    "cumsum": "sum",
    'and': "AND",
    "&": "AND",
    "&&": "AND",
    'or': 'OR',
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
    allow_extend_merges: bool
    default_SQL_format_options:SQLFormatOptions
    union_all_term_start: str
    union_all_term_end: str

    def __init__(
        self,
        *,
        identifier_quote='"',
        string_quote="'",
        sql_formatters=None,
        op_replacements=None,
        local_data_model=None,
        on_start="",
        on_end="",
        on_joiner="AND",
        drop_text="DROP TABLE",
        string_type="VARCHAR",
        join_name_map=None,
        supports_with=True,
        allow_extend_merges=True,
        default_SQL_format_options=None,
        union_all_term_start='(',
        union_all_term_end=')',
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
        if default_SQL_format_options is None:
            default_SQL_format_options = SQLFormatOptions()
        assert isinstance(default_SQL_format_options, SQLFormatOptions)
        self.default_SQL_format_options = default_SQL_format_options
        self.supports_with = supports_with
        self.allow_extend_merges = allow_extend_merges
        self.union_all_term_start = union_all_term_start
        self.union_all_term_end=union_all_term_end

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
            q = q.to_sql(db_model=self)
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
            self.execute(conn, self.drop_text + " " + q_table_name)

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
        assert isinstance(table_name, str)
        q_table_name = self.quote_table_name(table_name)
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
        assert isinstance(identifier, str)
        if self.identifier_quote in identifier:
            raise ValueError(
                "did not expect " + self.identifier_quote + " in identifier"
            )
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
        assert isinstance(string, str)
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
            return '(' + ', '.join([self.value_to_sql(vi) for vi in v.value]) + ')'
        if isinstance(v, data_algebra.expr_rep.Value):
            return self.value_to_sql(v.value)
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
            return '(' + ', '.join([self.value_to_sql(vi) for vi in v]) + ')'
        return str(v)

    def table_values_to_sql(self, v):
        assert v is not None
        m = v.shape[0]
        assert m > 0
        n = v.shape[1]
        assert n > 0
        qi = self.quote_identifier
        qv = self.value_to_sql

        def q_row(i):
            return (
                    self.union_all_term_start
                    + 'SELECT '
                    + ', '.join([f'{qv(v[v.columns[j]][i])} AS {qi(v.columns[j])}' for j in range(n)])
                    + self.union_all_term_end
                )

        sql = (
            [
                'SELECT',
                ' *',
                'FROM ('
            ]
            + ['    ' + ('' if (i < 1) else 'UNION ALL ') + q_row(i) for i in range(m)]
            + [f') {qi("table_values")}']
        )
        return '\n'.join(sql) + '\n'

    def expr_to_sql(self, expression, *, want_inline_parens=False):
        if isinstance(expression, str):
            return expression
        assert isinstance(expression, data_algebra.expr_rep.PreTerm)
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
            if (len(expression.args) > 1) and expression.inline:
                subs = [
                    self.expr_to_sql(ai, want_inline_parens=True) for ai in expression.args
                ]
                res = ''
                if want_inline_parens:
                    res = res + '('
                assert len(subs) > 0
                if len(subs) == 1:
                    res = op.upper() + subs[0]
                else:
                    res = res + (' ' + op.upper() + ' ').join(subs)
                if want_inline_parens:
                    res = res + ')'
                return res
            subs = [
                self.expr_to_sql(ai, want_inline_parens=False) for ai in expression.args
            ]
            return op.upper() + "(" + ", ".join(subs) + ")"
        if isinstance(expression, data_algebra.expr_rep.ListTerm):
            return self.value_to_sql(expression.value)
        raise TypeError("unexpected type: " + str(type(expression)))


    def _indent_and_sep_terms(self, terms, *, sep=',', sql_format_options=None):
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        n = len(terms)
        assert n >= 1
        comma_spacer = ' ' * len(sep)
        if sql_format_options.initial_commas:
            return [sql_format_options.sql_indent + (comma_spacer if i == 0 else sep) + ' ' + terms[i]
                    for i in range(n)]
        return [sql_format_options.sql_indent + terms[i] + ((' ' + sep) if i < (n - 1) else '')
                for i in range(n)]


    def table_def_to_sql(self, table_def, *, using=None, temp_id_source=None, sql_format_options=None):
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
        if (len(using) > 0) and (
            not (set([k for k in using]) == set([k for k in table_def.column_names]))
        ):
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
            )
        return near_sql

    def extend_to_sql(self, extend_node, *, using=None, temp_id_source=None, sql_format_options=None):
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
        window_vars = set()
        if (
            extend_node.windowed_situation
            or (len(extend_node.partition_by) > 0)
            or (len(extend_node.order_by) > 0)
        ):
            window_term = " OVER ( "
            if len(extend_node.partition_by) > 0:
                pt = [self.quote_identifier(ci) for ci in extend_node.partition_by]
                window_term = window_term + "PARTITION BY " + ", ".join(pt) + " "
                window_vars.update(extend_node.partition_by)
            if len(extend_node.order_by) > 0:
                revs = set(extend_node.reverse)
                rt = [
                    self.quote_identifier(ci) + (" DESC" if ci in revs else "")
                    for ci in extend_node.order_by
                ]
                window_term = window_term + "ORDER BY " + ", ".join(rt) + " "
                window_vars.update(extend_node.order_by)
            window_term = window_term + " ) "
        terms = OrderedDict()
        declared_term_dependencies = OrderedDict()
        origcols = [k for k in using if k not in subops.keys()]
        for k in origcols:
            terms[k] = None
            declared_term_dependencies[k] = set([k])
        for (ci, oi) in subops.items():
            terms[ci] = self.expr_to_sql(oi) + window_term
            cols_used_in_term = set()
            oi.get_column_names(cols_used_in_term)
            cols_used_in_term.update(window_vars)
            declared_term_dependencies[ci] = cols_used_in_term
        annotation = extend_node.to_python_implementation(
            print_sources=False, indent=-1
        )
        # TODO: see if we can merge with subsql instead of building a new one
        if (self.allow_extend_merges
            and isinstance(subsql, data_algebra.near_sql.NearSQLUnaryStep)
            and subsql.mergeable
            and (subsql.declared_term_dependencies is not None)
            and ((subsql.suffix is None) or (len(subsql.suffix) == 0))):
            # check detailed merge conditions
            def non_trivial_terms(*, dep_dict, term_dict):
                return [k for k, v in dep_dict.items() if (len(v - set([k])) > 0) or (k not in v) or
                        ((term_dict[k] is not None) and (term_dict[k] != k))]
            our_non_trivial_terms = non_trivial_terms(
                dep_dict=declared_term_dependencies,
                term_dict=terms)
            our_needs = set().union(*[declared_term_dependencies[k] for k in our_non_trivial_terms])
            sub_non_trivial_terms = non_trivial_terms(
                dep_dict=subsql.declared_term_dependencies,
                term_dict=subsql.terms)
            sub_needs = set().union(*[subsql.declared_term_dependencies[k] for k in sub_non_trivial_terms])
            contention = set().union(
                set(our_non_trivial_terms).intersection(sub_non_trivial_terms),
                set(our_non_trivial_terms).intersection(sub_needs),
                set(sub_non_trivial_terms).intersection(our_needs))
            if len(contention) == 0:
                # merge our stuff into subsql
                subsql.annotation = subsql.annotation + "." + annotation
                for k in our_non_trivial_terms:
                    subsql.terms[k] = terms[k]
                    subsql.declared_term_dependencies[k] = declared_term_dependencies[k]
                we_use = set.union(set(terms.keys()), set(declared_term_dependencies.keys()))
                excess_sub_term_keys = set(subsql.terms.keys()) - we_use
                for k in excess_sub_term_keys:
                    del subsql.terms[k]
                excess_sub_declared_keys = set(subsql.declared_term_dependencies.keys()) - we_use
                for k in excess_sub_declared_keys:
                    del subsql.declared_term_dependencies[k]
                return subsql
        view_name = "extend_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            annotation=annotation,
            mergeable=True,
            declared_term_dependencies=declared_term_dependencies,
        )
        return near_sql

    def project_to_sql(self, project_node, *, using=None, temp_id_source=None, sql_format_options=None):
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
        suffix = []
        if len(project_node.group_by) > 0:
            group_terms = [self.quote_identifier(c) for c in project_node.group_by]
            suffix = (
                    ["GROUP BY"]
                    + self._indent_and_sep_terms(group_terms, sql_format_options=sql_format_options)
            )
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            annotation=project_node.to_python_implementation(
                print_sources=False, indent=-1
            ),
        )
        return near_sql

    def select_rows_to_sql(self, select_rows_node, *, using=None, temp_id_source=None, sql_format_options=None):
        if select_rows_node.node_name != "SelectRowsNode":
            raise TypeError(
                "Expected select_rows_node to be a data_algebra.data_ops.SelectRowsNode)"
            )
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
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
        suffix = (
                ["WHERE"]
                + [sql_format_options.sql_indent + self.expr_to_sql(select_rows_node.expr)]
        )
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            annotation=select_rows_node.to_python_implementation(
                print_sources=False, indent=-1
            ),
        )
        return near_sql

    def select_columns_to_sql(
        self, select_columns_node, *, using=None, temp_id_source=None, sql_format_options=None
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
        self, drop_columns_node, *, using=None, temp_id_source=None, sql_format_options=None
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
            k: v
            for k, v in subsql.terms.items()
            if k not in drop_columns_node.column_deletions
        }
        return subsql

    def order_to_sql(self, order_node, *, using=None, temp_id_source=None, sql_format_options=None):
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
        suffix = []
        if len(order_node.order_columns) > 0:
            suffix = (
                suffix
                + ["ORDER BY"]
                + self._indent_and_sep_terms(
                    [
                        self.quote_identifier(ci)
                        + (" DESC" if ci in set(order_node.reverse) else "")
                        for ci in order_node.order_columns
                    ],
                    sql_format_options=sql_format_options
                )
            )
        if order_node.limit is not None:
            suffix = (
                    suffix
                    + ["LIMIT " + order_node.limit.__repr__()]
            )
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            annotation=order_node.to_python_implementation(
                print_sources=False, indent=-1
            ),
        )
        return near_sql

    def rename_to_sql(self, rename_node, *, using=None, temp_id_source=None, sql_format_options=None):
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
            annotation=rename_node.to_python_implementation(
                print_sources=False, indent=-1
            ),
        )
        return near_sql

    def _coalesce_terms(self, *, sub_view_name_left, sub_view_name_right, cols):
        coalesce_formatter = self.sql_formatters["coalesce"]

        class PseudoExpression:
            def __init__(self, args):
                self.args = args.copy()

        terms = {
            ci: coalesce_formatter(
                self,
                PseudoExpression(
                    [
                        sub_view_name_left + "." + self.quote_identifier(ci),
                        sub_view_name_right + "." + self.quote_identifier(ci),
                    ]
                ),
            )
            for ci in cols
        }
        return terms

    def natural_join_to_sql(self, join_node, *, using=None, temp_id_source=None, sql_format_options=None):
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
            cols=[ci for ci in common if ci in using],
        )
        terms.update({ci: None for ci in using_left - common})
        terms.update({ci: None for ci in using_right - common})
        on_terms = []
        if len(join_node.by) > 0:
            on_terms = (
                ["ON " + self.on_start]
                + self._indent_and_sep_terms(
                    [
                        sub_view_name_left
                        + "."
                        + self.quote_identifier(c)
                        + " = "
                        + sub_view_name_right
                        + "."
                        + self.quote_identifier(c)
                        for c in join_node.by
                    ],
                    sep=self.on_joiner,
                    sql_format_options=sql_format_options
                )
            )
            if (self.on_end is not None) and (len(self.on_end) > 0):
                on_terms = on_terms + [self.on_end]
        jointype = join_node.jointype
        try:
            jointype = self.join_name_map[
                jointype
            ]  # TODO: maybe move this mapping earlier
        except KeyError:
            pass
        near_sql = data_algebra.near_sql.NearSQLBinaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_bound_near_sql(columns=using_left, force_sql=False),
            joiner=jointype + " JOIN",
            sub_sql2=sql_right.to_bound_near_sql(columns=using_right, force_sql=False),
            suffix=on_terms,
            annotation=join_node.to_python_implementation(
                print_sources=False, indent=-1
            ),
        )
        return near_sql

    def concat_rows_to_sql(self, concat_node, *, using=None, temp_id_source=None, sql_format_options=None):
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
        near_sql = data_algebra.near_sql.NearSQLBinaryStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_bound_near_sql(
                columns=using_left, force_sql=True, constants=constants_left,
            ),
            joiner="UNION ALL",
            sub_sql2=sql_right.to_bound_near_sql(
                columns=using_right, force_sql=True, constants=constants_right,
            ),
            annotation=concat_node.to_python_implementation(
                print_sources=False, indent=-1
            ),
        )
        return near_sql

    def to_sql(
        self,
        ops,
        *,
        sql_format_options=None,
    ):
        assert isinstance(self, DBModel)
        assert isinstance(ops, data_algebra.data_ops.ViewRepresentation)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        ops.columns_used()  # for table consistency check/raise
        temp_id_source = [0]
        near_sql = ops.to_near_sql_implementation(
            db_model=self, using=None, temp_id_source=temp_id_source
        )
        assert isinstance(near_sql, data_algebra.near_sql.NearSQL)
        sql_str_list = None
        if sql_format_options.use_with and self.supports_with:
            sequence = near_sql.to_with_form()
            len_sequence = len(sequence)
            # can fall back to the non-with path
            if len(sequence) >= 2:
                sql_sequence = []
                for i in range(len_sequence - 1):
                    nmi = sequence[i][0]  # already quoted
                    sqli = sequence[i][1].to_sql(db_model=self, sql_format_options=sql_format_options)
                    sql_sequence = (
                        sql_sequence +
                        [f"{sql_format_options.sql_indent}{nmi} AS ("] +
                        [sql_format_options.sql_indent + sql_format_options.sql_indent +
                         s for s in sqli]
                    )
                    if i < (len_sequence - 2):
                        sql_sequence = sql_sequence + [sql_format_options.sql_indent + '),']
                    else:
                        sql_sequence = sql_sequence + [sql_format_options.sql_indent + ')']
                sql_last = sequence[len_sequence - 1].to_sql(
                    db_model=self, force_sql=True, sql_format_options=sql_format_options
                )
                sql_str_list = (
                        ["WITH"] +
                        sql_sequence +
                        sql_last
                )
        if sql_str_list is None:
            # non-with path
            sql_str_list = near_sql.to_sql(db_model=self, force_sql=True, sql_format_options=sql_format_options)
        if sql_format_options.annotate:
            model_descr = re.sub(r"\s+", " ", str(self))
            sql_str_list = (
                    [
                        f"-- data_algebra SQL https://github.com/WinVector/data_algebra",
                        f"--  dialect: {model_descr}",
                        f"--       string quote: {self.string_quote}",
                        f"--   identifier quote: {self.identifier_quote}",
                    ]
                    + sql_str_list
                )
        return '\n'.join(sql_str_list) + '\n'

    def row_recs_to_blocks_query(
        self, source_sql, record_spec, *, using=None, temp_id_source=None
    ):
        if temp_id_source is None:
            temp_id_source = [0]
        if isinstance(source_sql, str):
            source_sql = [source_sql]
        # if not isinstance(record_spec, data_algebra.cdata.RecordSpecification):
        #     raise TypeError(
        #         "record_spec should be a data_algebra.cdata.RecordSpecification"
        #     )
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
                        + " AS "
                        + self.string_type
                        + ") = "
                        + self.quote_string(str(source_col))
                        + " THEN a."
                        + self.quote_identifier(source_col)
                        + "\n"
                    )
                    cstmt = cstmt + col_sql
            cstmt = cstmt + "  ELSE NULL END AS " + self.quote_identifier(result_col)
            col_stmts.append(cstmt)
        ctab_sql = [self.table_values_to_sql(record_spec.control_table)]
        sql = (
            "SELECT\n"
            + _str_join_expecting_list(",\n", col_stmts)
            + "\n"
            + "FROM (\n  "
            + _str_join_expecting_list('\n', source_sql)
            + " ) a\n"
            + "CROSS JOIN (\n  "
            + _str_join_expecting_list('\n', ctab_sql)
            + " ) b\n"
            + " ORDER BY "
            + _str_join_expecting_list(", ", control_cols)
        )
        return sql

    # noinspection PyUnusedLocal
    def blocks_to_row_recs_query(
        self, source_sql, record_spec, *, using=None, temp_id_source=None, sql_format_options=None
    ):
        # if not isinstance(record_spec, data_algebra.cdata.RecordSpecification):
        #     raise TypeError(
        #         "record_spec should be a data_algebra.cdata.RecordSpecification"
        #     )
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if isinstance(source_sql, str):
            source_sql = [source_sql]
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
                            + " AS "
                            + self.string_type
                            + ") = "
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
            + _str_join_expecting_list(",\n", col_stmts)
            + "\n"
            + "FROM (\n  "
            + _str_join_expecting_list('\n', source_sql)
            + "\n"
            + " ) a\n"
            + " GROUP BY "
            + _str_join_expecting_list(", ", control_cols)
            + "\n"
            + " ORDER BY "
            + _str_join_expecting_list(", ", control_cols)
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

    def convert_nearsql_container_subsql_(self, nearsql_container, *, sql_format_options=None):
        assert isinstance(nearsql_container, data_algebra.near_sql.NearSQLContainer)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        sub_sql = nearsql_container.to_sql(self, sql_format_options=sql_format_options)
        assert isinstance(sub_sql, list)
        if isinstance(nearsql_container.near_sql, data_algebra.near_sql.NearSQLTable):
            sql = sub_sql
            if (
                nearsql_container.near_sql.quoted_query_name
                != nearsql_container.near_sql.quoted_table_name
            ):
                sql = sql + [nearsql_container.near_sql.quoted_query_name]
        elif isinstance(
            nearsql_container.near_sql,
            data_algebra.near_sql.NearSQLCommonTableExpression,
        ):
            sql = sub_sql
        else:
            sql = (
                ["("]
                + sub_sql
                + [") " + nearsql_container.near_sql.quoted_query_name]
            )
        return sql

    def nearsqlcte_to_sql_(
        self, near_sql, *, columns=None, force_sql=False, constants=None, sql_format_options=None
    ):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLCommonTableExpression)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if force_sql:
            return [
                'SELECT',
                sql_format_options.sql_indent + '*',
                'FROM ',
                sql_format_options.sql_indent + near_sql.quoted_query_name
            ]
        return [near_sql.quoted_query_name]

    def nearsqltable_to_sql_(
        self, near_sql, *, columns=None, force_sql=False, constants=None, sql_format_options=None
    ):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLTable)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
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
                terms_strs = [
                    f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}'
                ]
            return (
                ["SELECT"]
                + self._indent_and_sep_terms(terms_strs, sql_format_options=sql_format_options)
                + ["FROM"]
                + [sql_format_options.sql_indent + near_sql.quoted_table_name]
            )
        return [near_sql.quoted_table_name]

    def nearsqlunary_to_sql_(
        self, near_sql, *, columns=None, force_sql=False, constants=None, sql_format_options=None
    ):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLUnaryStep)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        terms_strs = ["*"]  # allow * notation if nothing is specified
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
                terms_strs = [
                    f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}'
                ]
        sql_start = "SELECT"
        if (
            sql_format_options.annotate
            and (near_sql.annotation is not None)
            and (len(near_sql.annotation) > 0)
        ):
            sql_start = "SELECT  -- " + _clean_annotation(near_sql.annotation)
        sql = (
            [sql_start] +
            self._indent_and_sep_terms(terms_strs, sql_format_options=sql_format_options) +
            ['FROM'] +
            [sql_format_options.sql_indent + si for si in
             near_sql.sub_sql.convert_subsql(db_model=self, sql_format_options=sql_format_options)]
        )
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + near_sql.suffix
        return sql

    def nearsqlbinary_to_sql_(
        self,
        near_sql,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        sql_format_options=None,
        quoted_query_name=None,
    ):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLBinaryStep)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if columns is None:
            columns = [k for k in near_sql.terms.keys()]
        terms = near_sql.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)
        terms_strs = [self.enc_term_(k, terms=terms) for k in columns]
        if len(terms_strs) < 1:
            terms_strs = [
                f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}'
            ]
        is_union = "union" in near_sql.joiner.lower()
        sql_start = "SELECT"
        if (
            sql_format_options.annotate
            and (near_sql.annotation is not None)
            and (len(near_sql.annotation) > 0)
        ):
            sql_start = "SELECT  -- " + _clean_annotation(near_sql.annotation)
        if is_union:
            substr_1 = near_sql.sub_sql1.to_sql(db_model=self, sql_format_options=sql_format_options)
            substr_2 = near_sql.sub_sql2.to_sql(db_model=self, sql_format_options=sql_format_options)
        else:
            substr_1 = near_sql.sub_sql1.convert_subsql(
                db_model=self, sql_format_options=sql_format_options
            )
            substr_2 = near_sql.sub_sql2.convert_subsql(
                db_model=self, sql_format_options=sql_format_options
            )
        sql = (
            [sql_start] +
            self._indent_and_sep_terms(terms_strs, sql_format_options=sql_format_options)
            + ["FROM"]
            + ["("]
            + [sql_format_options.sql_indent + si for si in substr_1]
            + [near_sql.joiner]
            + [sql_format_options.sql_indent + si for si in substr_2]
        )
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + near_sql.suffix
        if is_union and (quoted_query_name is not None) and (len(quoted_query_name)>0):
            sql = sql + [") " + quoted_query_name]
        else:
            sql = sql + [")"]
        return sql

    def nearsqlq_to_sql_(
        self, near_sql, *, columns=None, constants=None, sql_format_options=None
    ):
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLq)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
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
            terms_strs = [
                f'1 AS {self.quote_identifier("data_algebra_placeholder_col_name")}'
            ]
        sql_start = "SELECT"
        if (
            sql_format_options.annotate
            and (near_sql.annotation is not None)
            and (len(near_sql.annotation) > 0)
        ):
            sql_start = "SELECT  -- " + _clean_annotation(near_sql.annotation) + "\n "
        return (
            [sql_start] +
            self._indent_and_sep_terms(terms_strs, sql_format_options=sql_format_options)
            + ["FROM"]
            + ["("]
            + [near_sql.query]  # TODO: see if we can vectorized
            + [") " + near_sql.prev_quoted_query_name]
        )

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        return self.__str__()


class DBHandle(data_algebra.eval_model.EvalModel):
    def __init__(self, *, db_model, conn):
        assert isinstance(db_model, DBModel)
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
            head, table_name=table_name, qualifiers=qualifiers, row_limit=row_limit
        )

    def to_pandas(self, handle, *, data_map=None):
        if isinstance(handle, data_algebra.data_ops.TableDescription):
            handle = handle.table_name
        assert isinstance(handle, str)
        if data_map is not None:
            if handle not in data_map:
                return ValueError("Expected handle to be a data_map key " + handle)
            assert isinstance(data_map[handle], data_algebra.data_ops.TableDescription)
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

    def to_sql(
        self,
        ops,
        *,
        sql_format_options=None,
    ):
        return self.db_model.to_sql(
            ops=ops,
            sql_format_options=sql_format_options,
        )

    def query_to_csv(self, q, *, res_name):
        d = self.read_query(q)
        d.to_csv(res_name, index=False)

    def table_values_to_sql(self, v):
        return self.db_model.table_values_to_sql(v)

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
                assert isinstance(data_map[k], data_algebra.data_ops.TableDescription)
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
        drop_query = self.db_model.drop_text + " " + q_table_name
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
            str(type(self).__name__)
            + "("
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
