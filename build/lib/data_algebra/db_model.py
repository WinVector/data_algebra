import math
import re
import io

import data_algebra

import data_algebra.near_sql
import data_algebra.expr_rep
import data_algebra.util
import data_algebra.data_ops_types
import data_algebra.eval_model
import data_algebra.data_ops


# map from op-name to special SQL formatting code


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


db_expr_formatters = {
    "is_null": _db_is_null_expr,
    "is_bad": _db_is_bad_expr,
    "if_else": _db_if_else_expr,
}


db_default_op_replacements = {"==": "=", "cumsum": "sum"}


class DBModel:
    """A model of how SQL should be generated for a given database.
       """

    identifier_quote: str
    string_quote: str

    def __init__(
        self,
        *,
        identifier_quote='"',
        string_quote="'",
        sql_formatters=None,
        op_replacements=None,
        local_data_model=None
    ):
        if local_data_model is None:
            local_data_model = data_algebra.default_data_model
        self.local_data_model = local_data_model
        if sql_formatters is None:
            sql_formatters = {}
        self.identifier_quote = identifier_quote
        self.string_quote = string_quote
        self.sql_formatters = sql_formatters.copy()
        for k in db_expr_formatters.keys():
            if k not in self.sql_formatters.keys():
                self.sql_formatters[k] = db_expr_formatters[k]
        if op_replacements is None:
            op_replacements = db_default_op_replacements
        self.op_replacements = op_replacements

    def prepare_connection(self, conn):
        pass

    # database helpers

    # noinspection SqlNoDataSourceInspection
    def insert_table(
        self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False
    ):
        """

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :param qualifiers: schema and such
        :param allow_overwrite logical, if True drop previous table
        """

        cr = [
            d.columns[i].lower()
            + " "
            + (
                "double precision"
                if self.local_data_model.can_convert_col_to_numeric(d[d.columns[i]])
                else "VARCHAR"
            )
            for i in range(d.shape[1])
        ]
        q_table_name = self.build_qualified_table_name(
            table_name, qualifiers=qualifiers
        )
        cur = conn.cursor()
        # check for table
        table_exists = True
        # noinspection PyBroadException
        try:
            self.read_query(conn, "SELECT * FROM " + q_table_name + " LIMIT 1")
        except Exception:
            table_exists = False
        if table_exists:
            if not allow_overwrite:
                raise ValueError("table " + q_table_name + " already exists")
            else:
                cur.execute("DROP TABLE " + q_table_name)
                conn.commit()
        create_stmt = "CREATE TABLE " + q_table_name + " ( " + ", ".join(cr) + " )"
        cur.execute(create_stmt)
        conn.commit()
        buf = io.StringIO(d.to_csv(index=False, header=False, sep="\t"))
        cur.copy_from(buf, "d", columns=[c for c in d.columns])
        conn.commit()

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        """
        cur = conn.cursor()
        cur.execute(q)

    def read_query(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        :return: query results as table
        """
        cur = conn.cursor()
        cur.execute(q)
        r = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        r = self.local_data_model.pd.DataFrame(columns=colnames, data=r)
        r = r.reset_index(drop=True)
        return r

    # noinspection PyMethodMayBeStatic
    def read_table(self, conn, table_name, *, qualifiers=None, limit=None):
        if not isinstance(table_name, str):
            raise TypeError("Expect table_name to be a str")
        q_table_name = self.build_qualified_table_name(
            table_name, qualifiers=qualifiers
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
            raise TypeError('did not expect " in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def build_qualified_table_name(self, table_name, *, qualifiers=None):
        qt = self.quote_identifier(table_name)
        if qualifiers is None:
            qualifiers = {}
        if len(qualifiers) > 0:
            raise ValueError("This data model does not expect table qualifiers")
        return qt

    def quote_table_name(self, table_description):
        if isinstance(table_description, str):
            return self.quote_identifier(table_description)
        if table_description.node_name != "TableDescription":
            raise TypeError(
                "Expected table_description to be a data_algebra.data_ops.TableDescription)"
            )
        return self.build_qualified_table_name(
            table_description.table_name, qualifiers=table_description.qualifiers
        )

    def quote_string(self, string):
        if not isinstance(string, str):
            raise TypeError("expected string to be a str")
        # replace all single-quotes with doubled single quotes and return surrounded by single quotes
        return (
            self.string_quote
            + re.sub(self.string_quote, self.string_quote + self.string_quote, string)
            + self.string_quote
        )

    def quote_literal(self, string):
        return self.quote_string(str(string))

    def value_to_sql(self, v):
        if v is None:
            return "NULL"
        if isinstance(v, str):
            return self.quote_string(v)
        if isinstance(v, bool):
            if v:
                return "TRUE"
            else:
                return "FALSE"
        if isinstance(v, float):
            if math.isnan(v):
                return "NULL"
            return str(v)
        return str(v)

    def expr_to_sql(self, expression, *, want_inline_parens=False):
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
        if isinstance(expression, data_algebra.expr_rep.FnCall):
            op = expression.name
            if op in self.op_replacements.keys():
                op = self.op_replacements[op]
            if op in self.sql_formatters.keys():
                return self.sql_formatters[op](self, expression)
            subs = [
                self.expr_to_sql(ai, want_inline_parens=True) for ai in expression.args
            ]
            return op.upper() + "(" + ", ".join(subs) + ")"
        if isinstance(expression, data_algebra.expr_rep.FnValue):
            op = expression.name
            if op in self.op_replacements.keys():
                op = self.op_replacements[op]
            if op in self.sql_formatters.keys():
                return self.sql_formatters[op](self, expression)
            subs = [
                self.expr_to_sql(ai, want_inline_parens=True) for ai in expression.args
            ]
            return op.upper() + "(" + ", ".join(subs) + ")"
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
        if len(using) < 1:
            raise ValueError("must select at least one column")
        missing = using - table_def.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        cols_using = [c for c in table_def.column_names if c in using]
        view_name = "table_reference_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLTable(
            terms={k: self.quote_identifier(k) for k in cols_using},
            quoted_query_name=self.quote_identifier(view_name),
            quoted_table_name=self.quote_table_name(table_def),
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
        subops = {k: op for (k, op) in extend_node.ops.items() if k in using}
        if len(subops) <= 0:
            # know using was not None is this case as len(extend_node.ops)>0 and all keys are in extend_node.column_set
            return extend_node.sources[0].to_sql_implementation(
                db_model=self, using=using, temp_id_source=temp_id_source
            )
        if len(using) < 1:
            raise ValueError("must produce at least one column")
        missing = using - extend_node.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        # get set of columns we need from subquery
        subusing = extend_node.columns_used_from_sources(using=using)[0]
        subsql = extend_node.sources[0].to_sql_implementation(
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
        terms = {ci: self.expr_to_sql(oi) + window_term for (ci, oi) in subops.items()}
        origcols = {k: None for k in using if k not in subops.keys()}
        if len(origcols) > 0:
            terms.update(origcols)
        view_name = "extend_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            previous_step_summary=subsql.summary(),
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_sql(columns=subusing, db_model=self),
            temp_tables=subsql.temp_tables.copy(),
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
        if (len(project_node.group_by) + len(subusing)) < 1:
            raise ValueError("must use at least one column")
        terms = {ci: self.expr_to_sql(oi) for (ci, oi) in subops.items()}
        terms.update({g: None for g in project_node.group_by})
        subsql = project_node.sources[0].to_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "project_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        suffix = None
        if len(project_node.group_by) > 0:
            group_terms = [self.quote_identifier(c) for c in project_node.group_by]
            suffix = "GROUP BY " + ", ".join(group_terms)
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            previous_step_summary=subsql.summary(),
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_sql(columns=subusing, db_model=self),
            suffix=suffix,
            temp_tables=subsql.temp_tables.copy(),
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
        subsql = select_rows_node.sources[0].to_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "select_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = {ci: None for ci in using}
        suffix = " WHERE " + self.expr_to_sql(select_rows_node.expr)
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            previous_step_summary=subsql.summary(),
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_sql(columns=subusing, db_model=self),
            suffix=suffix,
            temp_tables=subsql.temp_tables.copy(),
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
        subsql = select_columns_node.sources[0].to_sql_implementation(
            db_model=self, using=set(subusing), temp_id_source=temp_id_source
        )
        # see if we can order columns
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
        subsql = drop_columns_node.sources[0].to_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        return subsql

    def order_to_sql(self, order_node, *, using=None, temp_id_source=None):
        if order_node.node_name != "OrderRowsNode":
            raise TypeError(
                "Expected order_node to be a data_algebra.data_ops.OrderRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = order_node.column_set
        subusing = order_node.columns_used_from_sources(using=using)[0]
        subusing = [c for c in order_node.column_names if c in subusing]  # fix order
        subsql = order_node.sources[0].to_sql_implementation(
            db_model=self, using=set(subusing), temp_id_source=temp_id_source
        )
        view_name = "order_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
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
            previous_step_summary=subsql.summary(),
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_sql(columns=subusing, db_model=self),
            suffix=suffix,
            temp_tables=subsql.temp_tables.copy(),
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
        subsql = rename_node.sources[0].to_sql_implementation(
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
            previous_step_summary=subsql.summary(),
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_sql(columns=subusing, db_model=self),
            temp_tables=subsql.temp_tables.copy(),
        )
        return near_sql

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
        using = using.union(by_set)
        if len(using) < 1:
            raise ValueError("must select at least one column")
        missing = using - join_node.column_set
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        subusing = join_node.columns_used_from_sources(using=using)
        using_left = subusing[0]
        using_right = subusing[1]
        sql_left = join_node.sources[0].to_sql_implementation(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        sub_view_name_left = sql_left.quoted_query_name
        sql_right = join_node.sources[1].to_sql_implementation(
            db_model=self, using=using_right, temp_id_source=temp_id_source
        )
        sub_view_name_right = sql_right.quoted_query_name
        view_name = "natural_join_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        common = using_left.intersection(using_right)
        terms = {
            ci: "COALESCE("
            + sub_view_name_left
            + "."
            + self.quote_identifier(ci)
            + ", "
            + sub_view_name_right
            + "."
            + self.quote_identifier(ci)
            + ")"
            for ci in common
        }
        terms.update({ci: None for ci in using_left - common})
        terms.update({ci: None for ci in using_right - common})
        on_terms = ""
        if len(join_node.by) > 0:
            on_terms = (
                " ON "
                + ", ".join(
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
                + " "
            )
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
            sub_sql1=sql_left.to_sql(columns=using_left, db_model=self),
            previous_step_summary1=sql_left.summary(),
            joiner=join_node.jointype + " JOIN",
            sub_sql2=sql_right.to_sql(columns=using_right, db_model=self),
            previous_step_summary2=sql_right.summary(),
            suffix=on_terms,
            temp_tables=temp_tables,
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
        sql_left = concat_node.sources[0].to_sql_implementation(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        sql_right = concat_node.sources[1].to_sql_implementation(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        view_name = "concat_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = {ci: None for ci in using_left}
        constants_left = None
        constants_right = None
        if concat_node.id_column is not None:
            constants_left = {
                concat_node.id_column: self.quote_literal(concat_node.a_name)
            }
            constants_right = {
                concat_node.id_column: self.quote_literal(concat_node.b_name)
            }
            terms.update({concat_node.id_column: None})
        confused_temps = set(sql_left.temp_tables.keys()).intersection(
            sql_right.temp_tables.keys()
        )
        if len(confused_temps) > 0:
            raise ValueError("name collisions on temp_tables: " + str(confused_temps))
        temp_tables = sql_left.temp_tables.copy()
        temp_tables.update(sql_right.temp_tables)
        near_sql = data_algebra.near_sql.NearSQLUStep(
            terms=terms,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_sql(
                columns=using_left,
                db_model=self,
                force_sql=True,
                constants=constants_left,
            ),
            previous_step_summary1=sql_left.summary(),
            sub_sql2=sql_right.to_sql(
                columns=using_right,
                db_model=self,
                force_sql=True,
                constants=constants_right,
            ),
            previous_step_summary2=sql_right.summary(),
            temp_tables=temp_tables,
        )
        return near_sql

    def row_recs_to_blocks_query(
        self, source_sql, record_spec, record_view, *, using=None, temp_id_source=None
    ):
        if temp_id_source is None:
            temp_id_source = [0]
        # if not isinstance(record_spec, data_algebra.cdata.RecordSpecification):
        #     raise TypeError(
        #         "record_spec should be a data_algebra.cdata.RecordSpecification"
        #     )
        if not isinstance(record_view, data_algebra.data_ops_types.OperatorPlatform):
            raise TypeError(
                "record_view should be a data_algebra.data_ops_types.OperatorPlatform"
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
                        + " AS VARCHAR) = "
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
            + record_view.to_sql_implementation(
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
                            + " AS VARCHAR) = "
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


class DBHandle(data_algebra.eval_model.EvalModel):
    def __init__(self, db_model, conn):
        if not isinstance(db_model, DBModel):
            raise TypeError(
                "expected db_model to be of class data_algebra.db_model.DBHandle"
            )
        data_algebra.eval_model.EvalModel.__init__(self)
        self.db_model = db_model
        self.conn = conn

    def build_rep(self, table_name, *, row_limit=7):
        head = self.db_model.read_query(
            conn=self.conn,
            q="SELECT * FROM "
            + self.db_model.quote_table_name(table_name)
            + " LIMIT "
            + str(row_limit),
        )
        return data_algebra.data_ops.TableDescription(
            column_names=head.columns,
            table_name=table_name,
            head=head,
            limit_was=row_limit,
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

    def insert_table(self, d, *, table_name, allow_overwrite=False):
        self.db_model.insert_table(
            conn=self.conn, d=d, table_name=table_name, allow_overwrite=allow_overwrite
        )
        return self.build_rep(table_name)

    def eval(self, ops, *, data_map=None, result_name=None, eval_env=None, narrow=True):
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
        drop_query = "DROP TABLE " + q_table_name
        create_query = "CREATE TABLE " + q_table_name + " AS " + query
        cur = self.conn.cursor()
        # noinspection PyBroadException
        try:
            cur.execute(drop_query)
        except Exception:
            pass
        cur.execute(create_query)
        res = self.build_rep(result_name)
        if data_map is not None:
            data_map[result_name] = res
        return result_name

    def __str__(self):
        return (
            "data_algebra.db_model.DBHandle("
            + "db_model="
            + str(self.db_model)
            + ", conn="
            + str(self.conn)
            + ")"
        )

    def __repr__(self):
        return self.__str__()
