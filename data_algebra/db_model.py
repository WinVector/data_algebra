import math
import re
import io

import pandas

import data_algebra.expr_rep
import data_algebra.data_ops
import data_algebra.util
#import data_algebra.cdata


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


def _db_neg_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return "( -" + subexpr + " )"


db_expr_formatters = {
    "is_null": _db_is_null_expr,
    "is_bad": _db_is_bad_expr,
    "neg": _db_neg_expr,
    "if_else": _db_if_else_expr,
}

db_default_op_replacements = {
    "==": "=",
    "cumsum": "sum"
}

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
        op_replacements=None
    ):
        if sql_formatters is None:
            sql_formatters = {}
        self.identifier_quote = identifier_quote
        self.string_quote = string_quote
        if sql_formatters is None:
            sql_formatters = {}
        self.sql_formatters = sql_formatters.copy()
        for k in db_expr_formatters.keys():
            if k not in self.sql_formatters.keys():
                self.sql_formatters[k] = db_expr_formatters[k]
        if op_replacements is None:
            op_replacements = db_default_op_replacements
        self.op_replacements = op_replacements

    def prepare_connection(self, conn):
        pass

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
        if not isinstance(table_description, data_algebra.data_ops.TableDescription):
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
        return self.quote_string(string)

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
        if not isinstance(expression, data_algebra.expr_rep.Term):
            raise TypeError("expression should be of class data_algebra.table_rep.Term")
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
        raise TypeError("unexpected type: " + str(type(expression)))

    def table_def_to_sql(self, table_def, *, using=None, force_sql=False):
        if not isinstance(table_def, data_algebra.data_ops.TableDescription):
            raise TypeError(
                "Expected table_def to be a data_algebra.data_ops.TableDescription)"
            )
        if force_sql:
            if using is None:
                using = table_def.column_set
            if len(using) < 1:
                raise ValueError("must select at least one column")
            missing = using - table_def.column_set
            if len(missing) > 0:
                raise KeyError("referred to unknown columns: " + str(missing))
            cols_using = [c for c in table_def.column_names if c in using]
            cols = [self.quote_identifier(ci) for ci in cols_using]
            sql_str = (
                "SELECT "
                + ", ".join(cols)
                + " FROM "
                + self.quote_table_name(table_def)
            )
            return sql_str
        return self.quote_table_name(table_def)

    def extend_to_sql(self, extend_node, *, using=None, temp_id_source=None):
        if not isinstance(extend_node, data_algebra.data_ops.ExtendNode):
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
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        window_term = ""
        if len(extend_node.partition_by) > 0 or len(extend_node.order_by) > 0:
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
        derived = [
            self.expr_to_sql(oi) + window_term + " AS " + self.quote_identifier(ci)
            for (ci, oi) in subops.items()
        ]
        origcols = [k for k in using if k not in subops.keys()]
        if len(origcols) > 0:
            ordered_orig = [c for c in extend_node.column_names if c in set(origcols)]
            derived = [self.quote_identifier(ci) for ci in ordered_orig] + derived
        sql_str = (
            "SELECT "
            + ", ".join(derived)
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        return sql_str

    def project_to_sql(self, project_node, *, using=None, temp_id_source=None):
        if not isinstance(project_node, data_algebra.data_ops.ProjectNode):
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
        grouping = [g for g in project_node.group_by]
        derived = [
            self.expr_to_sql(oi) + " AS " + self.quote_identifier(ci)
            for (ci, oi) in subops.items()
        ]
        subsql = project_node.sources[0].to_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sql_str = (
            "SELECT "
            + ", ".join(grouping + derived)
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        if len(project_node.group_by) > 0:
            group_terms = [self.quote_identifier(c) for c in project_node.group_by]
            sql_str = sql_str + " GROUP BY " + ", ".join(group_terms)
        return sql_str

    def select_rows_to_sql(self, select_rows_node, *, using=None, temp_id_source=None):
        if not isinstance(select_rows_node, data_algebra.data_ops.SelectRowsNode):
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
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sql_str = (
            "SELECT "
            + ", ".join([self.quote_identifier(ci) for ci in using])
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
            + " WHERE "
            + self.expr_to_sql(select_rows_node.expr)
        )
        return sql_str

    def select_columns_to_sql(
        self, select_columns_node, *, using=None, temp_id_source=None
    ):
        if not isinstance(select_columns_node, data_algebra.data_ops.SelectColumnsNode):
            raise TypeError(
                "Expected select_columns_to_sql to be a data_algebra.data_ops.SelectColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = select_columns_node.column_set
        subusing = select_columns_node.columns_used_from_sources(using=using)[0]
        subsql = select_columns_node.sources[0].to_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        # TODO: make sure select rows doesn't force its partition and order columns in, and then
        #       return the subsql here instead of tacking on an additional query.
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sql_str = (
            "SELECT "
            + ", ".join([self.quote_identifier(ci) for ci in subusing])
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        return sql_str

    def drop_columns_to_sql(
        self, drop_columns_node, *, using=None, temp_id_source=None
    ):
        if not isinstance(drop_columns_node, data_algebra.data_ops.DropColumnsNode):
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
        # TODO: make sure select rows doesn't force its partition and order columns in, and then
        #       return the subsql here instead of tacking on an additional query.
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sql_str = (
            "SELECT "
            + ", ".join([self.quote_identifier(ci) for ci in subusing])
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        return sql_str

    def order_to_sql(self, order_node, *, using=None, temp_id_source=None):
        if not isinstance(order_node, data_algebra.data_ops.OrderRowsNode):
            raise TypeError(
                "Expected order_node to be a data_algebra.data_ops.OrderRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = order_node.column_set
        subusing = order_node.columns_used_from_sources(using=using)[0]
        subsql = order_node.sources[0].to_sql_implementation(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        # TODO: make sure extend doesn't force its partition and order columns in, and then
        #       return the subsql here instead of tacking on an additional query.
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sql_str = (
            "SELECT "
            + ", ".join([self.quote_identifier(ci) for ci in subusing])
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        if len(order_node.order_columns) > 0:
            sql_str = (
                sql_str
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
            sql_str = sql_str + " LIMIT " + order_node.limit.__repr__()
        return sql_str

    def rename_to_sql(self, rename_node, *, using=None, temp_id_source=None):
        if not isinstance(rename_node, data_algebra.data_ops.RenameColumnsNode):
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
        sub_view_name = "SQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        unchanged_columns = subusing - set(rename_node.column_remapping.values()).union(
            rename_node.column_remapping.keys()
        )
        copies = [self.quote_identifier(vi) for vi in unchanged_columns]
        remaps = [
            self.quote_identifier(vi) + " AS " + self.quote_identifier(ki)
            for (ki, vi) in rename_node.column_remapping.items()
        ]
        sql_str = (
            "SELECT "
            + ", ".join(copies + remaps)
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        return sql_str

    def natural_join_to_sql(self, join_node, *, using=None, temp_id_source=None):
        if not isinstance(join_node, data_algebra.data_ops.NaturalJoinNode):
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
        sql_right = join_node.sources[1].to_sql_implementation(
            db_model=self, using=using_right, temp_id_source=temp_id_source
        )
        sub_view_name_left = "LQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sub_view_name_right = "RQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        common = using_left.intersection(using_right)
        col_exprs = (
            [
                "COALESCE("
                + self.quote_identifier(sub_view_name_left)
                + "."
                + self.quote_identifier(ci)
                + ", "
                + self.quote_identifier(sub_view_name_right)
                + "."
                + self.quote_identifier(ci)
                + ") AS "
                + self.quote_identifier(ci)
                for ci in common
            ]
            + [self.quote_identifier(ci) for ci in using_left - common]
            + [self.quote_identifier(ci) for ci in using_right - common]
        )
        on_terms = ""
        if len(join_node.by) > 0:
            on_terms = (
                " ON "
                + ", ".join(
                    [
                        self.quote_identifier(sub_view_name_left)
                        + "."
                        + self.quote_identifier(c)
                        + " = "
                        + self.quote_identifier(sub_view_name_right)
                        + "."
                        + self.quote_identifier(c)
                        for c in join_node.by
                    ]
                )
                + " "
            )
        sql_str = (
            "SELECT "
            + ", ".join(col_exprs)
            + " FROM ( "
            + sql_left
            + " ) "
            + self.quote_identifier(sub_view_name_left)
            + " "
            + join_node.jointype
            + " JOIN ( "
            + sql_right
            + " ) "
            + self.quote_identifier(sub_view_name_right)
            + on_terms
        )
        return sql_str

    # database helpers

    # noinspection PyMethodMayBeStatic,SqlNoDataSourceInspection
    def insert_table(self, conn, d, table_name):
        """

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :return:
        """

        cr = [
            d.columns[i].lower()
            + " "
            + (
                "double precision"
                if data_algebra.util.can_convert_v_to_numeric(d[d.columns[i]])
                else "VARCHAR"
            )
            for i in range(d.shape[1])
        ]
        create_stmt = "CREATE TABLE " + table_name + " ( " + ", ".join(cr) + " )"
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS " + table_name)
        conn.commit()
        cur.execute(create_stmt)
        conn.commit()
        buf = io.StringIO(d.to_csv(index=False, header=False, sep="\t"))
        cur.copy_from(buf, "d", columns=[c for c in d.columns])
        conn.commit()
        return data_algebra.data_ops.TableDescription(
            table_name=table_name, column_names=[c for c in d.columns]
        )

    # noinspection PyMethodMayBeStatic
    def read_query(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        :return:
        """
        cur = conn.cursor()
        cur.execute(q)
        r = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        r = pandas.DataFrame(columns=colnames, data=r)
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
            sql = sql + " LIMIT " + limit
        return self.read_query(conn, sql)

    def read(self, conn, table):
        if not isinstance(table, data_algebra.data_ops.TableDescription):
            raise TypeError(
                "Expect table to be a data_algebra.data_ops.TableDescription"
            )
        return self.read_table(
            conn=conn, table_name=table.table_name, qualifiers=table.qualifiers
        )

    def table_description(self, conn, table_name, *, qualifiers=None):
        example = self.read_table(
            conn=conn, table_name=table_name, qualifiers=qualifiers, limit=1
        )
        return data_algebra.data_ops.TableDescription(
            table_name=table_name,
            column_names=[c for c in example.columns],
            qualifiers=qualifiers,
        )

    def row_recs_to_blocks_query(
        self, source_sql, record_spec, record_view, *, using=None, temp_id_source=None
    ):
        if temp_id_source is None:
            temp_id_source = [0]
        # if not isinstance(record_spec, data_algebra.cdata.RecordSpecification):
        #     raise TypeError(
        #         "record_spec should be a data_algebra.cdata.RecordSpecification"
        #     )
        if not isinstance(record_view, data_algebra.data_ops.ViewRepresentation):
            raise TypeError(
                "record_view should be a data_algebra.data_ops.ViewRepresentation"
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
                        "  WHEN b."
                        + self.quote_identifier(result_col)
                        + " = "
                        + self.quote_string(source_col)
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
            )
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
                            " ( "
                            + self.quote_identifier(cc)
                            + " = "
                            + self.quote_string(record_spec.control_table[cc][i])
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
