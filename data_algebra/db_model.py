import math
import re

import data_algebra.expr_rep


class DBModel:
    """A model of how SQL should be generated for a given database.
       """

    identifier_quote: str
    string_quote: str

    def __init__(self, *, identifier_quote='"', string_quote="'", sql_formatters=None):
        if sql_formatters is None:
            sql_formatters = {}
        self.identifier_quote = identifier_quote
        self.string_quote = string_quote
        self.sql_formatters = sql_formatters

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise Exception("expected identifier to be a str")
        if self.identifier_quote in identifier:
            raise Exception('did not expect " in identifier')
        return self.identifier_quote + identifier + self.identifier_quote

    def quote_table_name(self, table_description):
        qt = self.quote_identifier(table_description.table_name)
        if len(table_description.qualifiers):
            raise Exception("This data model does not expect table qualifiers")
        return qt

    def quote_string(self, string):
        if not isinstance(string, str):
            raise Exception("expected string to be a str")
        # replace all single-quotes with doubled single quotes and return surrounded by single quotes
        return (
            self.string_quote
            + re.sub(self.string_quote, self.string_quote + self.string_quote, string)
            + self.string_quote
        )

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

    def expr_to_sql(self, expression):
        if not isinstance(expression, data_algebra.expr_rep.Term):
            raise Exception("expression should be of class data_algebra.table_rep.Term")
        if isinstance(expression, data_algebra.expr_rep.Value):
            return self.value_to_sql(expression.value)
        if isinstance(expression, data_algebra.expr_rep.ColumnReference):
            return self.quote_identifier(expression.column_name)
        if isinstance(expression, data_algebra.expr_rep.Expression):
            if expression.op in self.sql_formatters.keys():
                return self.sql_formatters[expression.op](self, expression)
            subs = [self.expr_to_sql(ai) for ai in expression.args]
            if len(subs) == 2 and expression.inline:
                return "(" + subs[0] + " " + expression.op.upper() + " " + subs[1] + ")"
            return expression.op.upper() + "(" + ", ".join(subs) + ")"

        raise Exception("unexpected type: " + str(type(expression)))

    def table_def_to_sql(self, table_def, *, using=None):
        # table_def should be a data_algebra.data_ops.TableDescription
        if using is None:
            using = table_def.column_set
        if len(using) < 1:
            raise Exception("must select at least one column")
        missing = using - table_def.column_set
        if len(missing) > 0:
            raise Exception("referred to unknown columns: " + str(missing))
        cols = [self.quote_identifier(ci) for ci in using]
        sql_str = (
            "SELECT " + ", ".join(cols) + " FROM " + self.quote_table_name(table_def)
        )
        return sql_str

    def extend_to_sql(self, extend_node, *, using=None, temp_id_source=None):
        # extend_node should be a data_algebra.data_ops.ExtendNode
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = extend_node.column_set
        subops = {k: op for (k, op) in extend_node.ops.items() if k in using}
        if len(subops) <= 0:
            # know using was not None is this case as len(extend_node.ops)>0 and all keys are in extend_node.column_set
            return extend_node.sources[0].to_sql(
                db_model=self, using=using, temp_id_source=temp_id_source
            )
        using = using.union(
            extend_node.partition_by, extend_node.order_by, extend_node.reverse
        )
        if len(using) < 1:
            raise Exception("must produce at least one column")
        missing = using - extend_node.column_set
        if len(missing) > 0:
            raise Exception("referred to unknown columns: " + str(missing))
        # get set of coumns we need from subquery
        subusing = (
            using.intersection(set(extend_node.sources[0].column_names)) - subops.keys()
        )
        for (k, o) in subops.items():
            o.get_column_names(subusing)
        if len(subusing) < 1:
            raise Exception("must consume at least one column")
        subsql = extend_node.sources[0].to_sql(
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
                    for ci in extend_node.partition_by
                ]
                window_term = window_term + "ORDER BY " + ", ".join(rt) + " "
            window_term = window_term + " ) "
        derived = [
            self.expr_to_sql(oi) + window_term + " AS " + self.quote_identifier(ci)
            for (ci, oi) in subops.items()
        ]
        origcols = {k for k in using if k not in subops.keys()}
        if len(origcols) > 0:
            derived = [self.quote_identifier(ci) for ci in origcols] + derived
        sql_str = (
            "SELECT "
            + ", ".join(derived)
            + " FROM ( "
            + subsql
            + " ) "
            + self.quote_identifier(sub_view_name)
        )
        return sql_str

    def natural_join_to_sql(self, join_node, *, using=None, temp_id_source=None):
        # join_node should be a data_algebra.data_ops.NaturalJoinNode
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = join_node.column_set
        by_set = set(join_node.by)
        using = using.union(by_set)
        if len(using) < 1:
            raise Exception("must select at least one column")
        missing = using - join_node.column_set
        if len(missing) > 0:
            raise Exception("referred to unknown columns: " + str(missing))
        using_left = join_node.sources[0].column_set.intersection(using)
        using_right = join_node.sources[0].column_set.intersection(using)
        sql_left = join_node.sources[0].to_sql(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        sql_right = join_node.sources[0].to_sql(
            db_model=self, using=using_right, temp_id_source=temp_id_source
        )
        sub_view_name_left = "LQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        sub_view_name_right = "RQ_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        common = using_left.intersection(using_right)
        col_exprs = (
            [
                "COALESE("
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
        )
        return sql_str
