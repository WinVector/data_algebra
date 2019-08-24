
import re
import math

import data_algebra.db_model
import data_algebra.expr_rep


# map from op-name to special SQL formatting code
sql_formatters = {
    '___': lambda dbmodel, expression : expression.to_python()
}


class PostgreSQLModel(data_algebra.db_model.DBModel):
    """A model of how SQL should be generated for PostgreSQL"""

    def __init__(self):
        data_algebra.db_model.DBModel.__init__(self)

    def quote_table_name(self, table_description):
        nm = table_description.table_name
        if not isinstance(nm, str):
            raise Exception('expected table_description.table_name to be a str')
        if '"' in nm:
            raise Exception('did not expect " in table_description.table_name')
        # TODO: qualifiers such as schema
        ql = table_description.qualifiers
        if len(ql)>0:
            raise Exception('expected table_description.qualifiers conversion not implemented yet')
        return '"' + table_description.table_name + '"'

    def quote_identifier(self, identifier):
        if not isinstance(identifier, str):
            raise Exception('expected identifier to be a str')
        if '"' in identifier:
            raise Exception('did not expect " in identifier')
        return '"' + identifier + '"'

    def quote_string(self, string):
        if not isinstance(string, str):
            raise Exception('expected string to be a str')
        # replace all single-quotes with doubled single quotes and return surrounded by single quotes
        return "'" + re.sub("'", "''", string) + "'"

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
            if expression.op in sql_formatters.keys():
                return sql_formatters[expression.op](self, expression)
            subs = [self.expr_to_sql(ai) for ai in expression.args]
            if len(subs) == 2 and expression.inline:
                return (
                        "("
                        + subs[0]
                        + " "
                        + expression.op.upper()
                        + " "
                        + subs[1]
                        + ")"
                )
            return expression.op.upper() + "(" + ', '.join(subs) + ")"

        raise Exception("unexpected type: " + str(type(expression)))
