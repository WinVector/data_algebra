
import re
import data_algebra.db_model
import data_algebra.table_rep

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

    def expr_to_sql(self, expression):
        if not isinstance(expression, data_algebra.table_rep.Term):
            raise Exception("expression should be of class data_algebra.table_rep.Term")
        # TODO: implement an actual tree visitor translation to SQL
        return expression.to_python()