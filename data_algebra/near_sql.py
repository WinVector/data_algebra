
class NearSQL:
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(self,
                 *,
                 terms,
                 quoted_query_name,
                 sub_sql=None,
                 suffix='',
                 quoted_table_name=None,
                 sub_is_table=False):
        if (terms is None) or (not isinstance(terms, dict)) or (len(terms)<=0):
            raise ValueError("terms is supposed to be a non-empty dictionary")
        self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.sub_sql = sub_sql
        self.suffix = suffix
        self.sub_is_table = sub_is_table
        self.quoted_table_name = quoted_table_name

    def to_sql(self,
               *,
               columns=None,
               force_sql=False,
               db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        if self.quoted_table_name is not None:
            if force_sql:
                terms_strs = [db_model.quote_identifier(k) for k in columns]
                return ('SELECT ' + ', '.join(terms_strs)
                + ' FROM ' + self.quoted_table_name)
            return self.quoted_table_name
        terms_strs = [self.terms[k] + ' AS ' + db_model.quote_identifier(k) for k in columns]
        if self.sub_is_table:
            return ('SELECT ' + ', '.join(terms_strs)
                    + ' FROM ' + self.sub_sql)
        return ('SELECT ' + ', '.join(terms_strs)
                + ' FROM ( ' + self.sub_sql + ' ) ' + self.quoted_subquery_name)
