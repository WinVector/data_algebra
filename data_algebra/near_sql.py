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
                 previous_step=None):
        if (terms is None) or (not isinstance(terms, dict)) or (len(terms) <= 0):
            raise ValueError("terms is supposed to be a non-empty dictionary")
        self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.sub_sql = sub_sql
        self.suffix = suffix
        self.sub_is_table = False
        self.quoted_table_name = quoted_table_name
        if previous_step is not None:
            if not isinstance(previous_step, NearSQL):
                raise TypeError("expected previous step to be data_algebra.near_sql.NearSQL or None")
            if quoted_table_name is not None:
                raise ValueError("can not both be a table and have a previous step")
            if previous_step.quoted_table_name is not None:
                self.sub_is_table = True
            self.previuos_quoted_query_name = previous_step.quoted_query_name


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

        def enc_term(k):
            v = self.terms[k]
            if v is None:
                return db_model.quote_identifier(k)
            return v + ' AS ' + db_model.quote_identifier(k)

        terms_strs = [enc_term(k) for k in columns]
        if self.sub_is_table:
            sql = ('SELECT ' + ', '.join(terms_strs)
                   + ' FROM ' + self.sub_sql)
        else:
            sql = ('SELECT ' + ', '.join(terms_strs)
                   + ' FROM ( ' + self.sub_sql + ' ) ' + self.previuos_quoted_query_name)
        if (self.suffix is not None) and (len(self.suffix) > 0):
            sql = sql + ' ' + self.suffix
        return sql
