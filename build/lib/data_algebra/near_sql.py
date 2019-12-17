
class NearSQL:
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(self,
                 *,
                 prologue='SELECT',
                 terms,
                 query_name,
                 epilogue=''):
        self.prologue = prologue
        self.terms = terms.copy()
        self.query_name = query_name
        self.epilogue = epilogue

    def to_str(self,
               *,
               columns=None,
               db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms_strs = [self.terms[k] + ' AS ' + db_model.quote_literal(k) for k in columns]
        return (self.prologue
                + ' ' + ', '.join(terms_strs)
                + ' ' + self.epilogue)
