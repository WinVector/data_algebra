# TODO: buld a term object that carries the column use information
class NearSQL:
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(self, *, terms, quoted_query_name, temp_tables):
        if (terms is None) or (not isinstance(terms, dict)) or (len(terms) <= 0):
            raise ValueError("terms is supposed to be a non-empty dictionary")
        self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.temp_tables = temp_tables

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        raise NotImplementedError("base method called")

    def summary(self):
        return {"quoted_query_name": self.quoted_query_name, "is_table": False}


class NearSQLTable(NearSQL):
    def __init__(self, *, terms, quoted_query_name, quoted_table_name):
        NearSQL.__init__(
            self, terms=terms, quoted_query_name=quoted_query_name, temp_tables=dict()
        )
        self.quoted_table_name = quoted_table_name

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        have_constants = (constants is not None) and (len(constants) > 0)
        if force_sql or have_constants:
            terms_strs = [db_model.quote_identifier(k) for k in columns]
            if have_constants:
                terms_strs = terms_strs + [
                    v + " AS " + db_model.quote_identifier(k)
                    for (k, v) in constants.items()
                ]
            return "SELECT " + ", ".join(terms_strs) + " FROM " + self.quoted_table_name
        return self.quoted_table_name

    def summary(self):
        return {"quoted_query_name": self.quoted_query_name, "is_table": True}


class NearSQLUnaryStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        quoted_query_name,
        sub_sql,
        suffix="",
        previous_step_summary,
        temp_tables
    ):
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            temp_tables=temp_tables,
        )
        self.sub_sql = sub_sql
        self.suffix = suffix
        if not isinstance(previous_step_summary, dict):
            raise TypeError("expected previous step to be a dict")
        self.previous_step_summary = previous_step_summary

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms = self.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)

        def enc_term(k):
            v = terms[k]
            if v is None:
                return db_model.quote_identifier(k)
            return v + " AS " + db_model.quote_identifier(k)

        terms_strs = [enc_term(k) for k in columns]
        if self.previous_step_summary["is_table"]:
            sql = "SELECT " + ", ".join(terms_strs) + " FROM " + self.sub_sql
        else:
            sql = (
                "SELECT "
                + ", ".join(terms_strs)
                + " FROM ( "
                + self.sub_sql
                + " ) "
                + self.previous_step_summary["quoted_query_name"]
            )
        if (self.suffix is not None) and (len(self.suffix) > 0):
            sql = sql + " " + self.suffix
        return sql


class NearSQLBinaryStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        quoted_query_name,
        sub_sql1,
        previous_step_summary1,
        joiner="",
        sub_sql2,
        previous_step_summary2,
        suffix="",
        temp_tables
    ):
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            temp_tables=temp_tables,
        )
        self.sub_sql1 = sub_sql1
        if not isinstance(previous_step_summary1, dict):
            raise TypeError("expected previous step to be a dict")
        self.previous_step_summary1 = previous_step_summary1
        self.joiner = joiner
        self.sub_sql2 = sub_sql2
        if not isinstance(previous_step_summary2, dict):
            raise TypeError("expected previous step to be a dict")
        self.previous_step_summary2 = previous_step_summary2
        self.suffix = suffix

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms = self.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)

        def enc_term(k):
            v = terms[k]
            if v is None:
                return db_model.quote_identifier(k)
            return v + " AS " + db_model.quote_identifier(k)

        terms_strs = [enc_term(k) for k in columns]
        sql = "SELECT " + ", ".join(terms_strs) + " FROM "
        if self.previous_step_summary1["is_table"]:
            sql = (
                sql
                + self.sub_sql1
                + " "
                + self.previous_step_summary1["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + "( "
                + self.sub_sql1
                + " ) "
                + self.previous_step_summary1["quoted_query_name"]
            )
        sql = sql + " " + self.joiner + " "
        if self.previous_step_summary2["is_table"]:
            sql = (
                sql
                + " "
                + self.sub_sql2
                + " "
                + self.previous_step_summary2["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + " ( "
                + self.sub_sql2
                + " ) "
                + self.previous_step_summary2["quoted_query_name"]
            )
        if (self.suffix is not None) and (len(self.suffix) > 0):
            sql = sql + " " + self.suffix
        return sql


class NearSQLUStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        quoted_query_name,
        sub_sql1,
        previous_step_summary1,
        sub_sql2,
        previous_step_summary2,
        temp_tables
    ):
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            temp_tables=temp_tables,
        )
        self.sub_sql1 = sub_sql1
        if not isinstance(previous_step_summary1, dict):
            raise TypeError("expected previous step to be a dict")
        self.previous_step_summary1 = previous_step_summary1
        self.sub_sql2 = sub_sql2
        if not isinstance(previous_step_summary2, dict):
            raise TypeError("expected previous step to be a dict")
        self.previous_step_summary2 = previous_step_summary2

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms = self.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)

        def enc_term(k):
            v = terms[k]
            if v is None:
                return db_model.quote_identifier(k)
            return v + " AS " + db_model.quote_identifier(k)

        terms_strs = [enc_term(k) for k in columns]
        sql = "SELECT " + ", ".join(terms_strs) + " FROM ( "
        if self.previous_step_summary1["is_table"]:
            sql = (
                sql
                + self.sub_sql1
                + " "
                + self.previous_step_summary1["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + "( "
                + self.sub_sql1
                + " ) "
                + self.previous_step_summary1["quoted_query_name"]
            )
        sql = sql + " UNION ALL "
        if self.previous_step_summary2["is_table"]:
            sql = (
                sql
                + " "
                + self.sub_sql2
                + " "
                + self.previous_step_summary2["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + " ( "
                + self.sub_sql2
                + " ) "
                + self.previous_step_summary2["quoted_query_name"]
            )
        sql = sql + " )"
        return sql


# TODO: get rid of uses of this class and this class
class NearSQLq(NearSQL):
    def __init__(
        self, *, quoted_query_name, query, terms, prev_quoted_query_name, temp_tables
    ):
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            temp_tables=temp_tables,
        )
        self.query = query
        self.prev_quoted_query_name = prev_quoted_query_name

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms = self.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)

        def enc_term(k):
            v = terms[k]
            if v is None:
                return db_model.quote_identifier(k)
            return v + " AS " + db_model.quote_identifier(k)

        terms_strs = [enc_term(k) for k in columns]
        return (
            "SELECT "
            + ", ".join(terms_strs)
            + " FROM ( "
            + self.query
            + " ) "
            + self.prev_quoted_query_name
        )
