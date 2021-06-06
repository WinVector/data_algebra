
from abc import ABC

import data_algebra.OrderedSet


# classes for holding object for SQL generation

# TODO: buld a term object that carries the column use information


class NearSQL(ABC):
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(self, *, terms, quoted_query_name, temp_tables):
        assert isinstance(terms, dict)
        assert isinstance(quoted_query_name, str)
        assert isinstance(temp_tables, dict)
        self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.temp_tables = temp_tables.copy()

    def to_near_sql(self, *, columns=None, force_sql=False, constants=None):
        return NearSQLContainer(near_sql=self, columns=columns, force_sql=force_sql, constants=constants)

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        raise NotImplementedError("base method called")

    def summary(self):
        return {"quoted_query_name": self.quoted_query_name, "is_table": False}


class NearSQLContainer:
    """
    Marked union of SQL text or NearSQL.

    This is a shim class for methods that have not yet switched from strings to NearSQL
    """

    def __init__(self, *,
                 sql_text=None,
                 near_sql=None, columns=None, force_sql=False, constants=None):
        assert isinstance(sql_text, (str, type(None)))
        assert isinstance(near_sql, (NearSQL, type(None)))
        assert isinstance(columns, (set, data_algebra.OrderedSet.OrderedSet, list, type(None)))
        assert isinstance(force_sql, bool)
        assert isinstance(constants, (dict, type(None)))
        near_sql_set = (near_sql is not None) or (columns is not None) or (constants is not None) or force_sql
        if (sql_text is None) != near_sql_set:
            raise ValueError("either sql_text must be set, or all of near_sql settings exclusively")
        self.sql_text = sql_text
        self.near_sql = near_sql
        self.columns = None
        if columns is not None:
            self.columns = columns.copy()
        self.columns = columns
        self.force_sql = force_sql
        self.constants = None
        if constants is not None:
            self.constants = constants.copy()

    def to_sql(self, db_model):
        if self.sql_text is not None:
            return self.sql_text
        return self.near_sql.to_sql(
            columns=self.columns,
            force_sql=self.force_sql,
            constants=self.constants,
            db_model=db_model)


class NearSQLTable(NearSQL):
    def __init__(self, *, terms, quoted_query_name, quoted_table_name):
        NearSQL.__init__(
            self, terms=terms, quoted_query_name=quoted_query_name, temp_tables=dict()
        )
        self.quoted_table_name = quoted_table_name

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        if len(columns) <= 0:
            force_sql = False
        have_constants = (constants is not None) and (len(constants) > 0)
        if force_sql or have_constants:
            terms_strs = [db_model.quote_identifier(k) for k in columns]
            if have_constants:
                terms_strs = terms_strs + [
                    v + " AS " + db_model.quote_identifier(k)
                    for (k, v) in constants.items()
                ]
            if len(terms_strs) < 1:
                terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
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
        assert isinstance(sub_sql, NearSQLContainer)
        assert isinstance(suffix,  (str, type(None)))
        assert isinstance(previous_step_summary, dict)
        self.sub_sql = sub_sql
        self.suffix = suffix
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
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
        if self.previous_step_summary["is_table"]:
            sql = "SELECT " + ", ".join(terms_strs) + " FROM " + self.sub_sql.to_sql(db_model)
        else:
            sql = (
                "SELECT "
                + ", ".join(terms_strs)
                + " FROM ( "
                + self.sub_sql.to_sql(db_model)
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
        assert isinstance(sub_sql1,  NearSQLContainer)
        assert isinstance(sub_sql2, NearSQLContainer)
        assert isinstance(previous_step_summary1, dict)
        assert isinstance(previous_step_summary2, dict)
        assert isinstance(suffix,  (str, type(None)))
        assert isinstance(joiner, str)
        self.sub_sql1 = sub_sql1
        self.previous_step_summary1 = previous_step_summary1
        self.joiner = joiner
        self.sub_sql2 = sub_sql2
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
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
        sql = "SELECT " + ", ".join(terms_strs) + " FROM "
        if self.previous_step_summary1["is_table"]:
            sql = (
                sql
                + self.sub_sql1.to_sql(db_model)
                + " "
                + self.previous_step_summary1["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + "( "
                + self.sub_sql1.to_sql(db_model)
                + " ) "
                + self.previous_step_summary1["quoted_query_name"]
            )
        sql = sql + " " + self.joiner + " "
        if self.previous_step_summary2["is_table"]:
            sql = (
                sql
                + " "
                + self.sub_sql2.to_sql(db_model)
                + " "
                + self.previous_step_summary2["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + " ( "
                + self.sub_sql2.to_sql(db_model)
                + " ) "
                + self.previous_step_summary2["quoted_query_name"]
            )
        if (self.suffix is not None) and (len(self.suffix) > 0):
            sql = sql + " " + self.suffix
        return sql


# UNION ALL step, can we merge this to binary step?
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
        assert isinstance(sub_sql1, NearSQLContainer)
        assert isinstance(sub_sql2, NearSQLContainer)
        assert isinstance(previous_step_summary1, dict)
        assert isinstance(previous_step_summary2, dict)
        self.sub_sql1 = sub_sql1
        self.previous_step_summary1 = previous_step_summary1
        self.sub_sql2 = sub_sql2
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
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
        sql = "SELECT " + ", ".join(terms_strs) + " FROM ( "
        if self.previous_step_summary1["is_table"]:
            sql = (
                sql
                + self.sub_sql1.to_sql(db_model)
                + " "
                + self.previous_step_summary1["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + "( "
                + self.sub_sql1.to_sql(db_model)
                + " ) "
                + self.previous_step_summary1["quoted_query_name"]
            )
        sql = sql + " UNION ALL "
        if self.previous_step_summary2["is_table"]:
            sql = (
                sql
                + " "
                + self.sub_sql2.to_sql(db_model)
                + " "
                + self.previous_step_summary2["quoted_query_name"]
            )
        else:
            sql = (
                sql
                + " ( "
                + self.sub_sql2.to_sql(db_model)
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
        assert isinstance(query, NearSQLContainer)
        assert isinstance(prev_quoted_query_name, str)
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
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
        return (
            "SELECT "
            + ", ".join(terms_strs)
            + " FROM ( "
            + self.query.to_sql(db_model)
            + " ) "
            + self.prev_quoted_query_name
        )
