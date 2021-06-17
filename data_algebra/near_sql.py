
from abc import ABC

import data_algebra.OrderedSet


# classes for holding object for SQL generation

# TODO: build a term object that carries the column use information


# assemble sub-sql
def _convert_subsql(*, sub_sql, db_model):
    assert isinstance(sub_sql, NearSQLContainer)
    if sub_sql.near_sql.is_table:
        sql = (
                " "
                + sub_sql.to_sql(db_model)
                + " "
                + sub_sql.near_sql.quoted_query_name
                + " "
        )
    else:
        sql = (
                " ( "
                + sub_sql.to_sql(db_model)
                + " ) "
                + sub_sql.near_sql.quoted_query_name
                + " "
        )
    return sql


# encode and name a term for use in a SQL expression
def _enc_term(k, *, terms, db_model):
    v = None
    try:
        v = terms[k]
    except KeyError:
        pass
    if v is None:
        return db_model.quote_identifier(k)
    return v + " AS " + db_model.quote_identifier(k)


class NearSQL(ABC):
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(self, *, terms, quoted_query_name, temp_tables, is_table=False):
        assert isinstance(terms, dict)
        assert isinstance(quoted_query_name, str)
        assert isinstance(temp_tables, dict)
        assert isinstance(is_table, bool)
        self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.is_table = is_table
        self.temp_tables = temp_tables.copy()

    def to_near_sql(self, *, columns=None, force_sql=False, constants=None):
        return NearSQLContainer(near_sql=self, columns=columns, force_sql=force_sql, constants=constants)

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        raise NotImplementedError("base method called")


class NearSQLContainer:
    """
    NearSQL with bound in columns, force_sql, and constants decisions
    """

    def __init__(self, *,
                 near_sql=None, columns=None, force_sql=False, constants=None):
        assert isinstance(near_sql, NearSQL)
        assert isinstance(columns, (set, data_algebra.OrderedSet.OrderedSet, list, type(None)))
        assert isinstance(force_sql, bool)
        assert isinstance(constants, (dict, type(None)))
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
        return self.near_sql.to_sql(
            columns=self.columns,
            force_sql=self.force_sql,
            constants=self.constants,
            db_model=db_model)

    def to_with_form(self):
        raise Exception("not implemented yet")  # TODO: implement


class NearSQLTable(NearSQL):
    def __init__(self, *, terms, quoted_query_name, quoted_table_name):
        NearSQL.__init__(
            self, terms=terms, quoted_query_name=quoted_query_name, temp_tables=dict(), is_table=True
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


class NearSQLUnaryStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        quoted_query_name,
        sub_sql,
        suffix="",
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
        self.sub_sql = sub_sql
        self.suffix = suffix

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms = self.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)
        terms_strs = [_enc_term(k, terms=terms, db_model=db_model) for k in columns]
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
        sql = "SELECT " + ", ".join(terms_strs) + " FROM " + _convert_subsql(sub_sql=self.sub_sql, db_model=db_model)
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
        joiner,
        sub_sql2,
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
        assert isinstance(suffix,  (str, type(None)))
        assert isinstance(joiner, str)
        self.sub_sql1 = sub_sql1
        self.joiner = joiner
        self.sub_sql2 = sub_sql2
        self.suffix = suffix

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model):
        if columns is None:
            columns = [k for k in self.terms.keys()]
        terms = self.terms
        if (constants is not None) and (len(constants) > 0):
            terms.update(constants)
        terms_strs = [_enc_term(k, terms=terms, db_model=db_model) for k in columns]
        if len(terms_strs) < 1:
            terms_strs = [f'1 AS {db_model.quote_identifier("data_algebra_placeholder_col_name")}']
        sql = (
                "SELECT " + ", ".join(terms_strs) + " FROM " + " ( "
                + _convert_subsql(sub_sql=self.sub_sql1, db_model=db_model)
                + " " + self.joiner + " "
                + _convert_subsql(sub_sql=self.sub_sql2, db_model=db_model)
                )
        if (self.suffix is not None) and (len(self.suffix) > 0):
            sql = sql + " " + self.suffix
        sql = sql + " ) "
        return sql


class NearSQLq(NearSQL):
    """
    Adapter to wrap a pre-existing query as a NearSQL

    """
    def __init__(
        self, *, quoted_query_name, query, terms, prev_quoted_query_name, temp_tables
    ):
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            temp_tables=temp_tables,
        )
        assert isinstance(query, str)
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
            + self.query
            + " ) "
            + self.prev_quoted_query_name
        )
