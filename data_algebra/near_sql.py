
from abc import ABC

import data_algebra.OrderedSet


# classes for holding object for SQL generation

# TODO: build a term object that carries the column use information


class NearSQL(ABC):
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(self, *, terms, quoted_query_name, temp_tables, is_table=False, annotation=None):
        assert isinstance(terms, (dict, type(None)))
        assert isinstance(quoted_query_name, str)
        assert isinstance(temp_tables, dict)
        assert isinstance(is_table, bool)
        assert isinstance(annotation, (str, type(None)))
        self.terms = None
        if terms is not None:
            self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.is_table = is_table
        self.temp_tables = temp_tables.copy()
        self.annotation = annotation

    def to_bound_near_sql(self, *, columns=None, force_sql=False, constants=None):
        return NearSQLContainer(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants)

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model, annotate=False):
        raise NotImplementedError("base method called")

    # return a list where last element is a NearSQL previous elements are (name, NearSQLContainer) pairs
    def to_with_form(self):
        sequence = list()
        sequence.append(self)
        return sequence


class NearSQLContainer:
    """
    NearSQL with bound in columns, force_sql, and constants decisions
    """

    def __init__(self, *,
                 near_sql, columns=None, force_sql=False, constants=None):
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

    def to_sql(self, db_model, annotate=False):
        return self.near_sql.to_sql(
            columns=self.columns,
            force_sql=self.force_sql,
            constants=self.constants,
            db_model=db_model,
            annotate=annotate)

    # assemble sub-sql
    def convert_subsql(self, *, db_model, annotate=False):
        assert isinstance(self, NearSQLContainer)
        assert isinstance(self.near_sql, NearSQL)
        return db_model.convert_nearsql_container_subsql_(nearsql_container=self, annotate=annotate)

    # return a list where last element is a NearSQLContainer previous elements are (name, NearSQLContainer) pairs
    def to_with_form_c(self):
        if self.near_sql.is_table:
            sequence = list()
            sequence.append(self)
            return sequence
        sequence = self.near_sql.to_with_form()
        endi = len(sequence) - 1
        last_step = sequence[endi]
        sequence[endi] = NearSQLContainer(
            near_sql=last_step, columns=self.columns, force_sql=self.force_sql, constants=self.constants)
        return sequence


class NearSQLCommonTableExpression(NearSQL):
    def __init__(self, *, quoted_query_name):
        NearSQL.__init__(
            self, terms=None, quoted_query_name=quoted_query_name, temp_tables=dict(), is_table=True
        )

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model, annotate=False):
        return self.quoted_query_name


class NearSQLTable(NearSQL):
    def __init__(self, *, terms, quoted_query_name, quoted_table_name):
        assert isinstance(terms, dict)
        NearSQL.__init__(
            self, terms=terms, quoted_query_name=quoted_query_name, temp_tables=dict(), is_table=True
        )
        self.quoted_table_name = quoted_table_name

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model, annotate=False):
        return db_model.nearsqltable_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            annotate=annotate
        )


class NearSQLUnaryStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        quoted_query_name,
        sub_sql,
        suffix="",
        temp_tables,
        annotation=None
    ):
        assert isinstance(terms, dict)
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            temp_tables=temp_tables,
            annotation=annotation
        )
        assert isinstance(sub_sql, NearSQLContainer)
        assert isinstance(suffix,  (str, type(None)))
        self.sub_sql = sub_sql
        self.suffix = suffix

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model, annotate=False):
        return db_model.nearsqlunary_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            annotate=annotate)

    def to_with_form(self):
        if self.sub_sql.near_sql.is_table:
            # tables don't need to be re-encoded
            sequence = list()
            sequence.append(self)
            return sequence
        # non-trivial sequence
        sequence = self.sub_sql.to_with_form_c()
        endi = len(sequence) - 1
        last_step = sequence[endi]
        stub = last_step
        if not stub.near_sql.is_table:
            stub = NearSQLContainer(
                near_sql=NearSQLCommonTableExpression(quoted_query_name=last_step.near_sql.quoted_query_name)
            )
        sequence[endi] = (last_step.near_sql.quoted_query_name, last_step)
        stubbed_step = NearSQLUnaryStep(
            terms=self.terms,
            quoted_query_name=self.quoted_query_name,
            sub_sql=stub,
            suffix=self.suffix,
            temp_tables=self.temp_tables,
            annotation=self.annotation)
        sequence.append(stubbed_step)
        return sequence


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
        assert isinstance(terms, dict)
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

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model, annotate=False):
        return db_model.nearsqlbinary_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            annotate=annotate
        )


class NearSQLq(NearSQL):
    """
    Adapter to wrap a pre-existing query as a NearSQL

    """
    def __init__(
        self, *, quoted_query_name, query, terms, prev_quoted_query_name, temp_tables
    ):
        assert isinstance(terms, dict)
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

    def to_sql(self, *, columns=None, force_sql=False, constants=None, db_model, annotate=False):
        return db_model.nearsqlq_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            annotate=annotate
        )
