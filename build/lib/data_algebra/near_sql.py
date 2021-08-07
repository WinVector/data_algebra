from abc import ABC

import data_algebra.OrderedSet


# classes for holding object for SQL generation

# TODO: build a term object that carries the column use information


class NearSQL(ABC):
    """
    Represent SQL queries in a mostly string-form
    """

    def __init__(
        self, *, terms, quoted_query_name, is_table=False, annotation=None
    ):
        assert isinstance(terms, (dict, type(None)))
        assert isinstance(quoted_query_name, str)
        assert isinstance(is_table, bool)
        assert isinstance(annotation, (str, type(None)))
        self.terms = None
        if terms is not None:
            assert isinstance(terms, dict)
            if len(terms) > 0:
                self.terms = terms.copy()
        self.quoted_query_name = quoted_query_name
        self.is_table = is_table
        self.annotation = annotation

    def to_bound_near_sql(self, *, columns=None, force_sql=False, constants=None):
        return NearSQLContainer(
            near_sql=self, columns=columns, force_sql=force_sql, constants=constants
        )

    def to_sql(
        self, *, columns=None, force_sql=False, constants=None, db_model, sql_format_options=None
    ):
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

    def __init__(self, *, near_sql, columns=None, force_sql=False, constants=None):
        assert isinstance(near_sql, NearSQL)
        assert isinstance(
            columns, (set, data_algebra.OrderedSet.OrderedSet, list, type(None))
        )
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

    def to_sql(self, db_model, sql_format_options=None):
        return self.near_sql.to_sql(
            columns=self.columns,
            force_sql=self.force_sql,
            constants=self.constants,
            db_model=db_model,
            sql_format_options=sql_format_options,
        )

    # assemble sub-sql
    def convert_subsql(self, *, db_model, sql_format_options=None):
        assert isinstance(self, NearSQLContainer)
        assert isinstance(self.near_sql, NearSQL)
        return db_model.convert_nearsql_container_subsql_(
            nearsql_container=self, sql_format_options=sql_format_options
        )

    # sequence: a list where last element is a NearSQLContainer previous elements are (name, NearSQLContainer) pairs
    # stub the replacement common table expression in a NearSQLContainer
    def to_with_form_stub(self):
        stub = None
        if self.near_sql.is_table:
            stub = self
            sequence = list()
        else:
            sequence = self.near_sql.to_with_form()
            endi = len(sequence) - 1
            last_step = sequence[endi]
            sequence[endi] = NearSQLContainer(
                near_sql=last_step,
                columns=self.columns,
                force_sql=self.force_sql,
                constants=self.constants,
            )
            endi = len(sequence) - 1
            last_step = sequence[endi]
            stub = last_step
            if not stub.near_sql.is_table:
                stub = NearSQLContainer(
                    near_sql=NearSQLCommonTableExpression(
                        quoted_query_name=last_step.near_sql.quoted_query_name
                    ),
                    force_sql=self.force_sql,
                )
            sequence[endi] = (last_step.near_sql.quoted_query_name, last_step)
        return stub, sequence


class NearSQLCommonTableExpression(NearSQL):
    def __init__(self, *, quoted_query_name):
        NearSQL.__init__(
            self,
            terms=None,
            quoted_query_name=quoted_query_name,
            is_table=True,
        )

    def to_sql(
        self, *, columns=None, force_sql=False, constants=None, db_model, sql_format_options=None
    ):
        return db_model.nearsqlcte_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            sql_format_options=sql_format_options,
        )


class NearSQLTable(NearSQL):
    def __init__(self, *, terms, quoted_query_name, quoted_table_name):
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            is_table=True,
        )
        self.quoted_table_name = quoted_table_name

    def to_sql(
        self, *, columns=None, force_sql=False, constants=None, db_model, sql_format_options=None
    ):
        return db_model.nearsqltable_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            sql_format_options=sql_format_options,
        )


class NearSQLUnaryStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        quoted_query_name,
        sub_sql,
        suffix=None,
        annotation=None,
        mergeable=False,
        declared_term_dependencies=None,
    ):
        assert isinstance(mergeable, bool)
        assert isinstance(sub_sql, NearSQLContainer)
        assert isinstance(suffix, (list, type(None)))
        if not suffix is None:
            assert all([isinstance(v, str) for v in suffix])
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            annotation=annotation,
        )
        self.sub_sql = sub_sql
        self.suffix = suffix
        self.mergeable = mergeable
        self.declared_term_dependencies = None
        if declared_term_dependencies is not None:
            assert isinstance(declared_term_dependencies, dict)
            self.declared_term_dependencies = declared_term_dependencies.copy()
        else:
            self.mergeable = False

    def to_sql(
        self, *, columns=None, force_sql=False, constants=None, db_model, sql_format_options=None
    ):
        return db_model.nearsqlunary_to_sql_(
            near_sql=self,
            columns=columns,
            constants=constants,
            sql_format_options=sql_format_options,
        )

    def to_with_form(self):
        if self.sub_sql.near_sql.is_table:
            # tables don't need to be re-encoded
            sequence = list()
            sequence.append(self)
            return sequence
        # non-trivial sequence
        stub, sequence = self.sub_sql.to_with_form_stub()
        stubbed_step = NearSQLUnaryStep(
            terms=self.terms,
            quoted_query_name=self.quoted_query_name,
            sub_sql=stub,
            suffix=self.suffix,
            annotation=self.annotation,
        )
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
        suffix=None,
        annotation=None
    ):
        assert isinstance(sub_sql1, NearSQLContainer)
        assert isinstance(sub_sql2, NearSQLContainer)
        assert isinstance(suffix, (list, type(None)))
        if not suffix is None:
            assert all([isinstance(v, str) for v in suffix])
        assert isinstance(joiner, str)
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            annotation=annotation,
        )
        self.sub_sql1 = sub_sql1
        self.joiner = joiner
        self.sub_sql2 = sub_sql2
        self.suffix = suffix

    def to_sql(
        self, *, columns=None, force_sql=False, constants=None, db_model, sql_format_options=None
    ):
        return db_model.nearsqlbinary_to_sql_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            sql_format_options=sql_format_options,
            quoted_query_name=self.quoted_query_name,
        )

    def to_with_form(self):
        if self.sub_sql1.near_sql.is_table and self.sub_sql2.near_sql.is_table:
            # tables don't need to be re-encoded
            sequence = list()
            sequence.append(self)
            return sequence
        # non-trivial sequence
        stub1, sequence1 = self.sub_sql1.to_with_form_stub()
        stub2, sequence2 = self.sub_sql2.to_with_form_stub()
        sequence = list()
        seen = set()
        for stepi in sequence1:
            nmi = stepi[0]
            seen.add(nmi)
            sequence.append(stepi)
        # assume any name collisions are the same table/common_table_expression
        for stepi in sequence2:
            nmi = stepi[0]
            if not nmi in seen:
                seen.add(nmi)
                sequence.append(stepi)
        stubbed_step = NearSQLBinaryStep(
            terms=self.terms,
            quoted_query_name=self.quoted_query_name,
            sub_sql1=stub1,
            joiner=self.joiner,
            sub_sql2=stub2,
            suffix=self.suffix,
            annotation=self.annotation,
        )
        sequence.append(stubbed_step)
        return sequence


class NearSQLq(NearSQL):
    """
    Adapter to wrap a pre-existing query as a NearSQL

    """

    def __init__(
        self,
        *,
        quoted_query_name,
        query,
        terms,
        prev_quoted_query_name,
        annotation=None
    ):
        assert isinstance(query, str)
        assert isinstance(prev_quoted_query_name, str)
        NearSQL.__init__(
            self,
            terms=terms,
            quoted_query_name=quoted_query_name,
            annotation=annotation,
        )
        self.query = query
        self.prev_quoted_query_name = prev_quoted_query_name

    def to_sql(
        self, *, columns=None, force_sql=False, constants=None, db_model, sql_format_options=None
    ):
        return db_model.nearsqlq_to_sql_(
            near_sql=self,
            columns=columns,
            constants=constants,
            sql_format_options=sql_format_options,
        )
