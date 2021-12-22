"""
Representation for operations that are nearly translated into SQL.
"""

from abc import ABC
from typing import List, Optional, Tuple

import data_algebra.OrderedSet


# classes for holding object for SQL generation

# TODO: build a term object that carries the column use information


class SQLWithList:
    """
    Carry an ordered sequence of SQL steps for use with a SQL WITH statement.
    """

    def __init__(
        self,
        *,
        last_step: "NearSQL",
        previous_steps: List[Tuple[str, "NearSQLContainer"]],
    ):
        assert isinstance(last_step, NearSQL)
        assert isinstance(previous_steps, List)
        assert all(
            [
                isinstance(vi, Tuple)
                and (len(vi) == 2)
                and isinstance(vi[0], str)
                and isinstance(vi[1], NearSQLContainer)
                for vi in previous_steps
            ]
        )
        # check no table like entities in previous steps
        assert not any([v.near_sql.is_table for k, v in previous_steps])
        # check keying is unique
        previous_keys = {k for k, v in previous_steps}
        if len(previous_keys) != len(previous_steps):
            assert len(previous_keys) == len(
                previous_steps
            )  # so we can set a breakpoint
        # check we are not coming up on a duplication
        if not last_step.is_table:
            assert last_step.quoted_query_name not in previous_keys
        self.last_step = last_step
        self.previous_steps = previous_steps


class NearSQL(ABC):
    """
    Represent SQL queries in a mostly string-form
    """

    terms: Optional[dict]
    query_name: Optional[str]
    quoted_query_name: str
    is_table: bool = False
    annotation: Optional[str] = None

    def __init__(
        self,
        *,
        terms: Optional[dict],
        query_name: Optional[str] = None,
        quoted_query_name: str,
        is_table: bool = False,
        annotation: Optional[str] = None,
    ):
        assert isinstance(terms, (dict, type(None)))
        assert isinstance(query_name, (str, type(None)))
        assert isinstance(quoted_query_name, str)
        assert isinstance(is_table, bool)
        assert isinstance(annotation, (str, type(None)))
        self.terms = None
        if terms is not None:
            assert isinstance(terms, dict)
            if len(terms) > 0:
                self.terms = terms.copy()
        self.query_name = query_name
        self.quoted_query_name = quoted_query_name
        self.is_table = is_table
        self.annotation = annotation

    def to_bound_near_sql(self, *, columns=None, force_sql=False, constants=None):
        return NearSQLContainer(
            near_sql=self, columns=columns, force_sql=force_sql, constants=constants
        )

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        raise NotImplementedError("base method called")

    def to_with_form(self) -> SQLWithList:
        raise NotImplementedError("base method called")


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

    def to_sql_str_list(self, db_model, sql_format_options=None) -> List[str]:
        return self.near_sql.to_sql_str_list(
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
    def to_with_form_stub(
        self,
    ) -> Tuple["NearSQLContainer", List[Tuple[str, "NearSQLContainer"]]]:
        if self.near_sql.is_table:
            return self, []
        in_with_form = self.near_sql.to_with_form()
        sequence = in_with_form.previous_steps
        stub = in_with_form.last_step
        if not stub.is_table:
            # replace step with a reference
            if stub.quoted_query_name not in {k for k, v in sequence}:
                sequence.append(
                    (
                        stub.quoted_query_name,
                        NearSQLContainer(near_sql=stub, force_sql=self.force_sql,),
                    )
                )
            stub = NearSQLContainer(
                near_sql=NearSQLCommonTableExpression(
                    query_name=stub.query_name,
                    quoted_query_name=stub.quoted_query_name,
                ),
                force_sql=self.force_sql,
            )
        return stub, sequence


class NearSQLNamedEntity(NearSQL):
    def __init__(self, *, terms, query_name, quoted_query_name):
        NearSQL.__init__(
            self,
            terms=terms,
            query_name=query_name,
            quoted_query_name=quoted_query_name,
            is_table=True,
        )

    def to_with_form(self) -> SQLWithList:
        return SQLWithList(last_step=self, previous_steps=[])

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        raise NotImplementedError("abstract class method called")


class NearSQLCommonTableExpression(NearSQLNamedEntity):
    def __init__(self, *, query_name, quoted_query_name):
        NearSQLNamedEntity.__init__(
            self, terms=None, query_name=query_name, quoted_query_name=quoted_query_name
        )

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlcte_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            sql_format_options=sql_format_options,
        )


class NearSQLTable(NearSQLNamedEntity):
    def __init__(self, *, terms, table_name, quoted_table_name):
        NearSQLNamedEntity.__init__(
            self,
            terms=terms,
            query_name=table_name,
            quoted_query_name=quoted_table_name,
        )
        self.table_name = table_name
        self.quoted_table_name = quoted_table_name

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqltable_to_sql_str_list_(
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
        query_name,
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
        if suffix is not None:
            assert all([isinstance(v, str) for v in suffix])
        NearSQL.__init__(
            self,
            terms=terms,
            query_name=query_name,
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

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlunary_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            constants=constants,
            sql_format_options=sql_format_options,
        )

    def to_with_form(self) -> SQLWithList:
        if self.sub_sql.near_sql.is_table:
            # table references don't need to be re-encoded
            return SQLWithList(last_step=self, previous_steps=[])
        # non-trivial sequence
        stub, sequence = self.sub_sql.to_with_form_stub()
        stubbed_step = NearSQLUnaryStep(
            terms=self.terms,
            query_name=self.query_name,
            quoted_query_name=self.quoted_query_name,
            sub_sql=stub,
            suffix=self.suffix,
            annotation=self.annotation,
        )
        return SQLWithList(last_step=stubbed_step, previous_steps=sequence)


class NearSQLBinaryStep(NearSQL):
    def __init__(
        self,
        *,
        terms,
        query_name,
        quoted_query_name,
        sub_sql1,
        joiner,
        sub_sql2,
        suffix=None,
        annotation=None,
    ):
        assert isinstance(sub_sql1, NearSQLContainer)
        assert isinstance(sub_sql2, NearSQLContainer)
        assert isinstance(suffix, (list, type(None)))
        if suffix is not None:
            assert all([isinstance(v, str) for v in suffix])
        assert isinstance(joiner, str)
        NearSQL.__init__(
            self,
            terms=terms,
            query_name=query_name,
            quoted_query_name=quoted_query_name,
            annotation=annotation,
        )
        self.sub_sql1 = sub_sql1
        self.joiner = joiner
        self.sub_sql2 = sub_sql2
        self.suffix = suffix

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlbinary_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            constants=constants,
            sql_format_options=sql_format_options,
            quoted_query_name=self.quoted_query_name,
        )

    def to_with_form(self) -> SQLWithList:
        if self.sub_sql1.near_sql.is_table and self.sub_sql2.near_sql.is_table:
            # tables references don't need to be re-encoded
            return SQLWithList(last_step=self, previous_steps=[])
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
            if nmi not in seen:
                seen.add(nmi)
                sequence.append(stepi)
        stubbed_step = NearSQLBinaryStep(
            terms=self.terms,
            query_name=self.query_name,
            quoted_query_name=self.quoted_query_name,
            sub_sql1=stub1,
            joiner=self.joiner,
            sub_sql2=stub2,
            suffix=self.suffix,
            annotation=self.annotation,
        )
        return SQLWithList(last_step=stubbed_step, previous_steps=sequence)


class NearSQLRawQStep(NearSQL):
    def __init__(
        self,
        *,
        prefix: List[str],
        query_name: str,
        quoted_query_name: str,
        sub_sql: Optional[NearSQLContainer],
        suffix: Optional[List[str]] = None,
        annotation: Optional[str] = None,
        add_select: bool = True,
    ):
        assert isinstance(add_select, bool)
        assert isinstance(prefix, list)
        assert len(prefix) > 0
        assert all([isinstance(v, str) for v in prefix])
        assert isinstance(query_name, str)
        assert isinstance(quoted_query_name, str)
        assert isinstance(sub_sql, (NearSQLContainer, type(None)))
        assert isinstance(suffix, (list, type(None)))
        if suffix is not None:
            assert all([isinstance(v, str) for v in suffix])
        assert isinstance(annotation, (str, type(None)))
        NearSQL.__init__(
            self,
            terms=None,  # not using this in this node
            query_name=query_name,
            quoted_query_name=quoted_query_name,
            annotation=annotation,
        )
        self.prefix = prefix
        self.sub_sql = sub_sql
        self.suffix = suffix
        self.add_select = add_select

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        constants=None,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlrawq_to_sql_str_list_(
            near_sql=self,
            sql_format_options=sql_format_options,
            add_select=self.add_select,
        )

    def to_with_form(self) -> SQLWithList:
        if self.sub_sql is None:
            # no sub-steps
            return SQLWithList(last_step=self, previous_steps=[])
        if self.sub_sql.near_sql.is_table:
            # table references don't need to be re-encoded
            return SQLWithList(last_step=self, previous_steps=[])
        # non-trivial sequence
        stub, sequence = self.sub_sql.to_with_form_stub()
        stubbed_step = NearSQLRawQStep(
            prefix=self.prefix,
            query_name=self.query_name,
            quoted_query_name=self.quoted_query_name,
            sub_sql=stub,
            suffix=self.suffix,
            annotation=self.annotation,
            add_select=self.add_select,
        )
        return SQLWithList(last_step=stubbed_step, previous_steps=sequence)
