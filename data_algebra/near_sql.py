"""
Representation for operations that are nearly translated into SQL.
"""

import abc
from typing import Dict, Iterable, List, Optional, Tuple

import data_algebra.OrderedSet


# classes for holding object for SQL generation


class SQLWithList:
    """
    Carry an ordered sequence of SQL steps for use with a SQL WITH statement.
    """

    last_step: "NearSQL"
    previous_steps: List[Tuple[str, "NearSQLContainer"]]

    def __init__(
        self,
        *,
        last_step: "NearSQL",
        previous_steps: Iterable[Tuple[str, "NearSQLContainer"]],
    ):
        assert isinstance(last_step, NearSQL)
        previous_steps = list(previous_steps)
        assert isinstance(previous_steps, List)
        for vi in previous_steps:
            assert isinstance(vi, tuple)
            assert len(vi) == 2
            assert isinstance(vi[0], str)
            assert isinstance(vi[1], NearSQLContainer)
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


class NearSQL(abc.ABC):
    """
    Represent SQL queries in a mostly string-form
    """

    terms: Optional[Dict[str, Optional[str]]]
    query_name: str
    quoted_query_name: str
    is_table: bool
    ops_key: Optional[str]
    annotation: Optional[str]

    def __init__(
        self,
        *,
        terms: Optional[Dict[str, Optional[str]]],
        query_name: str,
        quoted_query_name: str,
        is_table: bool = False,
        annotation: Optional[str] = None,
        ops_key: Optional[str],  # key for sub-ops, used to eliminate repeated sub-trees
    ):
        assert isinstance(terms, (dict, type(None)))
        assert isinstance(query_name, str)
        assert isinstance(quoted_query_name, str)
        assert isinstance(is_table, bool)
        assert isinstance(annotation, (str, type(None)))
        if ops_key is not None:
            assert isinstance(ops_key, str)
        self.terms = None
        if terms is not None:
            assert isinstance(terms, dict)
            if len(terms) > 0:
                self.terms = terms.copy()
        self.query_name = query_name
        self.quoted_query_name = quoted_query_name
        self.is_table = is_table
        self.annotation = annotation
        self.ops_key = ops_key

    def to_bound_near_sql(
        self,
        *,
        columns=None,
        force_sql: bool = False,
        public_name: Optional[str] = None,
        public_name_quoted: Optional[str] = None,
    ) -> "NearSQLContainer":
        return NearSQLContainer(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            public_name=public_name,
            public_name_quoted=public_name_quoted,
        )

    @abc.abstractmethod
    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        """export"""

    @abc.abstractmethod
    def to_with_form(self, *, cte_cache: Optional[Dict]) -> SQLWithList:
        """convert ot with form"""


class NearSQLContainer:
    """
    NearSQL with bound in columns, force_sql
    """

    near_sql: NearSQL
    force_sql: bool
    columns: Optional[data_algebra.OrderedSet.OrderedSet]
    public_name: Optional[str]
    public_name_quoted: Optional[str]

    def __init__(
        self,
        *,
        near_sql: NearSQL,
        columns: Optional[Iterable[str]] = None,
        force_sql: bool = False,
        public_name: Optional[str] = None,
        public_name_quoted: Optional[str] = None,
    ):
        assert isinstance(near_sql, NearSQL)
        assert isinstance(force_sql, bool)
        self.near_sql = near_sql
        self.columns = None
        if columns is not None:
            assert not isinstance(columns, str)
            self.columns = data_algebra.OrderedSet.OrderedSet(columns)
        self.force_sql = force_sql
        assert (public_name is not None) == (public_name_quoted is not None)
        if (public_name is not None) or (public_name_quoted is not None):
            assert isinstance(public_name, str)
            assert isinstance(public_name_quoted, str)
        self.public_name = public_name
        self.public_name_quoted = public_name_quoted

    def convert_subsql(
        self,
        *,
        db_model,
        sql_format_options=None,
        quoted_query_name_annotation: Optional[str] = None,
    ) -> List[str]:
        """Convert subsql, possibly adding query name"""
        non_trivial_annotation = (quoted_query_name_annotation is not None) and (
            quoted_query_name_annotation != self.near_sql.quoted_query_name
        )
        if isinstance(self.near_sql, data_algebra.near_sql.NearSQLNamedEntity) and (
            not self.force_sql
        ):
            # short circuit table and cte path
            if non_trivial_annotation:
                assert quoted_query_name_annotation is not None  # type hint
                assert isinstance(quoted_query_name_annotation, str)  # type hint
                sql = [
                    self.near_sql.quoted_query_name + " " + quoted_query_name_annotation
                ]  # table name and alias
            else:
                sql = [self.near_sql.quoted_query_name]
        else:
            # main path
            sql = self.near_sql.to_sql_str_list(
                columns=self.columns,
                force_sql=self.force_sql or non_trivial_annotation,
                db_model=db_model,
                sql_format_options=sql_format_options,
            )
            if quoted_query_name_annotation is not None:
                sql = ["("] + sql + [") " + quoted_query_name_annotation]
        return sql

    # sequence: a list where last element is a NearSQLContainer previous elements are (name, NearSQLContainer) pairs
    # stub the replacement common table expression in a NearSQLContainer
    def to_with_form_stub(
        self, *, cte_cache: Optional[Dict]
    ) -> Tuple["NearSQLContainer", List[Tuple[str, "NearSQLContainer"]]]:
        if self.near_sql.is_table:  # table or common table expression
            return self, []
        in_with_form = self.near_sql.to_with_form(cte_cache=cte_cache)
        sequence = in_with_form.previous_steps
        stub = in_with_form.last_step
        assert isinstance(stub, NearSQL)
        assert isinstance(sequence, List)
        for v in sequence:
            assert isinstance(v, tuple)
            assert len(v) == 2
            assert isinstance(v[0], str)
            assert isinstance(v[1], NearSQLContainer)
        if not stub.is_table:
            # first check cache
            ops_key = f"{self.near_sql.ops_key}"
            if self.columns is not None:
                ops_key = f"{ops_key}_{list(self.columns)}"
            if (cte_cache is not None) and (ops_key is not None):
                try:
                    retrieved_cte = cte_cache[ops_key]
                    # copy in context fields
                    new_stub = NearSQLContainer(
                        near_sql=retrieved_cte,
                        columns=self.columns,
                        force_sql=self.force_sql,
                        public_name=self.public_name,
                        public_name_quoted=self.public_name_quoted,
                    )
                    return new_stub, []
                except KeyError:
                    pass
            # replace step with a reference
            if stub.quoted_query_name not in {k for k, v in sequence}:
                sequence.append(
                    (
                        stub.quoted_query_name,
                        NearSQLContainer(
                            near_sql=stub,
                            force_sql=self.force_sql,
                            columns=self.columns,
                        ),
                    )
                )
            new_stub_cte = NearSQLCommonTableExpression(
                query_name=stub.query_name,
                quoted_query_name=stub.quoted_query_name,
                ops_key=ops_key,
            )
            new_stub = NearSQLContainer(
                near_sql=new_stub_cte,
                force_sql=self.force_sql,
                columns=self.columns,
                public_name=self.public_name,
                public_name_quoted=self.public_name_quoted,
            )
            if (cte_cache is not None) and (ops_key is not None):
                cte_cache[ops_key] = new_stub_cte
        else:
            assert len(sequence) == 0
            new_stub = NearSQLContainer(
                near_sql=stub,
                force_sql=True,
                columns=self.columns,
                public_name=self.public_name,
                public_name_quoted=self.public_name_quoted,
            )
        return new_stub, sequence


class NearSQLNamedEntity(NearSQL, abc.ABC):
    """Model for tables and common table expressions"""

    def __init__(
        self, *, terms, query_name: str, quoted_query_name: str, ops_key: Optional[str]
    ):
        NearSQL.__init__(
            self,
            terms=terms,
            query_name=query_name,
            quoted_query_name=quoted_query_name,
            is_table=True,
            ops_key=ops_key,
        )

    def to_with_form(self, *, cte_cache: Optional[Dict]) -> SQLWithList:
        return SQLWithList(last_step=self, previous_steps=[])

    @abc.abstractmethod
    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        """export"""


class NearSQLCommonTableExpression(NearSQLNamedEntity):
    def __init__(
        self, *, query_name: str, quoted_query_name: str, ops_key: Optional[str]
    ):
        NearSQLNamedEntity.__init__(
            self,
            terms=None,
            query_name=query_name,
            quoted_query_name=quoted_query_name,
            ops_key=ops_key,
        )

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlcte_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            sql_format_options=sql_format_options,
        )


class NearSQLTable(NearSQLNamedEntity):
    def __init__(self, *, terms, table_name: str, quoted_table_name: str):
        NearSQLNamedEntity.__init__(
            self,
            terms=terms,
            query_name=table_name,
            quoted_query_name=quoted_table_name,
            ops_key=table_name,
        )
        self.table_name = table_name
        self.quoted_table_name = quoted_table_name

    def to_sql_str_list(
        self,
        *,
        columns=None,
        force_sql=False,
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqltable_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            sql_format_options=sql_format_options,
        )


class NearSQLUnaryStep(NearSQL):

    sub_sql: NearSQLContainer
    mergeable: bool
    suffix: Optional[List]
    declared_term_dependencies: Optional[Dict]

    def __init__(
        self,
        *,
        terms,
        query_name: str,
        quoted_query_name: str,
        sub_sql,
        suffix=None,
        annotation=None,
        ops_key: Optional[str],
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
            ops_key=ops_key,
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
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlunary_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            sql_format_options=sql_format_options,
        )

    def to_with_form(self, *, cte_cache: Optional[Dict]) -> SQLWithList:
        if self.sub_sql.near_sql.is_table:
            # table references and common table expressions don't need to be re-encoded
            return SQLWithList(last_step=self, previous_steps=[])
        # non-trivial sequence
        stub, sequence = self.sub_sql.to_with_form_stub(cte_cache=cte_cache)
        stubbed_step = NearSQLUnaryStep(
            terms=self.terms,
            query_name=self.query_name,
            quoted_query_name=self.quoted_query_name,
            sub_sql=stub,
            suffix=self.suffix,
            annotation=self.annotation,
            ops_key=self.ops_key,
        )
        return SQLWithList(last_step=stubbed_step, previous_steps=sequence)


class NearSQLBinaryStep(NearSQL):

    sub_sql1: NearSQLContainer
    sub_sql2: NearSQLContainer
    suffix: Optional[List]
    joiner: str

    def __init__(
        self,
        *,
        terms,
        query_name: str,
        quoted_query_name: str,
        sub_sql1: NearSQLContainer,
        joiner: str,
        sub_sql2: NearSQLContainer,
        suffix=None,
        annotation=None,
        ops_key: Optional[str],
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
            ops_key=ops_key,
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
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlbinary_to_sql_str_list_(
            near_sql=self,
            columns=columns,
            force_sql=force_sql,
            sql_format_options=sql_format_options,
            quoted_query_name=self.quoted_query_name,
        )

    def to_with_form(self, *, cte_cache: Optional[Dict]) -> SQLWithList:
        if (
            self.sub_sql1.near_sql.is_table and self.sub_sql2.near_sql.is_table
        ):  # TODO: remove this?
            # tables references don't need to be re-encoded
            return SQLWithList(last_step=self, previous_steps=[])
        # non-trivial sequence
        stub1, sequence1 = self.sub_sql1.to_with_form_stub(cte_cache=cte_cache)
        stub2, sequence2 = self.sub_sql2.to_with_form_stub(cte_cache=cte_cache)
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
            ops_key=self.ops_key,
        )
        return SQLWithList(last_step=stubbed_step, previous_steps=sequence)


class NearSQLRawQStep(NearSQL):

    prefix: List
    sub_sql: Optional[NearSQLContainer]
    suffix: Optional[List]
    add_select: bool

    def __init__(
        self,
        *,
        prefix: List[str],
        query_name: str,
        quoted_query_name: str,
        sub_sql: Optional[NearSQLContainer],
        suffix: Optional[List[str]] = None,
        annotation: Optional[str] = None,
        ops_key: Optional[str],
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
            ops_key=ops_key,
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
        db_model,
        sql_format_options=None,
    ) -> List[str]:
        return db_model.nearsqlrawq_to_sql_str_list_(
            near_sql=self,
            sql_format_options=sql_format_options,
            add_select=self.add_select,
        )

    def to_with_form(self, *, cte_cache: Optional[Dict]) -> SQLWithList:
        if self.sub_sql is None:
            # no sub-steps
            return SQLWithList(last_step=self, previous_steps=[])
        if self.sub_sql.near_sql.is_table:  # TODO: do we want this?
            # table references don't need to be re-encoded
            return SQLWithList(last_step=self, previous_steps=[])
        # non-trivial sequence
        stub, sequence = self.sub_sql.to_with_form_stub(cte_cache=cte_cache)
        assert isinstance(self.query_name, str)  # type hint
        stubbed_step = NearSQLRawQStep(
            prefix=self.prefix,
            query_name=self.query_name,
            quoted_query_name=self.quoted_query_name,
            sub_sql=stub,
            suffix=self.suffix,
            annotation=self.annotation,
            add_select=self.add_select,
            ops_key=self.ops_key,
        )
        return SQLWithList(last_step=stubbed_step, previous_steps=sequence)
