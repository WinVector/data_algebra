Module data_algebra.near_sql
============================
Representation for operations that are nearly translated into SQL.

Classes
-------

`NearSQL(*, terms: Optional[Dict[str, Optional[str]]], query_name: str, quoted_query_name: str, is_table: bool = False, annotation: Optional[str] = None, ops_key: Optional[str])`
:   Represent SQL queries in a mostly string-form

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.near_sql.NearSQLBinaryStep
    * data_algebra.near_sql.NearSQLNamedEntity
    * data_algebra.near_sql.NearSQLRawQStep
    * data_algebra.near_sql.NearSQLUnaryStep

    ### Class variables

    `annotation: Optional[str]`
    :

    `is_table: bool`
    :

    `ops_key: Optional[str]`
    :

    `query_name: str`
    :

    `quoted_query_name: str`
    :

    `terms: Optional[Dict[str, Optional[str]]]`
    :

    ### Methods

    `to_bound_near_sql(self, *, columns=None, force_sql: bool = False, public_name: Optional[str] = None, public_name_quoted: Optional[str] = None) ‑> data_algebra.near_sql.NearSQLContainer`
    :

    `to_sql_str_list(self, *, columns=None, force_sql=False, db_model, sql_format_options=None) ‑> List[str]`
    :   export

    `to_with_form(self, *, cte_cache: Optional[Dict]) ‑> data_algebra.near_sql.SQLWithList`
    :   convert ot with form

`NearSQLBinaryStep(*, terms, query_name: str, quoted_query_name: str, sub_sql1: data_algebra.near_sql.NearSQLContainer, joiner: str, sub_sql2: data_algebra.near_sql.NearSQLContainer, suffix=None, annotation=None, ops_key: Optional[str])`
:   Represent SQL queries in a mostly string-form

    ### Ancestors (in MRO)

    * data_algebra.near_sql.NearSQL
    * abc.ABC

    ### Class variables

    `joiner: str`
    :

    `sub_sql1: data_algebra.near_sql.NearSQLContainer`
    :

    `sub_sql2: data_algebra.near_sql.NearSQLContainer`
    :

    `suffix: Optional[List]`
    :

`NearSQLCommonTableExpression(*, query_name: str, quoted_query_name: str, ops_key: Optional[str])`
:   Model for tables and common table expressions

    ### Ancestors (in MRO)

    * data_algebra.near_sql.NearSQLNamedEntity
    * data_algebra.near_sql.NearSQL
    * abc.ABC

`NearSQLContainer(*, near_sql: data_algebra.near_sql.NearSQL, columns: Optional[Iterable[str]] = None, force_sql: bool = False, public_name: Optional[str] = None, public_name_quoted: Optional[str] = None)`
:   NearSQL with bound in columns, force_sql

    ### Class variables

    `columns: Optional[data_algebra.OrderedSet.OrderedSet]`
    :

    `force_sql: bool`
    :

    `near_sql: data_algebra.near_sql.NearSQL`
    :

    `public_name: Optional[str]`
    :

    `public_name_quoted: Optional[str]`
    :

    ### Methods

    `convert_subsql(self, *, db_model, sql_format_options=None, quoted_query_name_annotation: Optional[str] = None) ‑> List[str]`
    :   Convert subsql, possibly adding query name

    `to_with_form_stub(self, *, cte_cache: Optional[Dict]) ‑> Tuple[data_algebra.near_sql.NearSQLContainer, List[Tuple[str, data_algebra.near_sql.NearSQLContainer]]]`
    :

`NearSQLNamedEntity(*, terms, query_name: str, quoted_query_name: str, ops_key: Optional[str])`
:   Model for tables and common table expressions

    ### Ancestors (in MRO)

    * data_algebra.near_sql.NearSQL
    * abc.ABC

    ### Descendants

    * data_algebra.near_sql.NearSQLCommonTableExpression
    * data_algebra.near_sql.NearSQLTable

`NearSQLRawQStep(*, prefix: List[str], query_name: str, quoted_query_name: str, sub_sql: Optional[data_algebra.near_sql.NearSQLContainer], suffix: Optional[List[str]] = None, annotation: Optional[str] = None, ops_key: Optional[str], add_select: bool = True)`
:   Represent SQL queries in a mostly string-form

    ### Ancestors (in MRO)

    * data_algebra.near_sql.NearSQL
    * abc.ABC

    ### Class variables

    `add_select: bool`
    :

    `prefix: List`
    :

    `sub_sql: Optional[data_algebra.near_sql.NearSQLContainer]`
    :

    `suffix: Optional[List]`
    :

`NearSQLTable(*, terms, table_name: str, quoted_table_name: str)`
:   Model for tables and common table expressions

    ### Ancestors (in MRO)

    * data_algebra.near_sql.NearSQLNamedEntity
    * data_algebra.near_sql.NearSQL
    * abc.ABC

`NearSQLUnaryStep(*, terms, query_name: str, quoted_query_name: str, sub_sql, suffix=None, annotation=None, ops_key: Optional[str], mergeable=False, declared_term_dependencies=None)`
:   Represent SQL queries in a mostly string-form

    ### Ancestors (in MRO)

    * data_algebra.near_sql.NearSQL
    * abc.ABC

    ### Class variables

    `declared_term_dependencies: Optional[Dict]`
    :

    `mergeable: bool`
    :

    `sub_sql: data_algebra.near_sql.NearSQLContainer`
    :

    `suffix: Optional[List]`
    :

`SQLWithList(*, last_step: NearSQL, previous_steps: Iterable[Tuple[str, ForwardRef('NearSQLContainer')]])`
:   Carry an ordered sequence of SQL steps for use with a SQL WITH statement.

    ### Class variables

    `last_step: data_algebra.near_sql.NearSQL`
    :

    `previous_steps: List[Tuple[str, data_algebra.near_sql.NearSQLContainer]]`
    :