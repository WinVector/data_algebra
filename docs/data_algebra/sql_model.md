Module data_algebra.sql_model
=============================
Base class for SQL adapters for data algebra.

Classes
-------

`SQLModel(*, identifier_quote: str = '"', string_quote: str = "'", sql_formatters=None, op_replacements=None, on_start: str = '', on_end: str = '', on_joiner: str = 'AND', drop_text: str = 'DROP TABLE', string_type: str = 'VARCHAR', float_type: str = 'FLOAT64', supports_with: bool = True, supports_cte_elim: bool = True, allow_extend_merges: bool = True, default_SQL_format_options=None, union_all_term_start: str = '(', union_all_term_end: str = ')')`
:   A model of how SQL should be generated for a given database.

    ### Descendants

    * data_algebra.PolarsSQL.PolarsSQLModel
    * data_algebra.db_model.DBModel

    ### Class variables

    `allow_extend_merges: bool`
    :

    `default_SQL_format_options: data_algebra.sql_format_options.SQLFormatOptions`
    :

    `drop_text: str`
    :

    `identifier_quote: str`
    :

    `known_methods: Optional[Set[data_algebra.data_ops_types.MethodUse]]`
    :

    `on_end: str`
    :

    `on_joiner: str`
    :

    `on_start: str`
    :

    `recommended_methods: Optional[Set[data_algebra.data_ops_types.MethodUse]]`
    :

    `string_quote: str`
    :

    `string_type: str`
    :

    `supports_cte_elim: bool`
    :

    `supports_with: bool`
    :

    `union_all_term_end: str`
    :

    `union_all_term_start: str`
    :

    ### Methods

    `blocks_to_row_recs_query_str_list_pair(self, record_spec) ‑> Tuple[List[str], List[str]]`
    :   Convert blocks to row recs transform into structures to help with SQL translation.

    `concat_rows_to_near_sql(self, concat_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert concat rows into NearSQL.

    `drop_columns_to_near_sql(self, drop_columns_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert drop columns to NearSQL

    `enc_term_(self, k, *, terms) ‑> str`
    :   encode and name a term for use in a SQL expression

    `expr_to_sql(self, expression, *, want_inline_parens: bool = False) ‑> str`
    :   Convert an expression to SQL.

    `extend_to_near_sql(self, extend_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert an extend step into NearSQL.

    `map_columns_to_near_sql(self, map_columns_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert map columns columns to NearSQL.

    `natural_join_to_near_sql(self, join_node, *, using=None, temp_id_source=None, sql_format_options=None, left_is_first=True) ‑> data_algebra.near_sql.NearSQL`
    :   Convert natural join into NearSQL.

    `nearsqlbinary_to_sql_str_list_(self, near_sql, *, columns=None, force_sql=False, sql_format_options=None, quoted_query_name=None) ‑> List[str]`
    :   Convert SQL binary operation to list of SQL string lines.

    `nearsqlcte_to_sql_str_list_(self, near_sql, *, columns=None, force_sql=False, sql_format_options=None) ‑> List[str]`
    :   Convert SQL common table expression to list of SQL string lines.

    `nearsqlrawq_to_sql_str_list_(self, near_sql, *, sql_format_options=None, add_select=True) ‑> List[str]`
    :   Convert user SQL query to list of SQL string lines.

    `nearsqltable_to_sql_str_list_(self, near_sql, *, columns=None, force_sql=False, sql_format_options=None) ‑> List[str]`
    :   Convert SQL table description to list of SQL string lines.

    `nearsqlunary_to_sql_str_list_(self, near_sql, *, columns=None, force_sql=False, sql_format_options=None) ‑> List[str]`
    :   Convert SQL unary operation to list of SQL string lines.

    `non_known_methods(self, ops: data_algebra.data_ops.ViewRepresentation) ‑> List[data_algebra.data_ops_types.MethodUse]`
    :   Return list of used non-recommended methods.

    `non_recommended_methods(self, ops: data_algebra.data_ops.ViewRepresentation) ‑> List[data_algebra.data_ops_types.MethodUse]`
    :   Return list of used non-recommended methods.

    `order_to_near_sql(self, order_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert order rows to NearSQL.

    `project_to_near_sql(self, project_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert a project step to NearSQL

    `quote_identifier(self, identifier: str) ‑> str`
    :   Quote identifier.

    `quote_string(self, string: str) ‑> str`
    :   Quote a string value.

    `quote_table_name(self, table_description) ‑> str`
    :   Quote a table name.

    `rename_to_near_sql(self, rename_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert rename columns to NearSQL.

    `row_recs_to_blocks_query_str_list_pair(self, record_spec) ‑> Tuple[List[str], List[str]]`
    :   Convert row recs to blocks transformation into structures to help with SQL conversion.

    `select_columns_to_near_sql(self, select_columns_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert select columns to NearSQL.

    `select_rows_to_near_sql(self, select_rows_node, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert select rows into NearSQL

    `table_def_to_near_sql(self, table_def, *, using=None, temp_id_source=None, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert a table description to NearSQL.

    `table_values_to_sql_str_list(self, v, *, result_name: str = 'table_values') ‑> List[str]`
    :   Convert a table of values to a SQL. Only for small tables.

    `to_sql(self, ops: data_algebra.data_ops.ViewRepresentation, *, sql_format_options: Optional[data_algebra.sql_format_options.SQLFormatOptions] = None) ‑> str`
    :   Convert ViewRepresentation into SQL string.
        
        :param ops: ViewRepresentation to convert
        :param sql_format_options: sql formatting options
        :return: sql string

    `value_to_sql(self, v) ‑> str`
    :   Convert a value to valid SQL.