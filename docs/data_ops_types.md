Module data_algebra.data_ops_types
==================================
Type defs for data operations.

Classes
-------

`MethodUse(op_name: str, is_project: bool = False, is_windowed: bool = False, is_ordered: bool = False)`
:   Carry description of a method use

    ### Ancestors (in MRO)

    * builtins.tuple

    ### Instance variables

    `is_ordered: bool`
    :   Alias for field number 3

    `is_project: bool`
    :   Alias for field number 1

    `is_windowed: bool`
    :   Alias for field number 2

    `op_name: str`
    :   Alias for field number 0

`OperatorPlatform(*, node_name: str)`
:   Abstract class representing ability to apply data_algebra operations.

    ### Ancestors (in MRO)

    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Descendants

    * data_algebra.view_representations.ViewRepresentation

    ### Class variables

    `node_name: str`
    :

    ### Methods

    `columns_produced(self) ‑> List[str]`
    :   Return list of columns produced by pipeline.

    `concat_rows(self, b, *, id_column='source_name', a_name='a', b_name='b')`
    :   Union or concatenate rows of self with rows of b.
        
        :param b: table with rows to add.
        :param id_column: optional name for new source identification column.
        :param a_name: source annotation to use for self/a.
        :param b_name: source annotation to use for b.
        :return: compose operator directed acyclic graph

    `convert_records(self, record_map: data_algebra.cdata.RecordMap)`
    :   Apply a record mapping taking blocks_in to blocks_out structures.
        
        :param record_map: data_algebra.cdata.RecordMap transform specification
        :return: compose operator directed acyclic graph

    `drop_columns(self, column_deletions)`
    :   Remove columns from result.
        
        :param column_deletions: list of columns to remove.
        :return: compose operator directed acyclic graph

    `eval(self, data_map: Dict[str, Any], *, data_model=None, strict: bool = False)`
    :   Evaluate operators with respect to Pandas data frames.
        
        :param data_map: map from table names to data frames or data sources
        :param data_model: adaptor to data dialect (Pandas for now)
        :param strict: if True, throw on unexpected columns
        :return: table result

    `ex(self, *, data_model=None, allow_limited_tables=False)`
    :   Evaluate operators with respect to Pandas data frames already stored in the operator chain.
        
        :param data_model: adaptor to data dialect (Pandas for now)
        :param allow_limited_tables: logical, if True allow execution on non-complete tables
        :return: table result

    `extend(self, ops, *, partition_by=None, order_by=None, reverse=None)`
    :   Add new derived columns, can replace existing columns.
        
        :param ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification.
        :param order_by: optional window ordering specification.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph

    `extend_parsed_(self, parsed_ops, *, partition_by=None, order_by=None, reverse=None)`
    :   Add new derived columns, can replace existing columns for parsed operations. Internal method.
        
        :param parsed_ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification.
        :param order_by: optional window ordering specification.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph

    `get_tables(self)`
    :   Get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent.

    `map_columns(self, column_remapping: Dict[str, str])`
    :   Map column names or rename.
        
        :param column_remapping: dictionary mapping old column sources to new column names (same
                                 direction as Pandas rename).
        :return: compose operator directed acyclic graph

    `map_records(self, blocks_in: Optional[data_algebra.cdata.RecordSpecification] = None, blocks_out: Optional[data_algebra.cdata.RecordSpecification] = None)`
    :   Apply a record mapping taking blocks_in to blocks_out structures.
        
        :param blocks_in: Optional incoming record specification
        :param blocks_out: Optional incoming record specification
        :return: compose operator directed acyclic graph

    `methods_used(self) ‑> Set[data_algebra.data_ops_types.MethodUse]`
    :   Return set of methods used.

    `natural_join(self, b, *, on: Optional[Iterable[str]] = None, jointype: str, check_all_common_keys_in_equi_spec: bool = False, by: Optional[Iterable[str]] = None, check_all_common_keys_in_by: bool = False)`
    :   Join self (left) results with b (right).
        
        :param b: second or right table to join to.
        :param on: list of join column names to enforce equality on.
        :param jointype: name of join type.
        :param check_all_common_keys_in_equi_spec: if True, raise if any non-equality key columns are common to tables.
        :param by: synonym for on, only set at most one of on or by (deprecated).
        :param check_all_common_keys_in_by: synonym for check_all_common_keys_in_equi_spec (deprecated).
        :return: compose operator directed acyclic graph

    `order_rows(self, columns, *, reverse=None, limit=None)`
    :   Order rows by column set.
        
        :param columns: columns to order by.
        :param reverse: optional columns to reverse order.
        :param limit: optional row limit to impose on result.
        :return: compose operator directed acyclic graph

    `project(self, ops=None, *, group_by=None)`
    :   Compute projection, or grouped calculation.
        
        :param ops: dictionary of calculations to perform, can be empty.
        :param group_by: optional group key(s) specification.
        :return: compose operator directed acyclic graph

    `project_parsed_(self, parsed_ops=None, *, group_by=None)`
    :   Compute projection, or grouped calculation for parsed ops. Internal method.
        
        :param parsed_ops: dictionary of calculations to perform, can be empty.
        :param group_by: optional group key(s) specification.
        :return: compose operator directed acyclic graph

    `rename_columns(self, column_remapping: Dict[str, str])`
    :   Rename columns.
        
        :param column_remapping: dictionary mapping new column names to old column sources (same
                                 direction as extend).
        :return: compose operator directed acyclic graph

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
        :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `select_columns(self, columns)`
    :   Narrow to columns in result.
        
        :param columns: list of columns to keep.
        :return: compose operator directed acyclic graph

    `select_rows(self, expr)`
    :   Select rows matching expr criteria.
        
        :param expr: logical expression specifying desired rows.
        :return: compose operator directed acyclic graph

    `select_rows_parsed_(self, parsed_expr)`
    :   Select rows matching parsed expr criteria. Internal method.
        
        :param parsed_expr: logical expression specifying desired rows.
        :return: compose operator directed acyclic graph

    `to_near_sql_implementation_(self, db_model, *, using, temp_id_source, sql_format_options=None)`
    :   Convert to NearSQL as a step in converting to a SQL string. Internal method.
        
        :param db_model: database model
        :param using: optional column restriction
        :param temp_id_source: temporary id source.
        :return: compose operator directed acyclic graph

    `transform(self, X, *, data_model=None, strict: bool = False)`
    :   apply self to data frame X, may or may not commute with composition
        
        :param X: input data frame
        :param data_model: implementation to use
        :param strict: if True, throw on unexpected columns
        :return: transformed data frame

    `use(self, user_function, *args, **kwargs)`
    :   Apply f as if it was a method on this chain.
        Defined as return f(self, *args, **kwargs).
        
        :param user_function: function to apply
        :param args: additional positional arguments
        :param kwargs: additional keyword arguments