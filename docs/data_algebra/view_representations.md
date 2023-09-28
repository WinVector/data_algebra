Module data_algebra.view_representations
========================================
Realization of data operation classes.

Classes
-------

`ConcatRowsNode(a, b, *, id_column='table_name', a_name='a', b_name='b')`
:   Class representation of .concat_rows() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `id_column: Optional[str]`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`ConvertRecordsNode(*, source, record_map)`
:   Class representation of .convert_records() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`DropColumnsNode(source, column_deletions)`
:   Class representation of .drop_columns() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `column_deletions: List[str]`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`ExtendNode(*, source, parsed_ops, partition_by=None, order_by=None, reverse=None)`
:   Class representation of .extend() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `ordered_windowed_situation: bool`
    :

    `partition_by: List[str]`
    :

    `windowed_situation: bool`
    :

    ### Methods

    `check_extend_window_fns_(self)`
    :   Confirm extend functions are all compatible with windowing in Pandas. Internal function.

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`MapColumnsNode(source, column_remapping)`
:   Class representation of .map_columns() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `column_deletions: List[str]`
    :

    `column_remapping: Dict[str, str]`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`NaturalJoinNode(a, b, *, on_a: List[str], on_b: List[str], jointype: str, check_all_common_keys_in_equi_spec: bool = False)`
:   Class representation of .natural_join() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `jointype: str`
    :

    `on_a: List[str]`
    :

    `on_b: List[str]`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`OrderRowsNode(source, columns, *, reverse=None, limit=None)`
:   Class representation of .order_rows() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `order_columns: List[str]`
    :

    `reverse: List[str]`
    :

    ### Methods

    `is_trivial_when_intermediate_(self) ‑> bool`
    :   Return if True if operator can be eliminated from interior of chain.

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`ProjectNode(*, source, parsed_ops, group_by=None)`
:   Class representation of .project() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`RenameColumnsNode(source, column_remapping)`
:   Class representation of .rename_columns() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `column_remapping: Dict[str, str]`
    :

    `reverse_mapping: Dict[str, str]`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`SQLNode(*, sql: Union[str, List[str]], column_names: List[str], view_name: str)`
:   Class representation of user SQL step in pipeline. Can be used to start a pipeline instead of a TableDescription.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`SelectColumnsNode(source, columns)`
:   Class representation of .select_columns() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `column_selection: List[str]`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`SelectRowsNode(source, ops)`
:   Class representation of .select() method/step.

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `decision_columns: Set[str]`
    :

    `expr: data_algebra.expr_rep.Expression`
    :

    ### Methods

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`TableDescription(*, table_name: Optional[str] = None, column_names: Iterable[str], qualifiers=None, sql_meta=None, head=None, limit_was: Optional[int] = None, nrows: Optional[int] = None)`
:   Describe columns, and qualifiers, of a table.
    
    Example:
        from data_algebra.view_representation import TableDescription
        d = TableDescription(table_name='d', column_names=['x', 'y'])
        print(d)

    ### Ancestors (in MRO)

    * data_algebra.view_representations.ViewRepresentation
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Class variables

    `qualifiers: Dict[str, str]`
    :

    `table_name: str`
    :

    `table_name_was_set_by_user: bool`
    :

    ### Methods

    `get_tables(self)`
    :   get a dictionary of all tables used in an operator DAG,
        raise an exception if the values are not consistent

    `replace_leaves(self, replacement_map: Dict[str, Any])`
    :   Replace leaves of DAG
        
        :param a: operators to apply to
         :param replacement_map, table/sqlkeys mapped to replacement Operator platforms
        :return: new operator DAG

    `same_table_description_(self, other)`
    :   Return true if other is a description of the same table. Internal method, ingores data.

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True)`
    :   Return text representing operations.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

`ViewRepresentation(column_names: Iterable[str], *, sources: Optional[Iterable[ForwardRef('ViewRepresentation')]] = None, node_name: str, key: Optional[str] = None)`
:   Structure to represent the columns of a query or a table.
    Abstract base class.

    ### Ancestors (in MRO)

    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Descendants

    * data_algebra.view_representations.ConcatRowsNode
    * data_algebra.view_representations.ConvertRecordsNode
    * data_algebra.view_representations.DropColumnsNode
    * data_algebra.view_representations.ExtendNode
    * data_algebra.view_representations.MapColumnsNode
    * data_algebra.view_representations.NaturalJoinNode
    * data_algebra.view_representations.OrderRowsNode
    * data_algebra.view_representations.ProjectNode
    * data_algebra.view_representations.RenameColumnsNode
    * data_algebra.view_representations.SQLNode
    * data_algebra.view_representations.SelectColumnsNode
    * data_algebra.view_representations.SelectRowsNode
    * data_algebra.view_representations.TableDescription

    ### Class variables

    `column_names: Tuple[str, ...]`
    :

    `key: Optional[str]`
    :

    `sources: Tuple[data_algebra.view_representations.ViewRepresentation, ...]`
    :

    ### Methods

    `act_on(self, b, *, correct_ordered_first_call: bool = False)`
    :   apply self to b, must associate with composition
        Operator is strict about column names.
        
        :param b: input data frame
        :param correct_ordered_first_call: indicate not on fallback path
        :return: transformed or composed result

    `as_table_description(self, table_name: str, *, qualifiers=None)`
    :   Return representation of operator as a table description.
        
        :param table_name: table name to use.
        :param qualifiers: db qualifiers to annotate

    `check_constraints(self, data_map, *, strict: bool = True)`
    :   Check tables supplied meet data consistency constraints.
        
        data_map: dictionary of column name lists.

    `cod(self, *, table_name: Optional[str] = None)`
    :   Description of operator co-domain, a table description.
        
        :param table_name: optional name for table
        :return: TableDescription representing produced columns.

    `column_map(self) ‑> collections.OrderedDict`
    :   Build a map of column names to ColumnReferences

    `columns_produced(self) ‑> List[str]`
    :   Return list of columns produced by operator dag.

    `columns_used(self, *, using=None) ‑> Dict`
    :   Determine which columns are used from source tables.

    `columns_used_from_sources(self, using: Optional[set] = None) ‑> List[str]`
    :   Get columns used from sources. Internal method.
        
        :param using: optional column restriction.
        :return: list of order sets (list parallel to sources).

    `columns_used_implementation_(self, *, using, columns_currently_using_records) ‑> None`
    :   Implementation of columns used calculation, internal method.

    `dom(self)`
    :   Description of domain.
        
        :return: map of tables names to table descriptions

    `extend(self, ops, *, partition_by=None, order_by=None, reverse=None) ‑> data_algebra.view_representations.ViewRepresentation`
    :   Add new derived columns, can replace existing columns.
        
        :param ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification, or 1.
        :param order_by: optional window ordering specification, or 1.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph

    `extend_parsed_(self, parsed_ops, *, partition_by=None, order_by=None, reverse=None) ‑> data_algebra.view_representations.ViewRepresentation`
    :   Add new derived columns, can replace existing columns for parsed operations. Internal method.
        
        :param parsed_ops: dictionary of calculations to perform.
        :param partition_by: optional window partition specification, or 1.
        :param order_by: optional window ordering specification, or 1.
        :param reverse: optional order reversal specification.
        :return: compose operator directed acyclic graph

    `get_method_uses_(self, methods_seen: Set[data_algebra.data_ops_types.MethodUse]) ‑> None`
    :   Implementation of get methods_used(), internal method.
        
        :params methods_seen: set to collect results in.
        :return: None

    `is_trivial_when_intermediate_(self) ‑> bool`
    :   Return if True if operator can be eliminated from interior chain.

    `merged_rep_id(self) ‑> str`
    :   String key for lookups.

    `natural_join(self, b, *, on=None, jointype: str, check_all_common_keys_in_equi_spec: bool = False, by=None, check_all_common_keys_in_by: bool = False) ‑> data_algebra.view_representations.ViewRepresentation`
    :   Join self (left) results with b (right).
        
        :param b: second or right table to join to.
        :param on: column names to enforce equality on (list of column names, list of tuples, or dictionary)
        :param jointype: name of join type.
        :param check_all_common_keys_in_equi_spec: if True, raise if any non-equality key columns are common to tables.
        :param by: synonym for on, only set at most one of on or by (deprecated).
        :param check_all_common_keys_in_by: synonym for check_all_common_keys_in_equi_spec (deprecated).
        :return: compose operator directed acyclic graph

    `to_near_sql_implementation_(self, db_model, *, using, temp_id_source, sql_format_options=None) ‑> data_algebra.near_sql.NearSQL`
    :   Convert operator dag into NearSQL type for translation to SQL string.
        
        :param db_model: database model
        :param using: optional column restriction set
        :param temp_id_source: source of temporary ids
        :param sql_format_options: options for sql formatting
        :return: data_algebra.near_sql.NearSQL

    `to_python(self, *, indent=0, strict=True, pretty=False, black_mode=None) ‑> str`
    :   Return Python source code for operations.
        
        :param indent: extra indent.
        :param strict: if False allow eliding of columns names and other long structures.
        :param pretty: if True re-format result with black.
        :param black_mode: black formatter parameters.

    `to_python_src_(self, *, indent=0, strict=True, print_sources=True) ‑> str`
    :   Return text representing operations. Internal method, allows skipping of sources.
        
        :param indent: additional indent to apply in formatting.
        :param strict: if False allow eliding of columns names and other long structures.
        :param print_sources: logical, print children.

    `to_sql(self, db_model=None, *, sql_format_options=None) ‑> str`
    :   Convert operator dag to SQL.
        
        :param db_model: SQLModel, DBModel, or DBHandle
        :param sql_format_options: options for sql formatting
        :return: string representation of SQL query

    `transform(self, X, *, data_model=None, strict: bool = False)`
    :   Apply data transform to a table
        
        :param X: tale to apply to
        :param data_model: data model for Pandas execution
        :param strict: if True, throw on unexpected columns
        :return: transformed data frame