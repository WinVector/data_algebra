Module data_algebra.polars_model
================================
Adapter to use Polars ( https://www.pola.rs ) in the data algebra.

Note: fully not implemented yet.

Functions
---------

    
`register_polars_model(key: Optional[str] = None)`
:   

Classes
-------

`ExpressionRequirementsCollector()`
:   Class to collect what accommodations an expression needs.

    ### Ancestors (in MRO)

    * data_algebra.expression_walker.ExpressionWalker
    * abc.ABC

    ### Class variables

    `collect_required: bool`
    :

    `one_constant_required: bool`
    :

    `zero_constant_required: bool`
    :

    ### Methods

    `act_on_expression(self, *, arg, values: List, op)`
    :   Action for a column name.
        
        :param arg: None
        :param values: list of values to work on
        :param op: operator to apply
        :return: arg acted on

    `add_in_temp_columns(self, temp_v_columns: List)`
    :   Add required temp columns to temp_v_columns_list

`PolarsExpressionActor(*, polars_model, extend_context: bool = False, project_context: bool = False, partition_by: Optional[Iterable[str]] = None)`
:   Act on expressions in Polars context

    ### Ancestors (in MRO)

    * data_algebra.expression_walker.ExpressionWalker
    * abc.ABC

    ### Class variables

    `extend_context: bool`
    :

    `partition_by: List[str]`
    :

    `project_context: bool`
    :

    ### Methods

    `act_on_expression(self, *, arg, values: List, op)`
    :   Action for a column name.
        
        :param arg: None
        :param values: list of values to work on
        :param op: operator to apply
        :return: arg acted on

`PolarsModel(*, use_lazy_eval: bool = True)`
:   Interface for realizing the data algebra as a sequence of steps over Polars https://www.pola.rs .

    ### Ancestors (in MRO)

    * data_algebra.data_model.DataModel
    * abc.ABC

    ### Class variables

    `extend_expr_impl_map: Dict[int, Dict[str, Callable]]`
    :

    `impl_map_arbitrary_arity: Dict[str, Callable]`
    :

    `presentation_model_name: str`
    :

    `project_expr_impl_map: Dict[int, Dict[str, Callable]]`
    :

    `rng: Any`
    :

    `sql_model: data_algebra.PolarsSQL.PolarsSQLModel`
    :

    `use_lazy_eval: bool`
    :

    `want_literals_unpacked: Set[str]`
    :

    ### Methods

    `bad_column_positions(self, x)`
    :   Return vector indicating which entries are null (vectorized).

    `eval(self, op: data_algebra.data_ops_types.OperatorPlatform, *, data_map: Dict[str, Any]) ‑> polars.dataframe.frame.DataFrame`
    :   Implementation of Polars evaluation of data algebra operators
        
        :param op: ViewRepresentation to evaluate
        :param data_map: dictionary mapping table and view names to data frames
        :return: data frame result

    `eval_as_sql(self, op, *, data_map: Dict[str, Any])`
    :   Implementation of Polars evaluation through Polars SQL interface.
        https://pola-rs.github.io/polars-book/user-guide/sql.html
        Not at a useful level of development yet.
        Currently creates non-reentrant pl.SQLContext() in call.
        
        :param op: ViewRepresentation to evaluate, or SQL string
        :param data_map: dictionary mapping table and view names to data frames or data sources
        :return: data frame result

    `to_sql(self, ops: data_algebra.data_ops.ViewRepresentation, *, sql_format_options: Optional[data_algebra.sql_format_options.SQLFormatOptions] = None) ‑> str`
    :   Convert ViewRepresentation into SQL string.
        Not at a useful level of development yet.
        
        :param ops: ViewRepresentation to convert
        :param sql_format_options: sql formatting options
        :return: sql string

`PolarsTerm(*, polars_term=None, is_literal: bool = False, is_column: bool = False, is_series: bool = False, lit_value=None)`
:   Class to carry Polars expression term and annotations about expression tree.
    
    Carry a Polars expression term (polars_term) plus annotations.
    
    :param polars_term: Optional Polars expression (None means collect info, not a true term)
    :param is_literal: True if term is a constant
    :param is_column: True if term is a column name
    :param lit_value: original value for a literal
    :param inputs: inputs to expression node

    ### Class variables

    `is_column: bool`
    :

    `is_literal: bool`
    :

    `is_series: bool`
    :

    `lit_value: Any`
    :

    `polars_term: Any`
    :