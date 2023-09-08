Module data_algebra.expression_walker
=====================================

Classes
-------

`ExpressionWalker()`
:   Abstract class will expression walking callbacks.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.pandas_base.PandasModelBase
    * data_algebra.polars_model.ExpressionRequirementsCollector
    * data_algebra.polars_model.PolarsExpressionActor

    ### Methods

    `act_on_column_name(self, *, arg, value)`
    :   Action for a column name.
        
        :param arg: None
        :param value: column name
        :return: arg acted on

    `act_on_expression(self, *, arg, values: List, op)`
    :   Action for a column name.
        
        :param arg: None
        :param values: list of values to work on
        :param op: data_algebra.expr_rep.Expression operator to apply
        :return: arg acted on

    `act_on_literal(self, *, value)`
    :   Action for a literal/constant in an expression.
        
        :param value: literal value being supplied
        :return: converted result