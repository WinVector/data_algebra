Module data_algebra.expr_parse_fn
=================================
Parse expresion to data ops in a medium environment. Uses eval() (don't run on untrusted expressions).
Used for confirming repr() is reversible.

Functions
---------

    
`eval_da_ops(ops_str: str, *, data_model_map: Optional[Dict[str, Any]]) ‑> data_algebra.view_representations.ViewRepresentation`
:   Parse ops_str into a ViewRepresentation. Uses eval() (don't run on untrusted expressions).
    Used for confirming repr() is reversible.
    
    :param ops_str: text representation of a data algebra pipeline or expression.
    :param data_model_map: tables
    :return: data algebra ops