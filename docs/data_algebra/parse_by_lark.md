Module data_algebra.parse_by_lark
=================================
Use Lark to parse a near-Python expression grammar.

Functions
---------

    
`parse_by_lark(source_str: str, *, data_def=None) ‑> data_algebra.expr_rep.Term`
:   Parse an expression in terms of data views and values.
    
    :param source_str: string to parse
    :param data_def: dictionary of data_algebra.expr_rep.ColumnReference
    :return: data_algebra.expr_rep.Term