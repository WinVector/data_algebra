Module data_algebra.expr_parse
==============================
Parse expressions.

Functions
---------

    
`parse_assignments_in_context(*, ops, view)`
:   Convert all entries of ops map to Term-expressions
    
    :param ops: dictionary from strings to expressions (either Terms or strings)
    :param view: a data_algebra.data_ops.ViewRepresentation
    :return: