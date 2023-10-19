Module data_algebra.sql_format_options
======================================
Simple class for holding SQL formatting options

Classes
-------

`SQLFormatOptions(use_with: bool = True, annotate: bool = True, sql_indent: str = ' ', initial_commas: bool = False, warn_on_method_support: bool = True, warn_on_novel_methods: bool = True, use_cte_elim: bool = False)`
:   Simple class for holding SQL formatting options
    
    :param use_with: bool, if True use with to introduce common table expressions
    :param annotate: bool, if True add annotations from original pipeline as SQL comments
    :param sql_indent: str = " ", indent string (must be non-empty and all whitespace)
    :param initial_commas: bool = False, if True write initial commas instead of after commas
    :param warn_on_method_support: bool = True, if True warn on translation to untrusted methods
    :param warn_on_novel_methods: bool = True, if True warn on translation to unrecognized methods
    :param use_cte_elim: bool = False, if True optimize SQL by re-using common table expressions (experimental)
    
    SQL formatting options.
    
    :param use_with: bool, if True use with to introduce common table expressions
    :param annotate: bool, if True add annotations from original pipeline as SQL comments
    :param sql_indent: str = " ", indent string (must be non-empty and all whitespace)
    :param initial_commas: bool = False, if True write initial commas instead of after commas
    :param warn_on_method_support: bool = True, if True warn on translation to untrusted methods
    :param warn_on_novel_methods: bool = True, if True warn on translation to unrecognized methods
    :param use_cte_elim: bool = False, if True optimize SQL by re-using common table expressions (experimental)

    ### Ancestors (in MRO)

    * types.SimpleNamespace