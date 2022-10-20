"""
Simple class for holding SQL formatting options
"""

from types import SimpleNamespace


class SQLFormatOptions(SimpleNamespace):
    """
    Simple class for holding SQL formatting options

    :param use_with: bool, if True use with to introduce common table expressions
    :param annotate: bool, if True add annotations from original pipeline as SQL comments
    :param sql_indent: str = " ", indent string (must be non-empty and all whitespace)
    :param initial_commas: bool = False, if True write initial commas instead of after commas
    :param warn_on_method_support: bool = True, if True warn on translation to untrusted methods
    :param warn_on_novel_methods: bool = True, if True warn on translation to unrecognized methods
    :param use_cte_elim: bool = False, if True optimize SQL by re-using common table expressions (experimental)
    """

    def __init__(
        self,
        use_with: bool = True,
        annotate: bool = True,
        sql_indent: str = " ",
        initial_commas: bool = False,
        warn_on_method_support: bool = True,
        warn_on_novel_methods: bool = True,
        use_cte_elim: bool = False,
    ):
        """
        SQL formatting options.

        :param use_with: bool, if True use with to introduce common table expressions
        :param annotate: bool, if True add annotations from original pipeline as SQL comments
        :param sql_indent: str = " ", indent string (must be non-empty and all whitespace)
        :param initial_commas: bool = False, if True write initial commas instead of after commas
        :param warn_on_method_support: bool = True, if True warn on translation to untrusted methods
        :param warn_on_novel_methods: bool = True, if True warn on translation to unrecognized methods
        :param use_cte_elim: bool = False, if True optimize SQL by re-using common table expressions (experimental)
        """
        assert isinstance(use_with, bool)
        assert isinstance(annotate, bool)
        assert isinstance(sql_indent, str)
        assert len(sql_indent) > 0
        assert len(sql_indent.strip()) == 0
        assert isinstance(initial_commas, bool)
        assert isinstance(use_cte_elim, bool)
        SimpleNamespace.__init__(
            self,
            use_with=use_with,
            annotate=annotate,
            sql_indent=sql_indent,
            initial_commas=initial_commas,
            warn_on_method_support=warn_on_method_support,
            warn_on_novel_methods=warn_on_novel_methods,
            use_cte_elim=use_cte_elim,
        )

    def __str__(self):
        return self.__repr__()

    # noinspection PyUnusedLocal
    def _repr_pretty_(self, p, cycle):
        """
        IPython pretty print, used at implicit print time
        https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        p.text(str(self))
