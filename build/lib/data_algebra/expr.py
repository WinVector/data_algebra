# adapters uses can use to avoid parsing path


import data_algebra.expr_rep


def _row_number():
    return data_algebra.expr_rep.Expression(op="row_number", args=[])


def row_number():
    return data_algebra.expr_rep.Expression(op="row_number", args=[])


def r_parse_env():
    return {
        "exp": lambda x: x.exp(),
        "sum": lambda x: x.sum(),
        "_row_number": _row_number,
        "row_number": row_number,
    }


class StandInNamespace:
    """implement get on all possible values"""

    def __init__(self):
        pass

    def __getitem__(self, key):
        return data_algebra.expr_rep.ColumnReference(view=None, column_name=key)

    def __setitem__(self, key, value):
        raise RuntimeError("__setitem__ not allowed")

    def __getattr__(self, name):
        return data_algebra.expr_rep.ColumnReference(view=None, column_name=name)

    def __setattr__(self, name, value):
        raise RuntimeError("__setattr__ not allowed")


def frame():
    return StandInNamespace()
