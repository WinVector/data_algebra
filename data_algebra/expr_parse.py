
import collections

import data_algebra.expr_rep
import data_algebra.parse_by_lark


def parse_assignments_in_context(ops, view, *, parse_env=None):
    """
    Convert all entries of ops map to Term-expressions

    :param ops: dictionary from strings to expressions (either Terms or strings)
    :param view: a data_algebra.data_ops.ViewRepresentation
    :param parse_env map of names to values to add to parsing environment, TODO: get rid of this
    :return:
    """
    if ops is None:
        ops = {}
    if isinstance(ops, tuple) or isinstance(ops, list):
        opslen = len(ops)
        ops = {k: v for (k, v) in ops}
        if opslen != len(ops):
            raise ValueError("ops keys must be unique")
    if not isinstance(ops, dict):
        raise TypeError("ops should be a dictionary")
    column_defs = view.column_map
    if not isinstance(column_defs, dict):
        raise TypeError("column_defs should be a dictionary")
    if parse_env is None:
        parse_env = {}
    # first: make sure all entries are parsed
    columns_used = set()
    newops = collections.OrderedDict()
    mp = column_defs.copy()
    data_algebra.expr_rep.populate_specials(column_defs=column_defs, destination=mp, user_values=parse_env)
    for k in ops.keys():
        if not isinstance(k, str):
            raise TypeError("ops keys should be strings")
        orig_v = ops[k]  # make debugging easier
        v = orig_v
        if not isinstance(v, data_algebra.expr_rep.PreTerm):
            v = data_algebra.parse_by_lark.parse_by_lark(
                source_str=str(v), data_def=mp, outer_environment=parse_env
            )
        else:
            v = v.replace_view(view)
        newops[k] = v
        used_here = set()
        v.get_column_names(used_here)
        columns_used = columns_used.union(
            used_here - {k}
        )  # can use a column to update itself
    intersect = set(ops.keys()).intersection(columns_used)
    if len(intersect) > 0:
        raise ValueError(
            "columns both produced and used in same expression set: " + str(intersect)
        )
    return newops
