"""
Utility to merge extend operations, when appropriate.
"""

import data_algebra.expr_rep


def try_to_merge_ops(ops1, ops2):
    """
    Try to merge two extends into one. Return merged op, or None if not possible.
    """
    ops1_columns_used = set(data_algebra.expr_rep.get_columns_used(ops1))
    ops1_columns_produced = set([k for k in ops1.keys()])
    ops2_columns_used = set(data_algebra.expr_rep.get_columns_used(ops2))
    ops2_columns_produced = set([k for k in ops2.keys()])
    common_produced = ops1_columns_produced.intersection(ops2_columns_produced)
    if len(common_produced) > 0:
        ops1_common = {k: ops1[k] for k in common_produced}
        ops2_common = {k: ops2[k] for k in common_produced}
        ops1_common_columns_used = set(
            data_algebra.expr_rep.get_columns_used(ops1_common)
        )
        ops2_common_columns_used = set(
            data_algebra.expr_rep.get_columns_used(ops2_common)
        )
        if len(ops1_common_columns_used.intersection(ops2_columns_produced)) > 0:
            return None
        if len(ops1_common_columns_used.intersection(ops1_columns_produced)) > 0:
            return None
        if len(ops2_common_columns_used.intersection(ops1_columns_produced)) > 0:
            return None
        if len(ops2_common_columns_used.intersection(ops2_columns_produced)) > 0:
            return None
        new_ops = {
            k: ops1[k]
            for k in ops1.keys()
            if k not in common_produced
        }
        new_ops.update(ops2)
        return new_ops
    # check required disjointness conditions
    if len(ops1_columns_produced.intersection(ops2_columns_produced)) > 0:
        return None
    if len(ops1_columns_used.intersection(ops2_columns_produced)) > 0:
        return None
    if len(ops2_columns_used.intersection(ops1_columns_produced)) > 0:
        return None

    # merge the extends
    new_ops = ops1.copy()
    new_ops2 = ops2.copy()
    new_ops.update(new_ops2)
    return new_ops
