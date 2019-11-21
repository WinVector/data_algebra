
import data_algebra.expr_rep

def try_to_merge_ops(ops1, ops2):
    ops1_columns_used = set(data_algebra.expr_rep.get_columns_used(ops1))
    ops1_columns_produced = set([k for k in ops1.keys()])
    ops2_columns_used = set(data_algebra.expr_rep.get_columns_used(ops2))
    ops2_columns_produced = set([k for k in ops2.keys()])
    common_produced = ops1_columns_produced.intersection(ops2_columns_produced)
    if len(common_produced) > 0:
        return None  # TODO: consider some merge opportunities here
    # check required disjointness conditions
    if len(ops1_columns_produced.intersection(ops2_columns_produced)) > 0:
        return None
    if len(ops1_columns_used.intersection(ops2_columns_produced)) > 0:
        return None
    if len(ops2_columns_used.intersection(ops1_columns_produced)) > 0:
        return None
    # merge the extends
    new_ops = ops1.copy()
    new_ops.update(ops2)
    return new_ops
