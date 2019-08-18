
import data_algebra.table_rep

# have user override this with:
# import data_algebra
# data_algebra._ref_to_global_namespace = globals()
_ref_to_global_namespace = None

def mk_td(table_name, column_names,
          *,
          qualifiers = None):
    """Make a table representation object.

       If data_algebra._ref_to_global_namespace = globals() then
       _ and _0 are set to column name maps as a side-effect.

       Example:
           import data_algebra
           data_algebra._ref_to_global_namespace = globals()
           d = data_algebra.mk_td('d', ['x', 'y'])
    """
    vr = data_algebra.table_rep.ViewRepresentation(
        table_name = table_name,
        column_names = column_names,
        qualifiers = qualifiers)
    # make last result referable by names _ and _0
    if _ref_to_global_namespace is not None:
        _ref_to_global_namespace['_'] = vr.column_map
        _ref_to_global_namespace['_0'] = vr.column_map
    return vr

