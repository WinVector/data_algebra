
import data_algebra.util
import data_algebra.env
import data_algebra.custom_functions

import data_algebra.expr_rep

# Some notes on trying to harden eval:
#  http://lybniz2.sourceforge.net/safeeval.html
def _parse_by_eval(source_str, *, data_def=None, outer_environment=None):
    if not isinstance(source_str, str):
        source_str = str(source_str)
    if data_def is None:
        data_def = {}
    if outer_environment is None:
        outer_environment = {}
    else:
        outer_environment = {
            k: v for (k, v) in outer_environment.items() if not k.startswith("_")
        }
    # don't have to completely kill this environment, as the code is something
    # the user intends to run (and may have even typed in).
    # But let's cut down the builtins anyway.
    outer_environment["__builtins__"] = {
        k: v for (k, v) in outer_environment.items() if isinstance(v, Exception)
    }
    v = eval(
        source_str, outer_environment, data_def
    )  # eval is eval(source, globals, locals)- so mp is first
    if not isinstance(v, data_algebra.expr_rep.PreTerm):
        v = data_algebra.expr_rep._enc_value(v)
    v.source_string = source_str
    return v