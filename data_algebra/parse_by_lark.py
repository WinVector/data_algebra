
import pkg_resources

import lark

import data_algebra.util
import data_algebra.env
import data_algebra.custom_functions

import data_algebra.expr_rep

# TODO: switch to lark parsing and use

# set up parser
parser = None
# noinspection PyBroadException
try:
    grammar = pkg_resources.resource_string(__name__, 'data/python3.lark').decode("utf-8")
    kwargs = {
        'start': 'single_input',
        }
    parser = lark.Lark(
        grammar,
        parser='lalr',
        **kwargs)
except:
    parser = None

def _parse_by_lark(source_str, *, data_def=None, outter_environemnt=None):
    assert parser is not None
    if not isinstance(source_str, str):
        source_str = str(source_str)
    if data_def is None:
        data_def = {}
    if outter_environemnt is None:
        outter_environemnt = {}
    else:
        outter_environemnt = {
            k: v for (k, v) in outter_environemnt.items() if not k.startswith("_")
        }
    # don't have to completely kill this environment, as the code is something
    # the user intends to run (and may have even typed in).
    # But let's cut down the builtins anyway.
    outter_environemnt["__builtins__"] = {
        k: v for (k, v) in outter_environemnt.items() if isinstance(v, Exception)
    }
    tree = parser.parse(source_str + '\n')
    # v = eval(
    #     source_str, outter_environemnt, data_def
    # )  # eval is eval(source, globals, locals)- so mp is first
    # if not isinstance(v, data_algebra.expr_rep.PreTerm):
    #     v = data_algebra.expr_rep._enc_value(v)
    # v.source_string = source_str
    # return v
    return tree
