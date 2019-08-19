
import types

# needed to see user variables and set up _, _0, _1, and _get side-effects
_namespace_stack = []
_ref_to_outer_namespace = None


def push_onto_namespace_stack(env):
    global _ref_to_outer_namespace
    global _namespace_stack
    _namespace_stack.append(_ref_to_outer_namespace)
    _ref_to_outer_namespace = env
    return env

def pop_from_namespace_stack():
    global _ref_to_outer_namespace
    global _namespace_stack
    _ref_to_outer_namespace = _namespace_stack.pop()
    return _ref_to_outer_namespace

def _outer_namespace():
    """get current reference to side-effect namespace"""
    global _ref_to_outer_namespace
    return _ref_to_outer_namespace


class Env:
    def __init__(self, env):
        if not isinstance(env, dict):
            raise Exception("env should be a dictionary such as locals() or globals()")
        self.env = env

    def __enter__(self):
        push_onto_namespace_stack(self.env)
        return self

    def __exit__(self, type, value, traceback):
        pop_from_namespace_stack()


class SimpleNamespaceDict(types.SimpleNamespace):
    """Allow square-bracket lookup on SimpleNamespace

       Example:
           import data_algebra.env
           d = data_algebra.env.SimpleNamespaceDict(**{'x':1, 'y':2})
           d.x
           var_name = 'y'
           d[var_name]

    """

    def __init__(self, **kwargs):
        types.SimpleNamespace.__init__(self, **kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]


def _populate_specials(*,
                       column_defs,
                       column_defs1 = None,
                       destination,
                       user_values = None):
    """populate a dictionary with special values
       column_defs is a dictionary,
         usually formed from a ViewRepresentation.column_map.__dict__
       destination is a dictionary,
         usually formed from a ViewRepresentation.column_map.__dict__.copy()
    """

    if not isinstance(column_defs, dict):
        raise Exception("column_defs should be a dictionary")
    if not isinstance(destination, dict):
        raise Exception("destination should be a dictionary")
    if user_values is None:
        user_values = {}
    if not isinstance(user_values, dict):
        raise Exception("user_values should be a dictionary")
    ns = SimpleNamespaceDict(**column_defs.copy())
    destination["_"] = ns
    destination["_0"] = ns
    if column_defs1 is not None:
        if not isinstance(column_defs1, dict):
            raise Exception("column_defs1 should be a dictionary")
        ns1 = SimpleNamespaceDict(**column_defs.copy())
        destination["_1"] = ns1
    else:
        destination["_1"] = None
    destination["_get"] = lambda key: user_values[key]

def _maybe_set_underbar(*, mp0, mp1=None):
    destination = _outer_namespace()
    if destination is not None:
        _populate_specials(column_defs = mp0,
                           column_defs1 = mp1,
                           destination=destination,
                           user_values=destination)

