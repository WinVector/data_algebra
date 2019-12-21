import types


# needed to see user variables and set up _, _0, _1, and _get side-effects
_namespace_stack = []
_ref_to_outer_namespace = None


# some notes on inserting user environments:
# https://stackoverflow.com/questions/11024646/is-it-possible-to-overload-python-assignment
#    https://stackoverflow.com/a/40645627/6901725


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


def outer_namespace():
    """get current reference to side-effect namespace"""
    global _ref_to_outer_namespace
    return _ref_to_outer_namespace


class Env:
    def __init__(self, env):
        if not isinstance(env, dict):
            raise TypeError("env should be a dictionary such as locals() or globals()")
        self.env = env

    def __enter__(self):
        push_onto_namespace_stack(self.env)
        return self

    # noinspection PyShadowingBuiltins
    def __exit__(self, type, value, traceback):
        pop_from_namespace_stack()


class SimpleNamespaceDict(types.SimpleNamespace):
    """Allow square-bracket lookup on read-only types.SimpleNamespace

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

    def __setitem__(self, key, value):
        raise RuntimeError("__setitem__ not allowed")

    def __setattr__(self, name, value):
        raise RuntimeError("__setattr__ not allowed")
