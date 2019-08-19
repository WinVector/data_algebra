
# needed to see user variables and set up _, _0, and _1 side-effects
_namespace_stack = []
_ref_to_outer_namespace = None

class Env:
    def __init__(self, env):
        if not isinstance(env, dict):
            raise Exception("env should be a dictionary such as locals() or globals()")
        self.env = env

    def __enter__(self):
        global _ref_to_outer_namespace
        global _namespace_stack
        _namespace_stack.append(_ref_to_outer_namespace)
        _ref_to_outer_namespace = self.env
        return self

    def __exit__(self, type, value, traceback):
        global _ref_to_outer_namespace
        global _namespace_stack
        _ref_to_outer_namespace = _namespace_stack.pop()

def outer_namespace():
    global _ref_to_outer_namespace
    return _ref_to_outer_namespace