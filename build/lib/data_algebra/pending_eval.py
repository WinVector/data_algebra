# Some good reading on thunks in Python here:
#   https://www.kylem.net/programming/tailcall.html


# Exception based idea


class PendingFunctionEvaluation(Exception):
    """
    Store a planned function evaluation as something to be realized later.
    Essentially a call-by-need, or lazy structure such as a "thunk".

    Note: this throw-pattern can only be used in the case where tail-calls are only called by tail-calls,
    as the raise will throw-through intermediate code.
    """

    def __init__(self, f, *args, **kwargs):
        if isinstance(f, PendingFunctionEvaluation):
            raise TypeError("f should not be a PendingFunctionEvaluation")
        self._f = f
        self._args = args
        self._kwargs = kwargs
        Exception.__init__(self, "PendingFunctionEvaluation")

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError("there should be no args to __call__")
        if len(kwargs) > 0:
            raise ValueError("there should be no kwargs to __call__")
        v = self._f(*self._args, **self._kwargs)
        return v


def tail_call(f):
    return lambda *args, **kwargs: PendingFunctionEvaluation(f, *args, **kwargs)


def unpack_result(v):
    while isinstance(v, PendingFunctionEvaluation):
        try:
            v = v()
        except PendingFunctionEvaluation as pve:
            v = pve
    return v


def eval_tail(f, *args, **kwargs):
    try:
        v = f(*args, **kwargs)
    except PendingFunctionEvaluation as pve:
        v = pve
    return unpack_result(v)


def tail_version(f):
    return lambda *args, **kwargs: eval_tail(f, *args, **kwargs)


def eval_using_exceptions(f, *args, **kwargs):
    return eval_tail(f, *args, **kwargs)
