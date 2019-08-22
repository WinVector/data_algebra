
# Adapted from:
#   https://www.kylem.net/programming/tailcall.html
# Note: this is not eliminating tail-calls (replacing them with jumps), but
#       replacing then with a non-stack based recursive
#       record keeping.  So this doesn't improve speed, but does allower
#       deep calling patterns.


class PendingAdaptedFunction:
    """
    Wrap a function f into
    a type of function that whose return value may be PendingFunctionEvaluations that
    need to be realized before being further used.
    """

    def __init__(self, f):
        if isinstance(f, PendingAdaptedFunction):
            raise Exception("f should not be a PendingAdaptedFunction")
        self._f = f

    def __call__(self, *args, **kwargs):
        ret = self._f(*args, **kwargs)
        while isinstance(ret, PendingFunctionEvaluation):
            ret = ret()  # realize pending result
        return ret


class PendingFunctionEvaluation:
    """
    Store a planned function evaluation as something to be realized later.
    Essentially a call-by-need, or lazy structure such as a "thunk".
    """

    def __init__(self, f, *args, **kwargs):
        if isinstance(f, PendingAdaptedFunction):
            raise Exception("f should not be a PendingAdaptedFunction")
        self._f = f
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise Exception("there should be no args to __call__")
        if len(kwargs) > 0:
            raise Exception("there should be no kwargs to __call__")
        return self._f(*self._args, **self._kwargs)


def pending_function(f):
    """Build an adaptation of a function f that instead of evaluating f returns a PendingFunctionEvaluation,
    to be realized later."""
    if isinstance(f, PendingAdaptedFunction):
        f = f._f
    return lambda *args, **kwargs: PendingFunctionEvaluation(f, *args, **kwargs)



# def recursive_example(n, d=1):
#     if n <= 1:
#         return d
#     else:
#         return recursive_example(n-1, d+1)
# # recursive_example(10000)  # will run-out default call stack
#
#
# @PendingAdaptedFunction
# def recursive_example_tc(n, d=1):
#     if n <= 1:
#         return d
#     else:
#         return pending_function(recursive_example_tc)(n-1, d+1)
# recursive_example_tc(100000)  # works
