


import functools

class PipeFunction:
    """wraps single argument functions into a >> pipeable entity
    Examples:
        import data_algebra.pipe
        import math
        s = PipeFunction(math.sin)
        c = PipeFunction(math.cos)
        5 >> s >> c
        # Out[]: 0.574400879193934
        5 >> (s >> c)
        # Out[]: 0.574400879193934
        (5 >> s) >> c
        # Out[]: 0.574400879193934
    """
    def __init__(self, func, name=None):
        self.func = func
        if name is None:
            self.__name__ = func.__name__
        else:
            self.__name__ = name

    def __rrshift__(self, other):
        return self.func(other)

    def __rshift__(self, other):
        if isinstance(other, PipeFunction):
            return PipeFunction(
                lambda x: other.func(self.func(x)),
                name = self.__name__ + " >> " + other.__name__)
        # assume RHS is a function
        return PipeFunction(
            lambda x: other(self.func(x)),
            name = self.__name__ + " >> " + other.__name__)

    def __repr__(self):
        return "(" + self.__name__ + ")"

    def __str__(self):
        return "(" + self.__name__ + ")"


class PipeValue:
    """Extend this class to be a value that can be piped into functions
    in additon to PipeFunctions.

    Examples:
        import data_algebra.pipe
        import math
        v = Value(5)
        v >> math.sin
        # Out[]: -0.9589242746631385
        s = PipeFunction(math.sin)
        v >> s
        # Out[]: -0.9589242746631385
    """
    def __init__(self):
        pass

    def __val__(self):
        return self

    def __rshift__(self, other):
        if isinstance(other, PipeFunction):
            return other.func(self.__val__())
        # assume RHS is a function
        return other(self.__val__())

    def __repr__(self):
        return "<" + str(self.__val__()) + ">"

    def __str__(self):
        return "<" + str(self.__val__()) + ">"


class Value(PipeValue):
    """wraps a value into a >> pipeable entity
    Examples:
        import data_algebra.pipe
        import math
        v = Value(5)
        v >> math.sin
        # Out[]: -0.9589242746631385
        s = PipeFunction(math.sin)
        v >> s
        # Out[]: -0.9589242746631385
    """
    def __init__(self, value):
        self.value = value

    def __val__(self):
        return self.value

    def __repr__(self):
        return "Value(" + self.__val__().__repr__() + ")"
