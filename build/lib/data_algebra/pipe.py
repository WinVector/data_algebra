class PipeStep:
    """class to extend to make pipe transform stages
    Examples:
        import data_algebra.pipe
        import math
        s = data_algebra.pipe.PipeFunction(math.sin)
        c = data_algebra.pipe.PipeFunction(math.cos)
        5 >> s >> c
        # Out[]: 0.574400879193934
        5 >> (s >> c)
        # Out[]: 0.574400879193934
        (5 >> s) >> c
        # Out[]: 0.574400879193934
    """

    def __init__(self, *, name=None):
        if name is None:
            name = "PipeStep"
        self.__name__ = name

    def apply(self, other, **kwargs):
        raise NotImplementedError("base method called")

    def __rrshift__(self, other):  # override other >> self
        return self.apply(other)

    def __rshift__(self, other):  # override self >> other
        if isinstance(other, PipeStep):
            return PipeFunction(
                lambda x: other.apply(self.apply(x)),
                name=self.__name__ + " >> " + other.__name__,
            )
        # assume RHS is a function
        return PipeFunction(
            lambda x: other(self.apply(x)), name=self.__name__ + " >> " + other.__name__
        )

    def __repr__(self):
        return "(" + self.__name__ + ")"

    def __str__(self):
        return "(" + self.__name__ + ")"


class PipeFunction(PipeStep):
    """wraps single argument functions into a >> pipeable entity
    Examples:
        import data_algebra.pipe
        import math
        s = data_algebra.pipe.PipeFunction(math.sin)
        c = data_algebra.pipe.PipeFunction(math.cos)
        5 >> s >> c
        # Out[]: 0.574400879193934
        5 >> (s >> c)
        # Out[]: 0.574400879193934
        (5 >> s) >> c
        # Out[]: 0.574400879193934
    """

    def __init__(self, func, *, args_to_override=None, partial_args=None, name=None):
        if not callable(func):
            raise TypeError("func must be callable")
        if partial_args is None:
            partial_args = {}
        if args_to_override is None:
            args_to_override = []
        self._partial_args = partial_args
        self._args_to_override = args_to_override
        self._func = func
        if name is None:
            PipeStep.__init__(self, name=func.__name__)
        else:
            PipeStep.__init__(self, name=name)

    def apply(self, other, **kwargs):
        if len(self._args_to_override) < 1:
            # place argument in first position
            return self._func(other, **self._partial_args)
        #  place argument by name
        args = self._partial_args.copy()
        for nm in self._args_to_override:
            args[nm] = other
        return self._func(**args)


class PipeValue:
    """Extend this class to be a value that can be piped into functions
    in addition to PipeFunctions.

    Examples:
        import data_algebra.pipe
        import math
        v = data_algebra.pipe.Value(5)
        v >> math.sin
        # Out[]: -0.9589242746631385
        s = data_algebra.pipe.PipeFunction(math.sin)
        v >> s
        # Out[]: -0.9589242746631385
    """

    def __init__(self):
        pass

    def __val__(self):
        return self

    def __rshift__(self, other):
        if isinstance(other, PipeStep):
            return other.apply(self.__val__())
        # assume RHS is a function
        return other(self.__val__())


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
        PipeValue.__init__(self)

    def __val__(self):
        return self.value

    def __repr__(self):
        return "Value(" + self.__val__().__repr__() + ")"


def build_pipeline(*steps):
    n = len(steps)
    if n < 1:
        raise ValueError("steps was empty")
    if n == 1:
        return steps[0]
    cur = steps[0]
    for i in range(1, n):
        # cur = cur >> steps[i]
        cur = steps[i].apply(cur)
    return cur
