import collections
import collections.abc

# adapted from:
# https://stackoverflow.com/a/1653978


class OrderedSet(collections.OrderedDict, collections.abc.MutableSet):
    def __init__(self, v=None):
        collections.OrderedDict.__init__(self)
        if v is not None:
            for val in v:
                self.add(val)

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return "OrderedSet([%s])" % (", ".join(map(repr, self.keys())))

    def __str__(self):
        return "{%s}" % (", ".join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self.keys()

    def union(self, *args):
        res = OrderedSet()
        for k in self.keys():
            res.add(k)
        for other in args:
            for k in other:
                if k not in res:
                    res.add(k)
        return res
