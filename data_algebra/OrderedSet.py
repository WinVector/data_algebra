"""
Ordered set to enhance presentation of column names.

Adapted from: https://stackoverflow.com/a/1653978
"""

import collections
import collections.abc
from collections.abc import Iterable
from typing import Optional

# adapted from:
# https://stackoverflow.com/a/1653978


class OrderedSet(collections.abc.MutableSet):
    """
    Ordered set to enhance presentation of column names.
    """

    impl: collections.OrderedDict

    def __init__(self, v: Optional[Iterable] = None):
        self.impl = collections.OrderedDict()
        if v is not None:
            for val in v:
                self.add(val)

    def update(self, *args, **kwargs):
        """add/replace elements"""
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        """add an element"""
        self.impl[elem] = None

    def discard(self, elem):
        """delete an element"""
        self.impl.pop(elem, None)

    def __copy__(self):
        return OrderedSet(self.impl.keys())

    def copy(self):
        return OrderedSet(self.impl.keys())

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return "OrderedSet([%s])" % (", ".join(map(repr, self.impl.keys())))

    def __str__(self):
        return "{%s}" % (", ".join(map(repr, self.impl.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)

    def __len__(self):
        return len(self.impl)

    def __iter__(self):
        return iter(self.impl.keys())

    def __contains__(self, item):
        return item in self.impl.keys()

    def union(self, *args):
        """create new set union"""
        res = OrderedSet()
        for k in self.impl.keys():
            res.add(k)
        for other in args:
            for k in other:
                if k not in res:
                    res.add(k)
        return res


def ordered_intersect(a: Iterable, b: Iterable) -> OrderedSet:
    """
    Intersection of two iterables, ordered by a.
    """
    b = set(b)
    return OrderedSet([v for v in a if v in b])


def ordered_union(a: Iterable, b: Iterable) -> OrderedSet:
    """
    Union of two iterables, ordered by a first, then b.
    """
    a = OrderedSet(a)
    for v in b:
        if v not in a:
            a.add(v)
    return a


def ordered_diff(a: Iterable, b: Iterable) -> OrderedSet:
    """
    a with b removed, a order preserved.
    """
    b = set(b)
    a = OrderedSet([v for v in a if v not in b])
    return a
