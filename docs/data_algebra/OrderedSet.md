Module data_algebra.OrderedSet
==============================
Ordered set to enhance presentation of column names.

Adapted from: https://stackoverflow.com/a/1653978

Functions
---------

    
`ordered_diff(a: collections.abc.Iterable, b: collections.abc.Iterable) ‑> data_algebra.OrderedSet.OrderedSet`
:   a with b removed, a order preserved.

    
`ordered_intersect(a: collections.abc.Iterable, b: collections.abc.Iterable) ‑> data_algebra.OrderedSet.OrderedSet`
:   Intersection of two iterables, ordered by a.

    
`ordered_union(a: collections.abc.Iterable, b: collections.abc.Iterable) ‑> data_algebra.OrderedSet.OrderedSet`
:   Union of two iterables, ordered by a first, then b.

Classes
-------

`OrderedSet(v: Optional[collections.abc.Iterable] = None)`
:   Ordered set to enhance presentation of column names.

    ### Ancestors (in MRO)

    * collections.abc.MutableSet
    * collections.abc.Set
    * collections.abc.Collection
    * collections.abc.Sized
    * collections.abc.Iterable
    * collections.abc.Container

    ### Class variables

    `impl: collections.OrderedDict`
    :

    ### Instance variables

    `difference`
    :

    `difference_update`
    :

    `intersection`
    :

    `intersection_update`
    :

    `issubset`
    :

    `issuperset`
    :

    `symmetric_difference`
    :

    `symmetric_difference_update`
    :

    ### Methods

    `add(self, elem)`
    :   add an element

    `copy(self)`
    :

    `discard(self, elem)`
    :   delete an element

    `union(self, *args)`
    :   create new set union

    `update(self, *args, **kwargs)`
    :   add/replace elements