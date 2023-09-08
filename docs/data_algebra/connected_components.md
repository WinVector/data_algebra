Module data_algebra.connected_components
========================================
Code for computing collected components.

Functions
---------

    
`connected_components(f, g)`
:   Compute connected components of undirected edges (f[i], g[i]).
    
    For the return value we are using the
    category formulation that these are the co-equalizer of f and g,
    meaning it is a finest partition such that return[f[i]] = return[g[i]]
    for all i.  We pick the least item in each component as the representation.
    This is just a long way of saying: as each side of an edge is in the same
    component, we return the assignment by labeling the edges by components
    (instead of the vertices).
    
    Not as fast as union/find but fast.
    
    f = [1, 4, 6, 2, 1]
    g = [2, 5, 7, 3, 7]
    res = connected_components(f, g)
    print(res)
    
    :param f: list or vector of hashable/comparable items of length n
    :param g: list or vector of hashable/comparable items of length n
    :return: list of assignments of length n (map both f and g to same values.

Classes
-------

`Component(item)`
:   Holder for a connected component.