class Component:
    def __init__(self, item):
        self.id = item
        self.items = {item}


def connected_components(f, g):
    """
    Compute connected components of undirected edges (f[i], g[i]).

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
    """
    keys = set([k for k in f]).union((k for k in g))
    components = {k: Component(k) for k in keys}
    for fi, gi in zip(f, g):
        # print({k: components[k].id for k in keys})
        component_f = components[fi]
        component_g = components[gi]
        if component_f.id != component_g.id:
            if len(component_f.items) >= len(component_g.items):
                merged = component_f
                donor = component_g
            else:
                merged = component_g
                donor = component_f
            merged.items.update(donor.items)
            merged.id = min(merged.id, donor.id)
            for k in donor.items:
                components[k] = merged
    assignments = [components[k].id for k in f]
    return assignments
