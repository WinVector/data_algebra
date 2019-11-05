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


def get_index_lists(partition_columns):
    """
    Find lists of indices for partition_ids level-sets

    :param partition_columns: non-empty list of lists of list of hashables/comparables
    :return: dictionary of index lists for each level-set of the keys
    """
    index_lists = {}
    for i in range(len(partition_columns[0])):
        pi = []
        for j in range(len(partition_columns)):
            pi.append(partition_columns[j][i])
        pi = tuple(pi)
        try:
            lst = index_lists[pi]
        except KeyError:
            lst = []
            index_lists[pi] = lst
        lst.append(i)
    return index_lists


def partitioned_eval(fn, arg_columns, partition_columns):
    """
    Evaluate fn(fn_columns) on level-sets of partition columms.

    :param fn: function with arity length(fn_columns) then returns lists/vectors of the common length of its arguments.
    :param arg_columns: non-empty list of lists of length n
    :param partition_columns: possibly empty list of lists of hashables/comparables of all length n
    :return: list of length n of fn(fn_columns) evaluted on level-sets of partition_columns
    """
    if (partition_columns is None) or (len(partition_columns) <= 0):
        return fn(*arg_columns)
    arity = len(arg_columns)
    res = [None] * len(arg_columns[0])
    index_lists = get_index_lists(partition_columns)
    for indexs in index_lists.values():
        # slice out a level set
        ni = len(indexs)
        fn_cols_i = [
            [arg_columns[j][indexs[i]] for i in range(ni)] for j in range(arity)
        ]
        # call function on slice
        hi = fn(*fn_cols_i)
        # copy back result
        for i in range(ni):
            res[indexs[i]] = hi[i]
    return res
