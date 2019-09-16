import collections

import data_algebra
import data_algebra.data_ops
import data_algebra.data_pipe

try:
    # noinspection PyUnresolvedReferences
    import yaml  # supplied by PyYAML
except ImportError:
    pass


# yaml notes:
#    https://stackoverflow.com/questions/2627555/how-to-deserialize-an-object-with-pyyaml-using-safe-load
#    https://stackoverflow.com/a/21912744/6901725


def fix_ordered_dict_yaml_rep():
    """Writer OrderedDict as simple structure"""
    # derived from: https://stackoverflow.com/a/16782282/6901725
    if not data_algebra.have_yaml:
        raise RuntimeError("yaml/PyYAML not installed")

    def represent_ordereddict(dumper, data):
        value = [
            (dumper.represent_data(node_key), dumper.represent_data(node_value))
            for (node_key, node_value) in data.items()
        ]
        return yaml.nodes.MappingNode(u"tag:yaml.org,2002:map", value)

    yaml.add_representer(collections.OrderedDict, represent_ordereddict)


def to_pipeline(obj, *, known_tables=None):
    """De-serialize data_algebra operator pipeline from a collect_representation() form.
       This form is good for yaml serialization/de-serialization.

       Note: eval() is called to interpret expressions on some nodes, so this
       function is not safe to use on untrusted code (though a somewhat restricted
       version of eval() is used to try and catch some issues).
    """
    if known_tables is None:
        known_tables = {}

    def maybe_get_dict(map, key):
        try:
            return map[key]
        except KeyError:
            return {}

    def maybe_get_list(map, key):
        try:
            return map[key]
        except KeyError:
            return []

    def maybe_get_none(map, key):
        try:
            return map[key]
        except KeyError:
            return None

    if isinstance(obj, dict):
        # a pipe stage
        op = obj["op"]
        # ugly switch statement
        if op == "TableDescription":
            tab = data_algebra.data_ops.TableDescription(
                table_name=obj["table_name"],
                column_names=obj["column_names"],
                qualifiers=maybe_get_dict(obj, "qualifiers"),
                )
            # canonicalize to one object per table
            k = tab.key
            if k in known_tables.keys():
                ov = known_tables[k]
                if tab.column_set != ov.column_set:
                    raise ValueError(
                        "two tables with same qualified table names and different declared column sets"
                    )
                tab = ov
            else:
                known_tables[k] = tab
            return tab
        elif op == "Extend":
            return data_algebra.data_pipe.Extend(
                ops=obj["ops"],
                partition_by=maybe_get_list(obj, "partition_by"),
                order_by=maybe_get_list(obj,"order_by"),
                reverse=maybe_get_list(obj, "reverse"),
            )
        elif op == "Project":
            return data_algebra.data_pipe.Project(
                ops=obj["ops"],
                group_by=maybe_get_list(obj, "group_by"),
            )
        elif op == "NaturalJoin":
            return data_algebra.data_pipe.NaturalJoin(
                by=obj["by"],
                jointype=obj["jointype"],
                b=to_pipeline(obj["b"], known_tables=known_tables),
            )
        elif op == "SelectRows":
            return data_algebra.data_pipe.SelectRows(expr=obj["expr"])
        elif op == "SelectColumns":
            return data_algebra.data_pipe.SelectColumns(columns=obj["columns"])
        elif op == "DropColumns":
            return data_algebra.data_pipe.DropColumns(
                column_deletions=obj["column_deletions"]
            )
        elif op == "Rename":
            return data_algebra.data_pipe.RenameColumns(
                column_remapping=obj["column_remapping"]
            )
        elif op == "Order":
            return data_algebra.data_pipe.OrderRows(
                columns=maybe_get_list(obj, "order_columns"),
                reverse=maybe_get_list(obj, "reverse"),
                limit=maybe_get_none(obj, "limit"),
            )
        else:
            raise TypeError("Unexpected op name: " + op)
    if isinstance(obj, list):
        # a pipeline, assumed non empty
        res = to_pipeline(obj[0], known_tables=known_tables)
        for i in range(1, len(obj)):
            nxt = to_pipeline(obj[i], known_tables=known_tables)
            # res = res >> nxt
            res = nxt.apply(res)
        return res
    raise TypeError("unexpected type: " + str(obj))


# for testing


def check_op_round_trip(o):
    if not isinstance(o, data_algebra.data_ops.ViewRepresentation):
        raise TypeError("expect o to be a data_algebra.data_ops.ViewRepresentation")
    if not data_algebra.have_yaml:
        raise RuntimeError("yaml/PyYAML not installed")
    strr = o.to_python(strict=True, pretty=False)
    strp = o.to_python(strict=True, pretty=True)
    obj = o.collect_representation()
    back = to_pipeline(obj)
    strr_back = back.to_python(strict=True, pretty=False)
    assert strr == strr_back
    strp_back = back.to_python(strict=True, pretty=True)
    assert strp == strp_back
    dmp = yaml.dump(obj)
    ld = yaml.safe_load(dmp)
    back = to_pipeline(ld)
    if isinstance(o, data_algebra.data_ops.ExtendNode):
        if len(o.ops) == 1:
            strr_back = back.to_python(strict=True, pretty=False)
            assert strr == strr_back
            strp_back = back.to_python(strict=True, pretty=True)
            assert strp == strp_back
