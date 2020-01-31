import re
import collections

import data_algebra
import data_algebra.data_ops
import data_algebra.data_pipe
import data_algebra.cdata_impl

have_yaml = False
try:
    # noinspection PyUnresolvedReferences
    import yaml  # supplied by PyYAML

    have_yaml = True
except ImportError:
    pass


# yaml notes:
#    https://stackoverflow.com/questions/2627555/how-to-deserialize-an-object-with-pyyaml-using-safe-load
#    https://stackoverflow.com/a/21912744/6901725


def fix_ordered_dict_yaml_rep():
    """Writer OrderedDict as simple structure"""
    # derived from: https://stackoverflow.com/a/16782282/6901725
    if not have_yaml:
        raise RuntimeError("yaml/PyYAML not installed")

    def represent_ordereddict(dumper, data):
        value = [
            (dumper.represent_data(node_key), dumper.represent_data(node_value))
            for (node_key, node_value) in data.items()
        ]
        return yaml.nodes.MappingNode(u"tag:yaml.org,2002:map", value)

    yaml.add_representer(collections.OrderedDict, represent_ordereddict)


def to_pipeline(obj, *, known_tables=None, parse_env=None):
    """De-serialize data_algebra operator pipeline from a collect_representation() form.
       This form is good for yaml serialization/de-serialization.

       Note: eval() is called to interpret expressions on some nodes, so this
       function is not safe to use on untrusted code (though a somewhat restricted
       version of eval() is used to try and catch some issues).
    """
    if known_tables is None:
        known_tables = {}

    def maybe_get_dict(omap, key):
        try:
            return omap[key]
        except KeyError:
            return {}

    def maybe_get_list(omap, key):
        try:
            v = omap[key]
            if v is None:
                return []
            if not isinstance(v, list):
                v = [v]
            return v
        except KeyError:
            return []

    def maybe_get_none(omap, key):
        try:
            return omap[key]
        except KeyError:
            return None

    def get_char_scalar(omap, key):
        v = omap[key]
        if isinstance(v, list):
            v = v[0]
        if isinstance(v, dict):
            v = [vi for vi in v.values()][0]
        return v

    if isinstance(obj, dict):
        # a pipe stage
        op = get_char_scalar(obj, "op")
        # ugly switch statement
        if op == "TableDescription":
            tab = data_algebra.data_pipe.TableDescription(
                table_name=get_char_scalar(obj, "table_name"),
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
            ops = obj["ops"]
            ops = {k: re.sub("=+", "==", o) for (k, o) in ops.items()}
            return data_algebra.data_pipe.Extend(
                ops=ops,
                partition_by=maybe_get_list(obj, "partition_by"),
                order_by=maybe_get_list(obj, "order_by"),
                reverse=maybe_get_list(obj, "reverse"),
            )
        elif op == "Project":
            ops = obj["ops"]
            ops = {k: re.sub("=+", "==", o) for (k, o) in ops.items()}
            return data_algebra.data_pipe.Project(
                ops=ops, group_by=maybe_get_list(obj, "group_by")
            )
        elif op == "NaturalJoin":
            return data_algebra.data_pipe.NaturalJoin(
                by=maybe_get_list(obj, "by"),
                jointype=obj["jointype"],
                b=to_pipeline(obj["b"], known_tables=known_tables, parse_env=parse_env),
            )
        elif op == "ConcatRows":
            return data_algebra.data_pipe.ConcatRows(
                id_column=obj["id_column"],
                b=to_pipeline(obj["b"], known_tables=known_tables, parse_env=parse_env),
            )
        elif op == "SelectRows":
            expr = get_char_scalar(obj, "expr")
            expr = re.sub("=+", "==", expr)
            return data_algebra.data_pipe.SelectRows(expr=expr)
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
        elif op == "ConvertRecords":
            return data_algebra.data_pipe.ConvertRecords(
                record_map=data_algebra.cdata_impl.record_map_from_simple_obj(
                    obj["record_map"]
                )
            )
        else:
            raise TypeError("Unexpected op name: " + op)
    if isinstance(obj, list):
        # a pipeline, assumed non empty
        res = to_pipeline(obj[0], known_tables=known_tables, parse_env=parse_env)
        for i in range(1, len(obj)):
            nxt = to_pipeline(obj[i], known_tables=known_tables, parse_env=parse_env)
            # res = res >> nxt
            res = nxt.apply_to(res, parse_env=parse_env)
        return res
    raise TypeError("unexpected type: " + str(obj))
