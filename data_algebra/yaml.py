import collections

have_yaml = False
try:
    import yaml   # supplied by PyYAML
    have_yaml = True
except ImportError:
    pass

import data_algebra.data_ops

# yaml notes:
#    https://stackoverflow.com/questions/2627555/how-to-deserialize-an-object-with-pyyaml-using-safe-load


def fix_ordered_dict_yaml_rep():
    """Writer OrderedDict as simple structure"""
    # derived from: https://stackoverflow.com/a/16782282/6901725
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
    if isinstance(obj, dict):
        # a pipe stage
        op = obj["op"]
        # ugly switch statement
        if op == "TableDescription":
            tab = data_algebra.data_ops.TableDescription(
                table_name=obj["table_name"],
                column_names=obj["column_names"],
                qualifiers=obj["qualifiers"],
            )
            # canonicalize to one object per table
            k = tab.key
            if k in known_tables.keys():
                ov = known_tables[k]
                if tab.column_set != ov.column_set:
                    raise Exception(
                        "two tables with same qualified table names and different declared column sets"
                    )
                tab = ov
            else:
                known_tables[k] = tab
            return tab
        elif op == "Extend":
            return data_algebra.data_ops.Extend(
                ops=obj["ops"],
                partition_by=obj["partition_by"],
                order_by=obj["order_by"],
                reverse=obj["reverse"])
        elif op == "NaturalJoin":
            return data_algebra.data_ops.NaturalJoin(
                by=obj["by"],
                jointype=obj["jointype"],
                b=to_pipeline(obj["b"], known_tables=known_tables),
            )
        elif op == "SelectRows":
            return data_algebra.data_ops.SelectRows(
                expr=obj['expr']
            )
        elif op == "SelectColumns":
            return data_algebra.data_ops.SelectColumns(
                columns=obj['columns']
            )
        elif op == "Rename":
            return data_algebra.data_ops.RenameColumns(
                column_remapping=obj['column_remapping']
            )
        elif op == "Order":
            return data_algebra.data_ops.OrderRows(
                columns=obj['order_columns'],
                reverse=obj['reverse'],
                limit=obj['limit']
            )
        else:
            raise Exception("Unexpected op name: " + op)
    if isinstance(obj, list):
        # a pipeline, assumed non empty
        res = to_pipeline(obj[0], known_tables=known_tables)
        for i in range(1, len(obj)):
            nxt = to_pipeline(obj[i], known_tables=known_tables)
            res = res >> nxt
        return res
    raise Exception("unexpected type: " + str(obj))
