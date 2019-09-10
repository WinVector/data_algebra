import re
import collections

import pandas

import data_algebra.data_types


class RecordSpecification:
    def __init__(self, control_table, *, record_keys=None, control_table_keys=None):
        control_table = data_algebra.data_types.convert_to_pandas_dataframe(control_table, "control_table")
        self.control_table = control_table.copy()
        self.control_table.reset_index(inplace=True, drop=True)
        if record_keys is None:
            record_keys = []
        self.record_keys = [k for k in record_keys]
        if control_table_keys is None:
            control_table_keys = [control_table.columns[0]]
        self.control_table_keys = [k for k in control_table_keys]
        confused = set(record_keys).intersection(control_table_keys)
        if len(confused) > 0:
            raise ValueError(
                "columns common to record_keys and control_table_keys: " + str(confused)
            )

    def row_version(self, *, include_record_keys=True):
        cols = []
        if include_record_keys:
            cols = cols + self.record_keys
        for ci in self.control_table.columns:
            if ci not in self.control_table_keys:
                for c in self.control_table[ci]:
                    cols.append(c)
        return cols

    def fmt(self):
        s = (
            "RecordSpecification\n"
            + "   record_keys: "
            + str(self.record_keys)
            + "\n"
            + "   control_table_keys: "
            + str(self.control_table_keys)
            + "\n"
            + "   control_table:\n"
            + "   "
            + re.sub("\n", "\n   ", str(self.control_table))
            + "\n"
        )
        return s

    def __repr__(self):
        return self.fmt()

    def __str__(self):
        return self.fmt()

    def to_simple_obj(self):
        """Create an object for YAML encoding"""

        obj = collections.OrderedDict()
        obj["type"] = "data_algebra.cdata.RecordSpecification"
        obj["record_keys"] = self.record_keys.copy()
        obj["control_table_keys"] = self.control_table_keys.copy()
        tbl = collections.OrderedDict()
        for k in self.control_table.columns:
            tbl[k] = [v for v in self.control_table[k]]
        obj["control_table"] = tbl
        return obj


def record_spec_from_simple_obj(obj):
    control_table = pandas.DataFrame()
    for k in obj["control_table"].keys():
        control_table[k] = obj["control_table"][k]
    return RecordSpecification(
        control_table,
        record_keys=obj["record_keys"],
        control_table_keys=obj["control_table_keys"],
    )
