import re
import pandas

import data_algebra.data_ops


class RecordSpecification:

    def __init__(self,
                 control_table,
                 *,
                 record_keys=None,
                 control_table_keys=None):
        if not isinstance(control_table, pandas.DataFrame):
            raise Exception("control_table should be a pandas.DataFrame")
        self.control_table = control_table.copy()
        self.control_table.reset_index(inplace=True, drop=True)
        if record_keys is None:
            record_keys = []
        self.record_keys = [k for k in record_keys]
        if control_table_keys is None:
            control_table_keys = [control_table.columns[0]]
        self.control_table_keys = [k for k in control_table_keys]

    def __repr__(self):
        s = (
                'RecordSpecification\n'
                + '   record_keys: ' + str(self.record_keys) + '\n'
                + '   control_table_keys: ' + str(self.control_table_keys) + '\n'
                + '   control_table:\n'
                + '   ' + re.sub('\n', '\n   ', str(self.control_table))
        )
        return s

    def __str__(self):
        s = (
                'RecordSpecification\n'
                + '   record_keys: ' + str(self.record_keys) + '\n'
                + '   control_table_keys: ' + str(self.control_table_keys) + '\n'
                + '   control_table:\n'
                + '   ' + re.sub('\n', '\n   ', str(self.control_table))
        )
        return s
