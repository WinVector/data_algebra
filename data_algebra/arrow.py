
import copy

import pandas

import data_algebra.data_ops



class DataOpArrow:
    """ Represent a section of operators as a categorical arrow."""

    def __init__(self, v):
        if not isinstance(v, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("expected v to be data_algebra.data_ops")
        self.v = v
        cused = v.columns_used()
        if len(cused) != 1:
            raise ValueError("v must use exactly one table")
        k = [k for k in cused.keys()][0]
        self.incoming_columns = cused[k]
        self.outgoing_columns = v.column_names

    def _r_copy_replace(self, ops):
        """re-write ops replacing any TableDescription with self.v"""
        if isinstance(ops, data_algebra.data_ops.TableDescription):
            return self.v
        node = copy.copy(ops)
        node.sources = [self._r_copy_replace(s) for s in node.sources]
        return node

    def transform(self, other):
        """replace self input table with other"""
        if isinstance(other, pandas.DataFrame):
            return self.v.transform(other)
        if isinstance(other, data_algebra.data_ops.ViewRepresentation):
            other = DataOpArrow(other)
        if not isinstance(other, DataOpArrow):
            raise TypeError("other must be a DataOpArrow")
        if set(self.incoming_columns) != set(other.outgoing_columns):
            raise TypeError("arrow composition conditions not met (incoming columsn don't match outgoing)")
        return DataOpArrow(other._r_copy_replace(self.v))

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __rshift__(self, other):  # override self >> other
        return other.transform(self)

    def __repr__(self):
        return "DataOpArrow(" + self.v.__repr__() + ")"

    def __str__(self):
        return "[" + str(self.incoming_columns) + " -> " + str(self.outgoing_columns) + "]"
