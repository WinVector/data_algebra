
import copy

import pandas

import data_algebra.data_ops



class DataOpArrow:
    """ Represent a section of operators as a categorical arrow."""

    def __init__(self, pipeline):
        if not isinstance(pipeline, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("expected pipeline to be data_algebra.data_ops")
        self.pipeline = pipeline
        cused = pipeline.columns_used()
        if len(cused) != 1:
            raise ValueError("pipeline must use exactly one table")
        k = [k for k in cused.keys()][0]
        self.incoming_columns = cused[k]
        self.outgoing_columns = pipeline.column_names

    def _r_copy_replace(self, ops):
        """re-write ops replacing any TableDescription with self.pipeline"""
        if isinstance(ops, data_algebra.data_ops.TableDescription):
            return self.pipeline
        node = copy.copy(ops)
        node.sources = [self._r_copy_replace(s) for s in node.sources]
        return node

    def transform(self, other, *, strict=True):
        """replace self input table with other"""
        if isinstance(other, pandas.DataFrame):
            cols = set(other.columns)
            missing = set(self.incoming_columns) - cols
            if len(missing) > 0:
                raise ValueError("missing required columns: " + str(missing))
            excess = cols - set(self.incoming_columns)
            if len(excess):
                if strict:
                    raise ValueError("extra incoming columns: " + str(excess))
                other = other[self.incoming_columns]
            return self.pipeline.transform(other)
        if isinstance(other, data_algebra.data_ops.ViewRepresentation):
            other = DataOpArrow(other)
        if not isinstance(other, DataOpArrow):
            raise TypeError("other must be a DataOpArrow")
        missing = set(self.incoming_columns) - set(other.outgoing_columns)
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        excess = set(other.outgoing_columns) - set(self.incoming_columns)
        if len(excess):
            if strict:
                raise ValueError("extra incoming columns: " + str(excess))
            # extra columns, in a strict categorical formulation we would
            # reject this. instead insert a select columns node to get the match
            other = DataOpArrow(other.pipeline.select_columns([c for c in self.incoming_columns]))
        # check categorical arrow composition conditions
        if set(self.incoming_columns) != set(other.outgoing_columns):
            raise ValueError("arrow composition conditions not met (incoming column set doesn't match outgoing)")
        return DataOpArrow(other._r_copy_replace(self.pipeline))

    def __rshift__(self, other):  # override self >> other
        return other.transform(self)

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __repr__(self):
        return "DataOpArrow(" + self.pipeline.__repr__() + ")"

    def __str__(self):
        return "[" + str(self.incoming_columns) + " -> " + str(self.outgoing_columns) + "]"
