
import copy

import pandas

import data_algebra.data_ops


class DataOpArrow:
    """ Represent a section of operators as a categorical arrow."""

    def __init__(self, pipeline):
        if not isinstance(pipeline, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("expected pipeline to be data_algebra.data_ops")
        self.pipeline = pipeline
        t_used = pipeline.get_tables()
        if len(t_used) != 1:
            raise ValueError("pipeline must use exactly one table")
        k = [k for k in t_used.keys()][0]
        c_used = pipeline.columns_used()
        if len(c_used) != 1:
            raise ValueError("pipeline must use exactly one table")
        self.incoming_columns = c_used[k]
        self.incoming_types = t_used[k].column_types
        self.outgoing_columns = pipeline.column_names
        self.outgoing_types = None

    def _r_copy_replace(self, ops):
        """re-write ops replacing any TableDescription with self.pipeline"""
        if isinstance(ops, data_algebra.data_ops.TableDescription):
            return self.pipeline
        node = copy.copy(ops)
        node.sources = [self._r_copy_replace(s) for s in node.sources]
        return node

    # noinspection PyPep8Naming
    def transform(self, X, *, strict=True):
        """replace self input table with X"""
        if isinstance(X, data_algebra.data_ops.ViewRepresentation):
            X = DataOpArrow(X)
        if isinstance(X, DataOpArrow):
            missing = set(self.incoming_columns) - set(X.outgoing_columns)
            if len(missing) > 0:
                raise ValueError("missing required columns: " + str(missing))
            excess = set(X.outgoing_columns) - set(self.incoming_columns)
            if len(excess):
                if strict:
                    raise ValueError("extra incoming columns: " + str(excess))
                # extra columns, in a strict categorical formulation we would
                # reject this. instead insert a select columns node to get the match
                x_outgoing_types = X.outgoing_types
                X = DataOpArrow(X.pipeline.select_columns([c for c in self.incoming_columns]))
                if x_outgoing_types is not None:
                    X.outgoing_types = {k: x_outgoing_types[k] for k in X.outgoing_columns}
            # check categorical arrow composition conditions
            if set(self.incoming_columns) != set(X.outgoing_columns):
                raise ValueError("arrow composition conditions not met (incoming column set doesn't match outgoing)")
            if (self.incoming_types is not None) and (X.outgoing_types is not None):
                for c in self.incoming_columns:
                    st = self.incoming_types[c]
                    xt = X.outgoing_types[c]
                    if st != xt:
                        raise ValueError("column " + c +
                                         " self incoming type is " + str(st) +
                                         ", while X outgoing type is " + str(xt))
            res = DataOpArrow(X._r_copy_replace(self.pipeline))
            res.incoming_types = X.incoming_types
            res.outgoing_types = self.outgoing_types
            return res
        # assume a pandas.DataFrame compatible object
        cols = set(X.columns)
        missing = set(self.incoming_columns) - cols
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        excess = cols - set(self.incoming_columns)
        if len(excess):
            if strict:
                raise ValueError("extra incoming columns: " + str(excess))
            X = X[self.incoming_columns]
        return self.pipeline.transform(X)

    def learn_types(self, data_in, data_out):
        if(data_in.shape[0]>0):
            types_in = {k: type(data_in.loc[0, k]) for k in self.incoming_columns}
            self.incoming_types = types_in
        if (data_out.shape[0] > 0):
            types_out = {k: type(data_out.loc[0, k]) for k in self.outgoing_columns}
            self.outgoing_types = types_out

    # noinspection PyPep8Naming
    def fit(self, X):
        """Learn input and output types from example, and return self"""
        # assume a pandas.DataFrame compatible object
        out = self.transform(X)
        self.learn_types(X, out)
        return self

    # noinspection PyPep8Naming
    def fit_transform(self, X):
        """Learn input and output types from example, and return transform."""
        out = self.transform(X)
        self.learn_types(X, out)
        return self.transform(X)

    def __rshift__(self, other):  # override self >> other
        return other.transform(self, strict=True)

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other, strict=True)

    def __repr__(self):
        return "DataOpArrow(" + self.pipeline.__repr__() + ")"

    def __str__(self):
        if self.incoming_types is not None:
            in_rep = str(self.incoming_types)
        else:
            in_rep = str([c for c in self.incoming_columns])
        if self.outgoing_types is not None:
            out_rep = str(self.outgoing_types)
        else:
            out_rep = str([c for c in self.outgoing_columns])
        return "[" + in_rep + " -> " + out_rep + "]"
