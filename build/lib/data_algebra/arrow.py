
import copy

import pandas

import data_algebra.data_ops
import data_algebra.flow_text


class Arrow:
    """Arrow from category theory: see Steve Awody, "Category Theory, 2nd Edition", Oxford Univ. Press, 2010 pg. 4."""

    def __init__(self):
        pass

    def dom(self):
        """return domain, object at base of arrow"""
        raise NotImplementedError("base class called")

    def cod(self):
        """return co-domain, object at head of arrow"""
        raise NotImplementedError("base class called")

    def identity_arrow(self, obj):
        """convert object to an identity arrow """
        raise NotImplementedError("base class called")

    # noinspection PyPep8Naming
    def transform(self, X, *, strict=True):
        """ transform X, compose arrows (right to left) """
        raise NotImplementedError("base class called")

    def __rshift__(self, other):  # override self >> other
        return other.transform(self, strict=True)

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other, strict=True)





class DataOpArrow(Arrow):
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
        Arrow.__init__(self)

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
            res = DataOpArrow(X.pipeline._r_copy_replace(self.pipeline))
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

    def __repr__(self):
        return "DataOpArrow(" + self.pipeline.__repr__() + ")"

    def dom(self):
        if self.incoming_types is not None:
            return self.incoming_types.copy()
        return self.incoming_columns.copy()

    def cod(self):
        if self.incoming_types is not None:
            return self.outgoing_types.copy()
        return self.outgoing_columns.copy()

    def identity_arrow(self, obj):
        """build identity arrow from object"""
        if isinstance(obj, data_algebra.data_ops.TableDescription):
            return DataOpArrow(obj)
        if isinstance(obj, list) or isinstance(obj, tuple):
            td = data_algebra.data_ops.TableDescription("obj", obj)
            return DataOpArrow(td)
        if isinstance(obj, dict):
            td = data_algebra.data_ops.TableDescription("obj", obj.keys(), column_types=obj)
            res = DataOpArrow(td)
            res.outgoing_types = obj.copy()
            return res
        raise TypeError("unexpected type: " + str(type(obj)))

    def __str__(self):
        align_right = 70
        sep_width = 2
        if self.incoming_types is not None:
            in_rep = [str(k) + ': ' + str(v) for (k, v) in self.incoming_types.items()]
        else:
            in_rep = [str(c) for c in self.incoming_columns]
        in_rep = data_algebra.flow_text.flow_text(in_rep,
                                                  align_right=align_right, sep_width=sep_width)
        in_rep = [', '.join(line) for line in in_rep]
        in_rep = ' [ ' + ',\n    '.join(in_rep) + ' ]'
        if self.outgoing_types is not None:
            out_rep = [str(k) + ': ' + str(v) for (k, v) in self.outgoing_types.items()]
        else:
            out_rep = [str(c) for c in self.outgoing_columns]
        out_rep = data_algebra.flow_text.flow_text(out_rep,
                                                   align_right=align_right, sep_width=sep_width)
        out_rep = [', '.join(line) for line in out_rep]
        out_rep = ' [ ' + ',\n    '.join(out_rep) + ' ]'
        return "[\n " + in_rep + "\n   ->\n " + out_rep + "\n]\n"
