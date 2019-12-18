import data_algebra
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

    def apply_to(self, b):
        """ apply_to b, compose arrows (right to left) """
        raise NotImplementedError("base class called")

    # noinspection PyPep8Naming
    def transform(self, X):
        """ transform X, act on X """
        raise NotImplementedError("base class called")

    def __rshift__(self, other):  # override self >> other
        return other.apply_to(self)

    def __rrshift__(self, other):  # override other >> self
        if isinstance(other, Arrow):
            return self.apply_to(other)
        return self.transform(other)


class DataOpArrow(Arrow):
    """
    Represent a dag of operators as a categorical arrow.

    """

    def __init__(self, pipeline, *, free_table_key=None, strict=False):
        if not isinstance(pipeline, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("expected pipeline to be data_algebra.data_ops")
        self.pipeline = pipeline
        self.strict = strict
        t_used = pipeline.get_tables()
        if free_table_key is None:
            if len(t_used) != 1:
                raise ValueError(
                    "pipeline must use exactly one table if free_table_key is not specified"
                )
            free_table_key = [k for k in t_used.keys()][0]
        else:
            if free_table_key not in t_used.keys():
                raise ValueError(
                    "free_table_key must be a table key used in the pipeline"
                )
        self.free_table_key = free_table_key
        self.incoming_columns = t_used[free_table_key].column_names.copy()
        self.incoming_types = None
        if t_used[free_table_key].column_types is not None:
            self.incoming_types = t_used[free_table_key].column_types.copy()
        self.outgoing_columns = pipeline.column_names.copy()
        self.outgoing_columns.sort()
        self.outgoing_types = None
        if (
            isinstance(pipeline, data_algebra.data_ops.TableDescription)
            and self.incoming_types is not None
        ):
            self.outgoing_types = self.incoming_types.copy()
        Arrow.__init__(self)

    def apply_to(self, b):
        """replace self input table with b"""
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            b = DataOpArrow(b)
        if not isinstance(b, DataOpArrow):
            raise TypeError("unexpected type: " + str(type(b)))
        missing = set(self.incoming_columns) - set(b.outgoing_columns)
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        if self.strict:
            excess = set(b.outgoing_columns) - set(self.incoming_columns)
            if len(excess) > 0:
                raise ValueError("extra incoming columns: " + str(excess))
        # check categorical arrow composition conditions
        if set(self.incoming_columns) != set(b.outgoing_columns):
            raise ValueError(
                "arrow composition conditions not met (incoming column set doesn't match outgoing)"
            )
        if (self.incoming_types is not None) and (b.outgoing_types is not None):
            for c in self.incoming_columns:
                st = self.incoming_types[c]
                xt = b.outgoing_types[c]
                if st != xt:
                    raise ValueError(
                        "column "
                        + c
                        + " self incoming type is "
                        + str(st)
                        + ", while b outgoing type is "
                        + str(xt)
                    )
        new_pipeline = self.pipeline.apply_to(
            b.pipeline, target_table_key=self.free_table_key
        )
        new_pipeline.get_tables()  # check tables are compatible
        res = DataOpArrow(pipeline=new_pipeline, free_table_key=b.free_table_key)
        res.incoming_types = b.incoming_types
        res.outgoing_types = self.outgoing_types
        return res

    # noinspection PyPep8Naming
    def transform(self, X):
        # assume a pandas.DataFrame compatible object
        # noinspection PyUnresolvedReferences
        cols = set(X.columns)
        missing = set(self.incoming_columns) - cols
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        excess = cols - set(self.incoming_columns)
        if len(excess) > 0:
            X = X[self.incoming_columns]
        return self.pipeline.transform(X)

    def learn_types(self, data_in, data_out):
        if (data_in is not None) and (data_in.shape[0] > 0):
            types_in = {k: type(data_in.loc[0, k]) for k in self.incoming_columns}
            self.incoming_types = types_in
        if (data_out is not None) and (data_out.shape[0] > 0):
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

    def dom(self):
        return DataOpArrow(
            data_algebra.data_ops.TableDescription(
                table_name=None,
                column_names=self.incoming_columns,
                column_types=self.incoming_types,
            )
        )

    def dom_as_table(self):
        return data_algebra.data_ops.TableDescription(
            table_name=None,
            column_names=self.incoming_columns,
            column_types=self.incoming_types,
        )

    def cod(self):
        return DataOpArrow(
            data_algebra.data_ops.TableDescription(
                table_name=None,
                column_names=self.outgoing_columns,
                column_types=self.outgoing_types,
            )
        )

    def cod_as_table(self):
        return data_algebra.data_ops.TableDescription(
            table_name=None,
            column_names=self.outgoing_columns,
            column_types=self.outgoing_types,
        )

    def __repr__(self):
        return (
            "DataOpArrow(\n "
            + self.pipeline.__repr__()
            + ",\n free_table_key="
            + self.free_table_key.__repr__()
            + ")"
        )

    def __str__(self):
        align_right = 70
        sep_width = 2
        if self.incoming_types is not None:
            in_rep = [str(k) + ": " + str(v) for (k, v) in self.incoming_types.items()]
        else:
            in_rep = [str(c) for c in self.incoming_columns]
        in_rep = data_algebra.flow_text.flow_text(
            in_rep, align_right=align_right, sep_width=sep_width
        )
        in_rep = [", ".join(line) for line in in_rep]
        in_rep = " [ " + ",\n    ".join(in_rep) + " ]"
        if self.outgoing_types is not None:
            out_rep = [str(k) + ": " + str(v) for (k, v) in self.outgoing_types.items()]
        else:
            out_rep = [str(c) for c in self.outgoing_columns]
        out_rep = data_algebra.flow_text.flow_text(
            out_rep, align_right=align_right, sep_width=sep_width
        )
        out_rep = [", ".join(line) for line in out_rep]
        out_rep = " [ " + ",\n    ".join(out_rep) + " ]"
        return (
            "[\n "
            + self.free_table_key.__repr__()
            + ":\n "
            + in_rep
            + "\n   ->\n "
            + out_rep
            + "\n]\n"
        )

    def __eq__(self, other):
        if not isinstance(other, DataOpArrow):
            return False
        if self.free_table_key != other.free_table_key:
            return False
        if self.incoming_columns != other.incoming_columns:
            return False
        if self.incoming_types != other.incoming_types:
            return False
        if self.outgoing_columns != other.outgoing_columns:
            return False
        if self.outgoing_types != other.outgoing_types:
            return False
        return self.pipeline == other.pipeline

    def __ne__(self, other):
        return not self.__eq__(other)
