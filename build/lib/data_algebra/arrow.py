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
    def act_on(self, X):
        """ act on X, must associate with composition """
        raise NotImplementedError("base class called")

    # noinspection PyPep8Naming
    def transform(self, X):
        """ transform X, may or may not associate with composition """
        return self.act_on(X)

    def __rshift__(self, other):  # override self >> other
        return other.apply_to(self)

    def __rrshift__(self, other):  # override other >> self
        if isinstance(other, Arrow):
            return self.apply_to(other)
        return self.act_on(other)

    # sklearn step style interface

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        pass

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    # noinspection PyUnusedLocal
    def get_feature_names(self, input_features=None):
        raise NotImplementedError("base class called")

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=False):
        return dict()

    def set_params(self, **params):
        pass

    # noinspection PyPep8Naming
    def inverse_transform(self, X):
        raise TypeError("data_algebra does not support inverse_transform")


class DataOpArrow(Arrow):
    """
    Represent a dag of operators as a categorical arrow.

    """

    def __init__(
        self, pipeline, *, free_table_key=None, strict=False, forbidden_to_produce=None
    ):
        if not isinstance(pipeline, data_algebra.data_ops.ViewRepresentation):
            raise TypeError("expected pipeline to be data_algebra.data_ops")
        self.pipeline = pipeline
        self.strict = strict
        t_used = pipeline.get_tables()
        if forbidden_to_produce is None:
            forbidden_to_produce = []
        if isinstance(forbidden_to_produce, str):
            forbidden_to_produce = [forbidden_to_produce]
        self.forbidden_to_produce = forbidden_to_produce
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
        self.disallowed_columns = pipeline.forbidden_columns()[free_table_key]
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

    def get_feature_names(self, input_features=None):
        cp = self.outgoing_columns.copy()
        if (not self.strict) and (input_features is not None):
            cp = cp + [f for f in input_features if f not in cp]
        return cp

    def apply_to(self, b):
        """replace self input table with b"""
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            b = DataOpArrow(b)
        if not isinstance(b, DataOpArrow):
            raise TypeError("unexpected type: " + str(type(b)))
        # check categorical arrow composition conditions
        missing = set(self.incoming_columns) - set(b.outgoing_columns)
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        problem_production = set(self.forbidden_columns()) - set(b.forbidden_to_produce)
        if len(problem_production) > 0:
            raise ValueError(
                "did not document non-produciton of columns: " + str(problem_production)
            )
        excess = set(b.outgoing_columns) - set(self.incoming_columns)
        if len(excess) > 0:
            problem_excess = excess.intersection(self.forbidden_columns())
            if len(problem_excess) > 0:
                raise ValueError("forbidden incoming columns: " + str(excess))
            if self.strict:
                raise ValueError("extra incoming columns: " + str(excess))
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
        res = DataOpArrow(
            pipeline=new_pipeline,
            free_table_key=b.free_table_key,
            forbidden_to_produce=self.forbidden_to_produce,
        )
        res.incoming_types = b.incoming_types
        res.outgoing_types = self.outgoing_types
        return res

    # noinspection PyPep8Naming
    def act_on(self, X):
        # assume a pandas.DataFrame compatible object
        # noinspection PyUnresolvedReferences
        cols = set(X.columns)
        missing = set(self.incoming_columns) - cols
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        excess = cols - set(self.incoming_columns)
        if len(excess) > 0:
            X = X[self.incoming_columns]
        return self.pipeline.act_on(X)

    def learn_types(self, data_in, data_out):
        if (data_in is not None) and (data_in.shape[0] > 0):
            types_in = {k: type(data_in.loc[0, k]) for k in self.incoming_columns}
            self.incoming_types = types_in
        if (data_out is not None) and (data_out.shape[0] > 0):
            types_out = {k: type(data_out.loc[0, k]) for k in self.outgoing_columns}
            self.outgoing_types = types_out

    # noinspection PyPep8Naming
    def fit(self, X, y=None):
        """Learn input and output types from example, and return self"""
        # assume a pandas.DataFrame compatible object
        out = self.act_on(X)
        self.learn_types(X, out)
        return self

    # noinspection PyPep8Naming
    def fit_transform(self, X, y=None):
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

    def required_columns(self):
        return self.incoming_columns.copy()

    def forbidden_columns(self):
        return self.disallowed_columns.copy()

    # noinspection PyMethodMayBeStatic
    def format_end_description(
        self, *, required_cols, col_types, forbidden_cols, align_right=70, sep_width=2
    ):
        if col_types is not None:
            in_rep = [str(c) + ": " + str(col_types[c]) for c in required_cols]
        else:
            in_rep = [str(c) for c in required_cols]
        in_rep = data_algebra.flow_text.flow_text(
            in_rep, align_right=align_right, sep_width=sep_width
        )
        col_rep = [", ".join(line) for line in in_rep]
        col_rep = " at least [ " + ",\n    ".join(col_rep) + " ]"
        if (forbidden_cols is not None) and (len(forbidden_cols) > 0):
            f_rep = [str(c) for c in forbidden_cols]
            f_rep = data_algebra.flow_text.flow_text(
                f_rep, align_right=align_right, sep_width=sep_width
            )
            f_rep = [", ".join(line) for line in f_rep]
            col_rep = col_rep + " , and none of [ " + ",\n    ".join(f_rep) + " ]"
        return col_rep

    def __str__(self):
        in_rep = self.format_end_description(
            required_cols=self.incoming_columns,
            col_types=self.incoming_types,
            forbidden_cols=self.disallowed_columns,
        )
        out_rep = self.format_end_description(
            required_cols=self.outgoing_columns,
            col_types=self.outgoing_types,
            forbidden_cols=self.forbidden_to_produce,
        )
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


def fmt_as_arrow(ops):
    return str(DataOpArrow(ops))
