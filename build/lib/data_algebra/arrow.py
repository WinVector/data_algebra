import abc

import data_algebra.data_ops
import data_algebra.flow_text
from data_algebra.shift_pipe_action import ShiftPipeAction


class Arrow(ShiftPipeAction):
    """
    Arrow from category theory: see Steve Awody,
    "Category Theory, 2nd Edition", Oxford Univ. Press, 2010 pg. 4.
    Essentially this is a tool to expose associativity, without forcing
    things to be a function to expose this effect.
    """

    def __init__(self):
        ShiftPipeAction.__init__(self)

    @abc.abstractmethod
    def dom(self):
        """return domain, object at base of arrow"""

    @abc.abstractmethod
    def cod(self):
        """return co-domain, object at head of arrow"""

    # noinspection PyPep8Naming
    @abc.abstractmethod
    def act_on(self, b):
        """act on b, must associate with composition"""

    # noinspection PyPep8Naming
    def transform(self, X):
        """transform X, may or may not associate with composition"""
        return self.act_on(X)


class DataOpArrow(Arrow):
    """
    Represent a dag of operators as a categorical arrow.

    """

    def __init__(self, pipeline, *, free_table_key=None):
        assert isinstance(pipeline, data_algebra.data_ops.ViewRepresentation)
        self.pipeline = pipeline
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
        self.incoming_columns = list(t_used[free_table_key].column_names)
        self.outgoing_columns = list(pipeline.column_names)
        self.outgoing_columns.sort()
        Arrow.__init__(self)

    def get_feature_names(self):
        cp = self.outgoing_columns.copy()
        return cp

    def act_on(self, b, *, correct_ordered_first_call: bool = False):
        """
        Apply self onto b.

        :param b: item to act on, or item that has been sent to self.
        :param correct_ordered_first_call: if True indicates this call is from __rshift__ or __rrshift__ and not the fallback paths.
        """
        assert isinstance(correct_ordered_first_call, bool)
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            b = DataOpArrow(b)
        if isinstance(b, DataOpArrow):
            # check categorical arrow composition conditions
            missing = set(self.incoming_columns) - set(b.outgoing_columns)
            if len(missing) > 0:
                raise ValueError("missing required columns: " + str(missing))
            excess = set(b.outgoing_columns) - set(self.incoming_columns)
            if len(excess) > 0:
                raise ValueError("extra incoming columns: " + str(excess))
            new_pipeline = self.pipeline.replace_leaves(
                {self.free_table_key: b.pipeline}
            )
            new_pipeline.get_tables()  # check tables are compatible
            res = DataOpArrow(
                pipeline=new_pipeline,
                free_table_key=b.free_table_key,
            )
            return res
        if correct_ordered_first_call and isinstance(b, ShiftPipeAction):
            return b.act_on(self, correct_ordered_first_call=False)  # fall back
        # assume a pandas.DataFrame compatible object
        # noinspection PyUnresolvedReferences
        cols = set(b.columns)
        missing = set(self.incoming_columns) - cols
        if len(missing) > 0:
            raise ValueError("missing required columns: " + str(missing))
        excess = cols - set(self.incoming_columns)
        assert len(excess) == 0
        if len(excess) > 0:
            b = b[self.incoming_columns]
        return self.pipeline.act_on(b)

    def dom(self):
        return DataOpArrow(
            data_algebra.data_ops.TableDescription(
                table_name=None,
                column_names=self.incoming_columns,
            )
        )

    def dom_as_table(self):
        return data_algebra.data_ops.TableDescription(
            table_name=None,
            column_names=self.incoming_columns,
        )

    def cod(self):
        return DataOpArrow(
            data_algebra.data_ops.TableDescription(
                table_name=None,
                column_names=self.outgoing_columns,
            )
        )

    def cod_as_table(self):
        return data_algebra.data_ops.TableDescription(
            table_name=None,
            column_names=self.outgoing_columns,
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

    # noinspection PyMethodMayBeStatic
    def format_end_description(self, *, required_cols, align_right=70, sep_width=2):
        in_rep = [str(c) for c in required_cols]
        in_rep = data_algebra.flow_text.flow_text(
            in_rep, align_right=align_right, sep_width=sep_width
        )
        col_rep = [", ".join(line) for line in in_rep]
        col_rep = " [ " + ",\n    ".join(col_rep) + " ]"
        return col_rep

    def __str__(self):
        in_rep = self.format_end_description(
            required_cols=self.incoming_columns,
        )
        out_rep = self.format_end_description(
            required_cols=self.outgoing_columns,
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
        assert isinstance(other, DataOpArrow)
        if self.free_table_key != other.free_table_key:
            return False
        if self.incoming_columns != other.incoming_columns:
            return False
        if self.outgoing_columns != other.outgoing_columns:
            return False
        return self.pipeline == other.pipeline

    def __ne__(self, other):
        return not self.__eq__(other)


def fmt_as_arrow(ops) -> str:
    return str(DataOpArrow(ops))
