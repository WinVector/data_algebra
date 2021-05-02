
import data_algebra.cdata


class PipeStep:
    def __init__(self):
        pass

    def apply_to(self, other, **kwargs):
        raise NotImplementedError("base class called")


class OperatorPlatform:
    """Abstract class representing ability to apply data_algebra operations."""

    def __init__(self, *, node_name):
        self.node_name = node_name

    # noinspection PyPep8Naming
    def transform(self, X, *, eval_env=None, data_model=None, narrow=True):
        """
        apply self to data frame X, may or may not commute with composition

        :param X: input data frame
        :param eval_env environment to look for symbols in
        :param data_model implementation to use
        :param narrow logical, if True don't copy unexpected columns
        :return: transformed dataframe
        """
        raise NotImplementedError("base class called")

    # noinspection PyPep8Naming
    def act_on(self, X, *, eval_env=None, data_model=None):
        """
        apply self to data frame X, must commute with composition

        :param X: input data frame
        :param eval_env environment to look for symbols in
        :param data_model implementation to use
        :return: transformed dataframe
        """
        return self.transform(
            X=X, eval_env=eval_env, data_model=data_model, narrow=False
        )

    def apply_to(self, a, *, target_table_key=None):
        """
        apply self to operator DAG a

        :param a: operators to apply to
        :param target_table_key: table key to replace with self, None counts as "match all"
        :return: new operator DAG
        """
        raise NotImplementedError("base class called")

    def __rrshift__(self, other):  # override other >> self
        """
        override other >> self
        self.apply_to/act_on(other)

        :param other:
        :return:
        """
        if isinstance(other, OperatorPlatform):
            return self.apply_to(other)
        if isinstance(other, PipeStep):
            return other.apply_to(other)
        return self.act_on(other)

    def __rshift__(self, other):  # override self >> other
        """
        override self >> other
        other.apply_to(self)

        :param other:
        :return:
        """
        # can't use type >> type if only __rrshift__ is defined (must have __rshift__ in this case)
        if isinstance(other, OperatorPlatform):
            return other.apply_to(self)
        if isinstance(other, PipeStep):
            return other.apply_to(self)
        raise TypeError("unexpected type: " + str(type(other)))

    # composition
    def add(self, other):
        """
        other.apply_to(self)

        :param other:
        :return:
        """
        return other.apply_to(self)

    # info

    def columns_produced(self):
        raise NotImplementedError("base class called")

    # query generation

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        raise NotImplementedError("base method called")

    # define builders for all non-initial node types on base class

    def extend_parsed(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        raise NotImplementedError("base class called")

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        raise NotImplementedError("base class called")

    def project_parsed(self, parsed_ops=None, *, group_by=None):
        raise NotImplementedError("base class called")

    def project(self, ops=None, *, group_by=None, parse_env=None):
        raise NotImplementedError("base class called")

    def natural_join(self, b, *, by, jointype):
        raise NotImplementedError("base class called")

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        raise NotImplementedError("base class called")

    def select_rows_parsed(self, parsed_expr):
        raise NotImplementedError("base class called")

    def select_rows(self, expr, *, parse_env=None):
        raise NotImplementedError("base class called")

    def drop_columns(self, column_deletions):
        raise NotImplementedError("base class called")

    def select_columns(self, columns):
        raise NotImplementedError("base class called")

    def rename_columns(self, column_remapping):
        raise NotImplementedError("base class called")

    def order_rows(self, columns, *, reverse=None, limit=None):
        raise NotImplementedError("base class called")

    def convert_records(self, record_map, *, temp_namer=None):
        raise NotImplementedError("base class called")

    def map_records(self, blocks_in=None, blocks_out=None, strict=False, temp_namer=None):
        if (blocks_in is None) and (blocks_out is None):
            return self  # NO-OP, return source node
        return self.convert_records(
            data_algebra.cdata.RecordMap(
                blocks_in=blocks_in,
                blocks_out=blocks_out,
                strict=strict),
                temp_namer=temp_namer)

    # sklearn step style interface

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        pass

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def get_feature_names(self, input_features=None):
        cp = self.columns_produced()
        if input_features is not None:
            cp = cp + [f for f in input_features if f not in cp]
        return cp

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_params(self, deep=False):
        return dict()

    def set_params(self, **params):
        pass

    # noinspection PyPep8Naming
    def inverse_transform(self, X):
        raise TypeError("data_algebra does not support inverse_transform")
