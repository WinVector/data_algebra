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
    def transform(self, X):
        """
        apply self to data frame X

        :param X: input data frame
        :return: transformed dataframe
        """
        raise NotImplementedError("base class called")

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
        self.apply_to/transform(other)

        :param other:
        :return:
        """
        if isinstance(other, OperatorPlatform):
            return self.apply_to(other)
        if isinstance(other, PipeStep):
            return other.apply_to(other)
        return self.transform(other)

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

    def natural_join(self, b, *, by=None, jointype="INNER"):
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

    def convert_records(self, record_map, *, blocks_out_table=None):
        raise NotImplementedError("base class called")
