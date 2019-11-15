

class PipeStep:
    def __init__(self):
        pass

    def apply(self, other, **kwargs):
        raise NotImplementedError("base class called")


class OperatorPlatform:
    """Abstract class representing ability to apply data_algebra operations."""

    def __init__(self, *, node_name):
        self.node_name = node_name

    # noinspection PyPep8Naming
    def transform(self, X):
        raise NotImplementedError("base class called")

    def __rrshift__(self, other):  # override other >> self
        return self.transform(other)

    def __rshift__(self, other):  # override self >> other
        # can't use type >> type if only __rrshift__ is defined (must have __rshift__ in this case)
        if isinstance(other, OperatorPlatform):
            return other.transform(self)
        if isinstance(other, PipeStep):
            other.apply(self)
        raise TypeError("unexpected type: " + str(type(other)))

    # composition
    def add(self, other):
        """interface to what we used to call PipeStep nodes"""
        return other.apply(self)

    # query generation

    def to_sql_implementation(self, db_model, *, using, temp_id_source):
        raise NotImplementedError("base method called")

    # define builders for all non-initial node types on base class

    def extend(
        self, ops, *, partition_by=None, order_by=None, reverse=None, parse_env=None
    ):
        raise NotImplementedError("base class called")

    def project(self, ops=None, *, group_by=None, parse_env=None):
        raise NotImplementedError("base class called")

    def natural_join(self, b, *, by=None, jointype="INNER"):
        raise NotImplementedError("base class called")

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
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
