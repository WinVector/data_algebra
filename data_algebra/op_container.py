
import collections
from types import SimpleNamespace

import data_algebra.data_ops_types
import data_algebra.data_ops


# for use in building expressions such as one.sum()
one = data_algebra.expr_rep.Value(1)


class OpC(data_algebra.data_ops_types.OperatorPlatform):
    """Container that redirects to another to non-quoted notation."""

    ops: data_algebra.data_ops_types.OperatorPlatform
    column_namespace: SimpleNamespace
    used_result: bool

    def __init__(self):
        self.ops = None
        self.column_namespace = SimpleNamespace()  # allows a dot notation
        self.used_result = False
        data_algebra.data_ops_types.OperatorPlatform.__init__(
            self, node_name="container", column_map=collections.OrderedDict()
        )

    def set(self, other):
        assert isinstance(other, data_algebra.data_ops_types.OperatorPlatform)
        assert not isinstance(other, OpC)  # don't allow deep nesting for now
        self.ops = other
        self.column_namespace.__dict__.clear()
        self.column_namespace.__dict__.update(self.ops.column_map)
        self.column_map.clear()
        self.column_map.update(self.ops.column_map)
        return self

    def start(self, other):
        assert self.ops is None
        return self.set(other)

    def get_ops(self):
        self.used_result = True
        return self.ops

    def ex(self, *, data_model=None, narrow=True, allow_limited_tables=False):
        """
        Evaluate operators with respect to Pandas data frames already stored in the operator chain.

        :param data_model: adaptor to data dialect (Pandas for now)
        :param narrow: logical, if True don't copy unexpected columns
        :param allow_limited_tables: logical, if True allow execution on non-complete tables
        :return: table result
        """
        self.used_result = True
        return self.ops.ex(data_model=data_model, narrow=narrow, allow_limited_tables=allow_limited_tables)

    # noinspection PyPep8Naming
    def transform(self, X, *, data_model=None, narrow=True):
        return self.ops.transform(X=X, data_model=data_model, narrow=narrow)

    # noinspection PyPep8Naming
    def act_on(self, X, *, data_model=None):
        self.set(self.ops.act_on(X=X, data_model=data_model))
        return self

    def apply_to(self, a, *, target_table_key=None):
        self.set(self.ops.apply_to(a=a, target_table_key=target_table_key))
        return self

    def __rrshift__(self, other):  # override other >> self
        self.set(self.ops.__rrshift__(other))
        return self

    def __rshift__(self, other):  # override self >> other
        self.set(self.ops.__rshift__(other))
        return self

    # composition
    def add(self, other):
        self.set(self.ops.add(other))
        return self

    # info

    def columns_produced(self):
        return self.ops.columns_produced()

    # query generation

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return self.ops.to_near_sql_implementation(
            db_model=db_model, using=using, temp_id_source=temp_id_source
        )

    # define builders for all non-initial ops types on base class

    def extend_parsed(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        self.set(
            self.ops.extend_parsed(
                parsed_ops=parsed_ops,
                partition_by=partition_by,
                order_by=order_by,
                reverse=reverse,
            )
        )
        return self

    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        self.set(
            self.ops.extend(
                ops=ops, partition_by=partition_by, order_by=order_by, reverse=reverse
            )
        )
        return self

    def project_parsed(self, parsed_ops=None, *, group_by=None):
        self.set(self.ops.project_parsed(parsed_ops=parsed_ops, group_by=group_by))
        return self

    def project(self, ops=None, *, group_by=None):
        self.set(self.ops.project(ops=ops, group_by=group_by, ))
        return self

    def natural_join(self, b, *, by, jointype, check_all_common_keys_in_by=False):
        self.set(self.ops.natural_join(
            b=b,
            by=by,
            jointype=jointype,
            check_all_common_keys_in_by=check_all_common_keys_in_by))
        return self

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        self.set(
            self.ops.concat_rows(b=b, id_column=id_column, a_name=a_name, b_name=b_name)
        )
        return self

    def select_rows_parsed(self, parsed_expr):
        self.set(self.ops.select_rows_parsed(parsed_expr=parsed_expr))
        return self

    def select_rows(self, expr):
        self.set(self.ops.select_rows(expr=expr, ))
        return self

    def drop_columns(self, column_deletions):
        self.set(self.ops.drop_columns(column_deletions=column_deletions))
        return self

    def select_columns(self, columns):
        self.set(self.ops.select_columns(columns=columns))
        return self

    def rename_columns(self, column_remapping):
        self.set(self.ops.rename_columns(column_remapping=column_remapping))
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        self.set(self.ops.order_rows(columns=columns, reverse=reverse, limit=limit))
        return self

    def convert_records(self, record_map, *, temp_namer=None):
        self.ops.convert_records(record_map=record_map, temp_namer=temp_namer)
        return self

    def map_records(
        self, blocks_in=None, blocks_out=None, strict=False, temp_namer=None
    ):
        self.ops.map_records(
            blocks_in=blocks_in,
            blocks_out=blocks_out,
            strict=strict,
            temp_namer=temp_namer,
            )
        return self

    # sklearn step style interface

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        self.ops.fit(X=X, y=y)
        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        return self.ops.fit_transform(X=X, y=y)

    def get_feature_names(self, input_features=None):
        return self.ops.get_feature_names(input_features=input_features)


# pop 0343 context manager
# https://www.python.org/dev/peps/pep-0343/#use-cases
class Pipeline:
    def __init__(self):
        self.container = OpC()

    def __enter__(self):
        return self.container, self.container.column_namespace

    def __exit__(self, *args):
        assert self.container is not None
        assert self.container.used_result
        self.container = None
