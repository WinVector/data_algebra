
import collections
from types import SimpleNamespace

import data_algebra.data_ops_types
import data_algebra.data_ops


# for use in building expressions such as one.sum()
one = data_algebra.expr_rep.Value(1)


class OpC(data_algebra.data_ops_types.OperatorPlatform):
    """Container that redirects to another to allow method chaining."""

    node: data_algebra.data_ops_types.OperatorPlatform
    column_namespace: SimpleNamespace

    def __init__(self):
        self.node = None
        self.column_namespace = SimpleNamespace()  # allows a dot notation
        data_algebra.data_ops_types.OperatorPlatform.__init__(
            self, node_name="container", column_map=collections.OrderedDict()
        )

    def set(self, other):
        assert isinstance(other, data_algebra.data_ops_types.OperatorPlatform)
        assert not isinstance(other, OpC)  # don't allow deep nesting for now
        self.node = other
        self.column_namespace.__dict__.clear()
        self.column_namespace.__dict__.update(self.node.column_map)
        self.column_map.clear()
        self.column_map.update(self.node.column_map)
        return self

    def describe_table(
        self,
        d,
        table_name="data_frame",
        *,
        qualifiers=None,
        sql_meta=None,
        column_types=None,
        row_limit=7
    ):
        td = data_algebra.data_ops.describe_table(
            d=d,
            table_name=table_name,
            qualifiers=qualifiers,
            sql_meta=sql_meta,
            column_types=column_types,
            row_limit=row_limit,
        )
        return self.set(td)

    def ops(self):
        return self.node

    # noinspection PyPep8Naming
    def transform(self, X, *, data_model=None, narrow=True):
        return self.node.transform(X=X, data_model=data_model, narrow=narrow)

    # noinspection PyPep8Naming
    def act_on(self, X, *, data_model=None):
        self.set(self.node.act_on(X=X, data_model=data_model))
        return self

    def apply_to(self, a, *, target_table_key=None):
        self.set(self.node.apply_to(a=a, target_table_key=target_table_key))
        return self

    def __rrshift__(self, other):  # override other >> self
        self.set(self.node.__rrshift__(other))
        return self

    def __rshift__(self, other):  # override self >> other
        self.set(self.node.__rshift__(other))
        return self

    # composition
    def add(self, other):
        self.set(self.node.add(other))
        return self

    # info

    def columns_produced(self):
        return self.node.columns_produced()

    # query generation

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return self.node.to_near_sql_implementation(
            db_model=db_model, using=using, temp_id_source=temp_id_source
        )

    # define builders for all non-initial node types on base class

    def extend_parsed(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        self.set(
            self.node.extend_parsed(
                parsed_ops=parsed_ops,
                partition_by=partition_by,
                order_by=order_by,
                reverse=reverse,
            )
        )
        return self

    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        self.set(
            self.node.extend(
                ops=ops, partition_by=partition_by, order_by=order_by, reverse=reverse
            )
        )
        return self

    def project_parsed(self, parsed_ops=None, *, group_by=None):
        self.set(self.node.project_parsed(parsed_ops=parsed_ops, group_by=group_by))
        return self

    def project(self, ops=None, *, group_by=None):
        self.set(self.node.project(ops=ops, group_by=group_by,))
        return self

    def natural_join(self, b, *, by, jointype):
        self.set(self.node.natural_join(b=b, by=by, jointype=jointype))
        return self

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        self.set(
            self.node.concat_rows(b=b, id_column=id_column, a_name=a_name, b_name=b_name)
        )
        return self

    def select_rows_parsed(self, parsed_expr):
        self.set(self.node.select_rows_parsed(parsed_expr=parsed_expr))
        return self

    def select_rows(self, expr):
        self.set(self.node.select_rows(expr=expr,))
        return self

    def drop_columns(self, column_deletions):
        self.set(self.node.drop_columns(column_deletions=column_deletions))
        return self

    def select_columns(self, columns):
        self.set(self.node.select_columns(columns=columns))
        return self

    def rename_columns(self, column_remapping):
        self.set(self.node.rename_columns(column_remapping=column_remapping))
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        self.set(self.node.order_rows(columns=columns, reverse=reverse, limit=limit))
        return self

    def convert_records(self, record_map, *, temp_namer=None):
        self.node.convert_records(record_map=record_map, temp_namer=temp_namer)
        return self

    def map_records(
        self, blocks_in=None, blocks_out=None, strict=False, temp_namer=None
    ):
        self.node.map_records(
            blocks_in=blocks_in,
            blocks_out=blocks_out,
            strict=strict,
            temp_namer=temp_namer,
            )
        return self

    # sklearn step style interface

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        self.node.fit(X=X, y=y)
        return self

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        return self.node.fit_transform(X=X, y=y)

    def get_feature_names(self, input_features=None):
        return self.node.get_feature_names(input_features=input_features)
