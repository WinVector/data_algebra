
import collections
from types import SimpleNamespace

import data_algebra.data_ops_types
import data_algebra.data_ops


class OpC(data_algebra.data_ops_types.OperatorPlatform):
    """Container that redirects to another to allow method chaining."""

    def __init__(self):
        self.nd = None
        data_algebra.data_ops_types.OperatorPlatform.__init__(self,
                                                              node_name='container',
                                                              column_map=collections.OrderedDict())

    def set(self, other):
        assert isinstance(other, data_algebra.data_ops_types.OperatorPlatform)
        self.nd = other
        self.column_map = other.column_map.copy()
        self.c = SimpleNamespace(**other.column_map)  # allows a dot notation
        return self

    def describe_table(self, d, table_name="data_frame",
                   *,
                   qualifiers=None,
                   sql_meta=None,
                   column_types=None,
                   row_limit=7):
        td = data_algebra.data_ops.describe_table(
            d=d,
            table_name=table_name,
            qualifiers=qualifiers,
            sql_meta=sql_meta,
            column_types=column_types,
            row_limit=row_limit
        )
        return self.set(td)

    def ops(self):
        return self.nd

    # noinspection PyPep8Naming
    def transform(self, X, *, data_model=None, narrow=True):
        return self.nd.transform(X=X, data_model=data_model, narrow=narrow)

    # noinspection PyPep8Naming
    def act_on(self, X, *, data_model=None):
        self.set(self.nd.act_on(X=X, data_model=data_model))
        return self

    def apply_to(self, a, *, target_table_key=None):
        self.set(self.nd.apply_to(a=a, target_table_key=target_table_key))
        return self

    def __rrshift__(self, other):  # override other >> self
        self.set(self.nd.__rrshift__(other))
        return self

    def __rshift__(self, other):  # override self >> other
        self.set(self.nd.__rshift__(other))
        return self

    # composition
    def add(self, other):
        self.set(self.nd.add(other))
        return self

    # info

    def columns_produced(self):
        return self.nd.columns_produced()

    # query generation

    def to_near_sql_implementation(self, db_model, *, using, temp_id_source):
        return self.nd.to_near_sql_implementation(db_model=db_model, using=using, temp_id_source=temp_id_source)

    # define builders for all non-initial node types on base class

    def extend_parsed(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        self.set(self.nd.extend_parsed(
            parsed_ops=parsed_ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse
        ))
        return self

    def extend(self, ops, *, partition_by=None, order_by=None, reverse=None):
        self.set(self.nd.extend(
            ops=ops,
            partition_by=partition_by,
            order_by=order_by,
            reverse=reverse))
        return self

    def project_parsed(self, parsed_ops=None, *, group_by=None):
        self.set(self.nd.project_parsed(
            parsed_ops=parsed_ops,
            group_by=group_by
        ))
        return self

    def project(self, ops=None, *, group_by=None):
        self.set(self.nd.project(
            ops=ops,
            group_by=group_by,
        ))
        return self

    def natural_join(self, b, *, by, jointype):
        self.set(self.nd.natural_join(
            b=b,
            by=by,
            jointype=jointype
        ))
        return self

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        self.set(self.nd.concat_rows(
            b=b,
            id_column=id_column,
            a_name=a_name,
            b_name=b_name
        ))
        return self

    def select_rows_parsed(self, parsed_expr):
        self.set(self.nd.select_rows_parsed(
            parsed_expr=parsed_expr
        ))
        return self

    def select_rows(self, expr):
        self.set(self.nd.select_rows(
            expr=expr,
        ))
        return self

    def drop_columns(self, column_deletions):
        self.set(self.nd.drop_columns(
            column_deletions=column_deletions
        ))
        return self

    def select_columns(self, columns):
        self.set(self.nd.select_columns(
            columns=columns
        ))
        return self

    def rename_columns(self, column_remapping):
        self.set(self.nd.rename_columns(
            column_remapping=column_remapping
        ))
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        self.set(self.nd.order_rows(
            columns=columns,
            reverse=reverse,
            limit=limit
        ))
        return self

    def convert_records(self, record_map, *, temp_namer=None):
        return self.nd.convert_records(record_map=record_map, temp_namer=temp_namer)

    def map_records(self, blocks_in=None, blocks_out=None, strict=False, temp_namer=None):
        return self.nd.map_records(
            blocks_in=blocks_in,
            blocks_out=blocks_out,
            strict=strict,
            temp_namer=temp_namer
        )

    # sklearn step style interface

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X, y=None):
        return self.nd.fit(X=X, y=y)

    # noinspection PyPep8Naming, PyUnusedLocal
    def fit_transform(self, X, y=None):
        return self.nd.fit_transform(X=X, y=y)

    def get_feature_names(self, input_features=None):
        return self.nd.get_feature_names(input_features=input_features)
