"""
Redirecting container.
"""

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set

import data_algebra.data_ops_types
import data_algebra.data_ops
import data_algebra.expr_rep


# for use in building expressions such as one.sum()
from data_algebra.data_ops_types import MethodUse

one = data_algebra.expr_rep.Value(1)


class OpC(data_algebra.data_ops_types.OperatorPlatform):
    """Container that redirects to another to non-quoted notation."""

    ops: Optional[
        data_algebra.data_ops.ViewRepresentation
    ]  # this reference gets replaced
    column_namespace: SimpleNamespace  # don't replace the reference, instead mutate (reference shared!)
    used_result: bool

    def __init__(self, other: data_algebra.data_ops.ViewRepresentation):
        self.column_namespace = SimpleNamespace()  # allows a dot notation
        self.used_result = False
        data_algebra.data_ops_types.OperatorPlatform.__init__(
            self, node_name="container"
        )
        self.ops = None
        self.set(other)

    def set(self, other: data_algebra.data_ops.ViewRepresentation):
        assert isinstance(other, data_algebra.data_ops.ViewRepresentation)
        assert not isinstance(other, OpC)  # don't allow deep nesting
        self.ops = other
        self.column_namespace.__dict__.clear()
        self.column_namespace.__dict__.update(self.ops.column_map())
        return self

    def methods_used(self) -> Set[MethodUse]:
        assert self.ops is not None  # type hint and guard
        return self.ops.methods_used()

    def get_ops(self):
        self.used_result = True
        res = self.ops
        self.ops = None
        return res

    def ex(self, *, data_model=None, allow_limited_tables=False):
        """
        Evaluate operators with respect to Pandas data frames already stored in the operator chain.

        :param data_model: adaptor to data dialect (Pandas for now)
        :param allow_limited_tables: logical, if True allow execution on non-complete tables
        :return: table result
        """
        assert not self.used_result
        assert self.ops is not None
        self.used_result = True
        res = self.ops.ex(
            data_model=data_model,
            allow_limited_tables=allow_limited_tables,
        )
        self.ops = None
        return res

    # noinspection PyPep8Naming
    def transform(
        self,
        X,
        *,
        data_model=None
    ):
        assert isinstance(self.ops, data_algebra.data_ops.ViewRepresentation)
        return self.ops.transform(
            X=X,
            data_model=data_model
        )

    # noinspection PyPep8Naming
    def act_on(self, X, *, data_model=None):
        self.set(self.ops.act_on(X=X, data_model=data_model))
        return self

    def replace_leaves(self, replacement_map: Dict[str, Any]):
        self.set(self.ops.replace_leaves(replacement_map))
        return self

    def eval(self, data_map: Dict[str, Any], *, data_model=None,):
        return self.ops.eval(data_map=data_map, data_model=data_model)

    def get_tables(self):
        return self.ops.get_tables()

    # composition
    def add(self, other):
        self.set(self.ops.add(other))
        return self

    # info

    def columns_produced(self):
        return self.ops.columns_produced()

    # query generation

    def to_near_sql_implementation_(self, db_model, *, using, temp_id_source, sql_format_options=None):
        return self.ops.to_near_sql_implementation_(
            db_model=db_model, using=using, temp_id_source=temp_id_source, sql_format_options=sql_format_options
        )

    # define builders for all non-initial ops types on base class

    def extend_parsed_(
        self, parsed_ops, *, partition_by=None, order_by=None, reverse=None
    ):
        self.set(
            self.ops.extend_parsed_(
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

    def project_parsed_(self, parsed_ops=None, *, group_by=None):
        self.set(self.ops.project_parsed_(parsed_ops=parsed_ops, group_by=group_by))
        return self

    def project(self, ops=None, *, group_by=None):
        self.set(
            self.ops.project(
                ops=ops,
                group_by=group_by,
            )
        )
        return self

    def natural_join(
        self,
        b,
        *,
        on: Optional[Iterable[str]] = None,
        jointype: str,
        check_all_common_keys_in_equi_spec: bool = False,
        by: Optional[Iterable[str]] = None,
        check_all_common_keys_in_by: bool = False
    ):
        assert (on is None) or (by is None)
        if by is not None:
            on = by
        self.set(
            self.ops.natural_join(
                b=b,
                on=on,
                jointype=jointype,
                check_all_common_keys_in_by=(
                    check_all_common_keys_in_equi_spec or check_all_common_keys_in_by
                ),
            )
        )
        return self

    def concat_rows(self, b, *, id_column="source_name", a_name="a", b_name="b"):
        self.set(
            self.ops.concat_rows(b=b, id_column=id_column, a_name=a_name, b_name=b_name)
        )
        return self

    def select_rows_parsed_(self, parsed_expr):
        self.set(self.ops.select_rows_parsed_(parsed_expr=parsed_expr))
        return self

    def select_rows(self, expr):
        self.set(
            self.ops.select_rows(
                expr=expr,
            )
        )
        return self

    def drop_columns(self, column_deletions):
        self.set(self.ops.drop_columns(column_deletions=column_deletions))
        return self

    def select_columns(self, columns):
        self.set(self.ops.select_columns(columns=columns))
        return self

    def map_columns(self, column_remapping):
        self.set(self.ops.map_columns(column_remapping=column_remapping))
        return self

    def rename_columns(self, column_remapping):
        self.set(self.ops.rename_columns(column_remapping=column_remapping))
        return self

    def order_rows(self, columns, *, reverse=None, limit=None):
        self.set(self.ops.order_rows(columns=columns, reverse=reverse, limit=limit))
        return self

    def convert_records(self, record_map):
        self.ops.convert_records(record_map=record_map)
        return self

    def map_records(self, blocks_in=None, blocks_out=None):
        self.ops.map_records(
            blocks_in=blocks_in,
            blocks_out=blocks_out,
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

    def __str__(self) -> str:
        return f"OpC({self.ops})"
    
    def __repr__(self) -> str:
        return f"OpC({self.ops})"


# pop 0343 context manager
# https://www.python.org/dev/peps/pep-0343/#use-cases
class Pipeline:
    def __init__(self, other):
        self.container = OpC(other)

    def __enter__(self):
        return self.container, self.container.column_namespace

    def __exit__(self, *args):
        assert self.container is not None
        assert self.container.used_result
        self.container = None
