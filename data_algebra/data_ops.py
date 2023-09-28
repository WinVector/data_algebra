"""
Realization of data operations.
"""

from typing import Optional

import data_algebra.expr_parse
import data_algebra.flow_text
import data_algebra.data_model
import data_algebra.expr_rep
import data_algebra.data_ops_utils
import data_algebra.near_sql
import data_algebra.util
from data_algebra.data_ops_types import OperatorPlatform
from data_algebra.view_representations import ViewRepresentation, TableDescription


def describe_table(
    d,
    table_name: Optional[str] = None,
    *,
    qualifiers=None,
    sql_meta=None,
    row_limit: Optional[int] = 7,
    keep_sample=True,
    keep_all=False,
) -> TableDescription:
    """
    :param d: data table table to describe
    :param table_name: name of table
    :param qualifiers: optional, able qualifiers
    :param sql_meta: optional, sql meta information map
    :param row_limit: how many rows to sample
    :param keep_sample: logical, if True retain head of table
    :param keep_all: logical, if True retain all of table
    :return: TableDescription
    """
    assert not isinstance(d, OperatorPlatform)
    assert not isinstance(d, ViewRepresentation)
    assert isinstance(keep_sample, bool)
    assert isinstance(keep_all, bool)
    assert isinstance(
        table_name, (str, type(None))
    )  # TODO: see if we can change this to never None
    # confirm our data model is loaded
    data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
    assert data_model.is_appropriate_data_instance(d)
    column_names = list(d.columns)
    head = None
    nrows = d.shape[0]
    if keep_sample or keep_all:
        if keep_all or (row_limit is None) or (row_limit >= nrows):
            row_limit = None
            head = d
        else:
            head = d.head(row_limit)
    return TableDescription(
        table_name=table_name,
        column_names=column_names,
        qualifiers=qualifiers,
        sql_meta=sql_meta,
        head=head,
        limit_was=row_limit,
        nrows=nrows,
    )


def table(d, *, table_name=None):
    """
    Capture a table for later use

    :param d: Pandas data frame to capture
    :param table_name: name for this table
    :return: a table description, with values retained
    """
    return describe_table(
        d=d,
        table_name=table_name,
        qualifiers=None,
        sql_meta=None,
        row_limit=None,
        keep_sample=True,
        keep_all=True,
    )


def descr(**kwargs):
    """
    Capture a named partial table as a description.

    :param kwargs: exactly one named table of the form table_name=table_value
    :return: a table description (not all values retained)
    """
    assert len(kwargs) == 1
    table_name = [k for k in kwargs.keys()][0]
    d = kwargs[table_name]
    return describe_table(
        d=d,
        table_name=table_name,
        qualifiers=None,
        sql_meta=None,
        row_limit=7,
        keep_sample=True,
        keep_all=False,
    )


def data(*args, **kwargs):
    """
    Capture a full table for later use. Exactly one of args/kwags can be set.

    :param args: at most one unnamed table of the form table_name=table_value
    :param kwargs: at most one named table of the form table_name=table_value
    :return: a table description, with all values retained
    """
    assert (len(args) + len(kwargs)) == 1
    if len(kwargs) == 1:
        table_name = [k for k in kwargs.keys()][0]
        d = kwargs[table_name]
        return table(d=d, table_name=table_name)
    d = args[0]
    return table(d=d, table_name=None)


def ex(d, *, data_model=None, allow_limited_tables: bool = False):
    """
    Evaluate operators with respect to data frames already stored in the operator chain.

    :param d: data algebra pipeline/DAG to evaluate.
    :param data_model: adaptor to data dialect
    :param allow_limited_tables: logical, if True allow execution on non-complete tables
    :return: table result
    """
    return d.ex(data_model=data_model, allow_limited_tables=allow_limited_tables)
