Module data_algebra.data_ops
============================
Realization of data operations.

Functions
---------

    
`data(*args, **kwargs)`
:   Capture a full table for later use. Exactly one of args/kwags can be set.
    
    :param args: at most one unnamed table of the form table_name=table_value
    :param kwargs: at most one named table of the form table_name=table_value
    :return: a table description, with all values retained

    
`descr(**kwargs)`
:   Capture a named partial table as a description.
    
    :param kwargs: exactly one named table of the form table_name=table_value
    :return: a table description (not all values retained)

    
`describe_table(d, table_name: Optional[str] = None, *, qualifiers=None, sql_meta=None, row_limit: Optional[int] = 7, keep_sample=True, keep_all=False) ‑> data_algebra.view_representations.TableDescription`
:   :param d: data table table to describe
    :param table_name: name of table
    :param qualifiers: optional, able qualifiers
    :param sql_meta: optional, sql meta information map
    :param row_limit: how many rows to sample
    :param keep_sample: logical, if True retain head of table
    :param keep_all: logical, if True retain all of table
    :return: TableDescription

    
`ex(d, *, data_model=None, allow_limited_tables: bool = False)`
:   Evaluate operators with respect to data frames already stored in the operator chain.
    
    :param d: data algebra pipeline/DAG to evaluate.
    :param data_model: adaptor to data dialect
    :param allow_limited_tables: logical, if True allow execution on non-complete tables
    :return: table result

    
`table(d, *, table_name=None)`
:   Capture a table for later use
    
    :param d: Pandas data frame to capture
    :param table_name: name for this table
    :return: a table description, with values retained