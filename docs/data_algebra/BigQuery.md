Module data_algebra.BigQuery
============================
Adapter for Google BigQuery database

Functions
---------

    
`example_handle()`
:   Return an example db handle for testing. Returns None if helper packages not present.
    Note: binds in a data_catalog and data schema prefix. So this handle is specific
    to one database.

Classes
-------

`BigQueryModel(*, table_prefix: Optional[str] = None)`
:   A model of how SQL should be generated for BigQuery
    connection should be google.cloud.bigquery.client.Client

    ### Ancestors (in MRO)

    * data_algebra.db_model.DBModel
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC
    * data_algebra.sql_model.SQLModel

    ### Methods

    `get_table_name(self, table_description)`
    :

`BigQuery_DBHandle(*, db_model=BigQueryModel, conn)`
:   Container for database connection handles.
    
    Represent a db connection.
    
    :param db_model: associated database model
    :param conn: database connection
    :param db_engine: optional sqlalchemy style engine (for closing)

    ### Ancestors (in MRO)

    * data_algebra.db_model.DBHandle
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Methods

    `describe_bq_table(self, *, table_catalog, table_schema, table_name, row_limit=7) ‑> data_algebra.view_representations.TableDescription`
    :

    `query_to_csv(self, q, *, res_name) ‑> None`
    :   Write query to csv