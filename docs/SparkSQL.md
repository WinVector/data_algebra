Module data_algebra.SparkSQL
============================
SparkSQL adapter for the data algebra.

Functions
---------

    
`example_handle()`
:   Return an example db handle for testing. Returns None if helper packages not present.

Classes
-------

`SparkConnection(*, spark_context, spark_session)`
:   Holder for spark conext and session as a connection (defines close).

    ### Methods

    `close(self)`
    :   Stop context and release reference to context and session.

`SparkSQLModel()`
:   A model of how SQL should be generated for SparkSQL.
    
    Known issue: doesn't coalesce NaN

    ### Ancestors (in MRO)

    * data_algebra.db_model.DBModel
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC
    * data_algebra.sql_model.SQLModel

    ### Methods

    `execute(self, conn, q)`
    :   Execute a SQL query or operator dag.

    `insert_table(self, conn, d, table_name, *, qualifiers=None, allow_overwrite=False)`
    :   Insert table into database.

    `read_query(self, conn, q)`
    :   Execute a SQL query or operator dag, return result as Pandas data frame.