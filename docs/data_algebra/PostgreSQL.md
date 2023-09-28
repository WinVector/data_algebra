Module data_algebra.PostgreSQL
==============================
PostgreSQL database adapter for data algebra.

Functions
---------

    
`example_handle()`
:   Return an example db handle for testing. Returns None if helper packages not present.

Classes
-------

`PostgreSQLModel()`
:   A model of how SQL should be generated for PostgreSQL.
    Assuming we are using a sqlalchemy engine as our connection

    ### Ancestors (in MRO)

    * data_algebra.db_model.DBModel
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC
    * data_algebra.sql_model.SQLModel