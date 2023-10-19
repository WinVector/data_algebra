Module data_algebra.MySQL
=========================
Partial adapter of data algebra for MySQL. Not all data algebra operations are supported on this database at this time.

Functions
---------

    
`example_handle()`
:   Return an example db handle for testing. Returns None if helper packages not present.

Classes
-------

`MySQLModel()`
:   A model of how SQL should be generated for MySQL.
    Assuming we are using a sqlalchemy engine as our connection.

    ### Ancestors (in MRO)

    * data_algebra.db_model.DBModel
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC
    * data_algebra.sql_model.SQLModel