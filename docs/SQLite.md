Module data_algebra.SQLite
==========================
Adapt data_algebra to SQLite database.

Functions
---------

    
`example_handle() ‑> data_algebra.db_model.DBHandle`
:   Return an example db handle for testing. Returns None if helper packages not present.

Classes
-------

`CollectingAgg()`
:   Aggregate from a collection. SQLite user class.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.SQLite.MedianAgg
    * data_algebra.SQLite.SampStdDevAgg
    * data_algebra.SQLite.SampVarDevAgg

    ### Methods

    `calc(self) ‑> float`
    :   Perform the calculation (only called with non-trivial data)

    `finalize(self)`
    :   Return result.

    `step(self, value)`
    :   Observe value

`MedianAgg()`
:   Aggregate as median. SQLite user class.

    ### Ancestors (in MRO)

    * data_algebra.SQLite.CollectingAgg
    * abc.ABC

    ### Methods

    `calc(self) ‑> float`
    :   do it

`SQLiteModel()`
:   A model of how SQL should be generated for SQLite

    ### Ancestors (in MRO)

    * data_algebra.db_model.DBModel
    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC
    * data_algebra.sql_model.SQLModel

    ### Methods

    `natural_join_to_near_sql(self, join_node, *, using=None, temp_id_source=None, sql_format_options=None, left_is_first=True)`
    :   Translate a join into SQL, converting right and full joins to replacement code (as SQLite doesn't have these).

    `prepare_connection(self, conn)`
    :   Insert user functions into db.

`SampStdDevAgg()`
:   Aggregate as sample standard deviation. SQLite user class.
    This version keeps the data instead of using the E[(x-E[x])^2] = E[x^2] - E[x]^2 formula

    ### Ancestors (in MRO)

    * data_algebra.SQLite.CollectingAgg
    * abc.ABC

    ### Methods

    `calc(self) ‑> float`
    :   do it

`SampVarDevAgg()`
:   Aggregate as sample standard deviation. SQLite user class.
    This version keeps the data instead of using the E[(x-E[x])^2] = E[x^2] - E[x]^2 formula

    ### Ancestors (in MRO)

    * data_algebra.SQLite.CollectingAgg
    * abc.ABC

    ### Methods

    `calc(self) ‑> float`
    :   do it