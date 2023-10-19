Module data_algebra.db_model
============================
Base class for SQL database adapters for data algebra.

Classes
-------

`DBHandle(*, db_model: data_algebra.db_model.DBModel, conn, db_engine=None)`
:   Container for database connection handles.
    
    Represent a db connection.
    
    :param db_model: associated database model
    :param conn: database connection
    :param db_engine: optional sqlalchemy style engine (for closing)

    ### Ancestors (in MRO)

    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC

    ### Descendants

    * data_algebra.BigQuery.BigQuery_DBHandle

    ### Methods

    `close(self) ‑> None`
    :   Dispose of engine, or close connection.

    `create_table(self, *, table_name: str, q)`
    :   Create table from query.
        
        :param table_name: table to create
        :param q: query
        :return: table description

    `describe_table(self, table_name: str, *, qualifiers=None, row_limit: Optional[int] = 7)`
    :   Return a description of a database table.

    `drop_table(self, table_name: str) ‑> None`
    :   Remove a table.

    `execute(self, q) ‑> None`
    :   Execute a SQL query or operator dag.

    `insert_table(self, d, *, table_name: str, allow_overwrite: bool = False) ‑> data_algebra.view_representations.TableDescription`
    :   Insert a table into the database.

    `query_to_csv(self, q, *, res_name: str) ‑> None`
    :   Execute a query and save the results as a CSV file.

    `read_query(self, q)`
    :   Return results of query as a Pandas data frame.

    `read_table(self, table_name: str)`
    :   Return table as a Pandas data frame.
        
        :param table_name: table to read

    `table_values_to_sql_str_list(self, v, *, result_name: str = 'table_values') ‑> List[str]`
    :   Convert a table of values to a SQL. Only for small tables.

    `to_sql(self, ops: data_algebra.view_representations.ViewRepresentation, *, sql_format_options: Optional[data_algebra.sql_format_options.SQLFormatOptions] = None) ‑> str`
    :   Convert operations into SQL

`DBModel(*, identifier_quote: str = '"', string_quote: str = "'", sql_formatters=None, op_replacements=None, on_start: str = '', on_end: str = '', on_joiner: str = 'AND', drop_text: str = 'DROP TABLE', string_type: str = 'VARCHAR', float_type: str = 'FLOAT64', supports_with: bool = True, supports_cte_elim: bool = True, allow_extend_merges: bool = True, default_SQL_format_options=None, union_all_term_start: str = '(', union_all_term_end: str = ')')`
:   A model of how SQL should be generated for a given database, and database connection managed.

    ### Ancestors (in MRO)

    * data_algebra.shift_pipe_action.ShiftPipeAction
    * abc.ABC
    * data_algebra.sql_model.SQLModel

    ### Descendants

    * data_algebra.BigQuery.BigQueryModel
    * data_algebra.MySQL.MySQLModel
    * data_algebra.PostgreSQL.PostgreSQLModel
    * data_algebra.SQLite.SQLiteModel
    * data_algebra.SparkSQL.SparkSQLModel

    ### Methods

    `db_handle(self, conn, *, db_engine=None)`
    :   Create a db handle (adapter plus connection).
        
        :param conn: database connection
        :param db_engine: optional sqlalchemy style engine (for closing)

    `drop_table(self, conn, table_name: str, *, check: bool = True) ‑> None`
    :   Remove a table.

    `execute(self, conn, q)`
    :   :param conn: database connection
        :param q: sql query

    `insert_table(self, conn, d, table_name: str, *, qualifiers=None, allow_overwrite=False) ‑> None`
    :   Insert a table.
        
        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :param qualifiers: schema and such
        :param allow_overwrite logical, if True drop previous table

    `prepare_connection(self, conn)`
    :   Do any augmentation or preparation of a database connection. Example: adding stored procedures.

    `read(self, conn, table)`
    :   Return table as a pandas data frame for table description.

    `read_query(self, conn, q)`
    :   :param conn: database connection
        :param q: sql query
        :return: query results as table

    `read_table(self, conn, table_name: str, *, qualifiers=None, limit=None)`
    :   Return table contents as a Pandas data frame.

    `table_exists(self, conn, table_name: str) ‑> bool`
    :   Return true if table exists.