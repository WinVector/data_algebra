"""
Base class for SQL database adapters for data algebra.
"""

from typing import List, Optional

import data_algebra
import data_algebra.data_model
import data_algebra.near_sql
import data_algebra.expr_rep
import data_algebra.util
import data_algebra.data_ops_types
import data_algebra.data_ops
from data_algebra.sql_format_options import SQLFormatOptions
from data_algebra.shift_pipe_action import ShiftPipeAction
from data_algebra.sql_model import SQLModel


class DBModel(ShiftPipeAction, SQLModel):
    """A model of how SQL should be generated for a given database, and database connection managed."""

    def __init__(
        self,
        *,
        identifier_quote: str = '"',
        string_quote: str = "'",
        sql_formatters=None,
        op_replacements=None,
        on_start: str = "",
        on_end: str = "",
        on_joiner: str = "AND",
        drop_text: str = "DROP TABLE",
        string_type: str = "VARCHAR",
        float_type: str = "FLOAT64",
        supports_with: bool = True,
        supports_cte_elim: bool = True,
        allow_extend_merges: bool = True,
        default_SQL_format_options=None,
        union_all_term_start: str = "(",
        union_all_term_end: str = ")",
    ):
        ShiftPipeAction.__init__(self)
        SQLModel.__init__(
            self,
            identifier_quote=identifier_quote,
            string_quote=string_quote,
            sql_formatters=sql_formatters,
            op_replacements=op_replacements,
            on_start=on_start,
            on_end=on_end,
            on_joiner=on_joiner,
            drop_text=drop_text,
            string_type=string_type,
            float_type=float_type,
            supports_with=supports_with,
            supports_cte_elim=supports_cte_elim,
            allow_extend_merges=allow_extend_merges,
            default_SQL_format_options=default_SQL_format_options,
            union_all_term_start=union_all_term_start,
            union_all_term_end=union_all_term_end,
        )
        self.local_data_model = data_algebra.data_model.default_data_model()

    def db_handle(self, conn, *, db_engine=None):
        """
        Create a db handle (adapter plus connection).

        :param conn: database connection
        :param db_engine: optional sqlalchemy style engine (for closing)
        """
        return DBHandle(db_model=self, conn=conn, db_engine=db_engine)

    def prepare_connection(self, conn):
        """
        Do any augmentation or preparation of a database connection. Example: adding stored procedures.
        """
        pass

    # database helpers

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        """
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        assert isinstance(q, str)
        conn.execute(q)

    def read_query(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        :return: query results as table
        """
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        assert isinstance(q, str)
        r = self.local_data_model.pd.io.sql.read_sql_query(q, conn)
        return r

    def table_exists(self, conn, table_name: str) -> bool:
        """
        Return true if table exists.
        """
        assert isinstance(table_name, str)
        q_table_name = self.quote_table_name(table_name)
        table_exists = True
        # noinspection PyBroadException
        try:
            self.read_query(conn, "SELECT * FROM " + q_table_name + " LIMIT 1")
        except Exception:
            table_exists = False
        return table_exists

    def drop_table(self, conn, table_name: str, *, check: bool = True) -> None:
        """
        Remove a table.
        """
        if (not check) or self.table_exists(conn, table_name):
            q_table_name = self.quote_table_name(table_name)
            self.execute(conn, self.drop_text + " " + q_table_name)

    # noinspection PyMethodMayBeStatic,SqlNoDataSourceInspection
    def insert_table(
        self, conn, d, table_name: str, *, qualifiers=None, allow_overwrite=False
    ) -> None:
        """
        Insert a table.

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :param qualifiers: schema and such
        :param allow_overwrite logical, if True drop previous table
        """

        if qualifiers is not None:
            raise ValueError("non-empty qualifiers not yet supported on insert")
        incoming_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
        d = incoming_data_model.to_pandas(d)
        if self.table_exists(conn, table_name):
            if not allow_overwrite:
                raise ValueError("table " + table_name + " already exists")
            else:
                self.drop_table(conn, table_name, check=False)
        # Note: the Pandas to_sql() method appears to have SQLite hard-wired into it
        # it refers to sqlite_master
        # this behavior seems to change if sqlalchemy is active
        # n.b. sqlalchemy tries to insert values on %
        d = data_algebra.data_model.lookup_data_model_for_dataframe(d).to_pandas(d)
        if d.shape[1] < 1:
            # deal with no columns case
            d = d.copy()
            d["_index"] = range(d.shape[0])
        d.to_sql(name=table_name, con=conn, index=False)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def read_table(self, conn, table_name: str, *, qualifiers=None, limit=None):
        """
        Return table contents as a Pandas data frame.
        """
        assert isinstance(table_name, str)
        q_table_name = self.quote_table_name(table_name)
        sql = "SELECT * FROM " + q_table_name
        if limit is not None:
            sql = sql + " LIMIT " + limit.__repr__()
        return self.read_query(conn, sql)

    def read(self, conn, table):
        """
        Return table as a pandas data frame for table description.
        """
        if table.node_name != "TableDescription":
            raise TypeError(
                "Expect table to be a data_algebra.data_ops.TableDescription"
            )
        return self.read_table(
            conn=conn, table_name=table.table_name, qualifiers=table.qualifiers
        )

    def act_on(self, b, *, correct_ordered_first_call: bool = False):
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            return self.to_sql(b)
        if correct_ordered_first_call and isinstance(b, ShiftPipeAction):
            return b.act_on(self, correct_ordered_first_call=False)  # fall back
        raise TypeError(f"inappropriate type to DBModel.act_on(): {type(b)}")

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        return self.__str__()


class DBHandle(ShiftPipeAction):
    """
    Container for database connection handles.
    """

    def __init__(self, *, db_model: DBModel, conn, db_engine=None):
        """
        Represent a db connection.

        :param db_model: associated database model
        :param conn: database connection
        :param db_engine: optional sqlalchemy style engine (for closing)
        """
        # TODO: user controllable data model?
        ShiftPipeAction.__init__(self)
        assert isinstance(db_model, DBModel)
        self.db_model = db_model
        self.db_engine = db_engine
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def read_query(self, q):
        """
        Return results of query as a Pandas data frame.
        """
        return self.db_model.read_query(conn=self.conn, q=q)

    def act_on(self, b, *, correct_ordered_first_call: bool = False):
        if isinstance(b, data_algebra.data_ops.ViewRepresentation):
            return self.read_query(b)
        if isinstance(b, str):
            return self.read_query(b)
        if correct_ordered_first_call and isinstance(b, ShiftPipeAction):
            return b.act_on(self, correct_ordered_first_call=False)  # fall back
        raise TypeError(f"inappropriate type to DBHandle.act_on(): {type(b)}")

    def read_table(self, table_name: str):
        """
        Return table as a Pandas data frame.

        :param table_name: table to read
        """
        tn = self.db_model.quote_table_name(table_name)
        return self.read_query(f"SELECT * FROM {tn}")

    def create_table(self, *, table_name: str, q):
        """
        Create table from query.

        :param table_name: table to create
        :param q: query
        :return: table description
        """
        tn = self.db_model.quote_table_name(table_name)
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self.db_model)
        else:
            q = str(q)
        self.execute(f"CREATE TABLE {tn} AS {q}")
        return self.describe_table(table_name)

    def describe_table(
        self, table_name: str, *, qualifiers=None, row_limit: Optional[int] = 7
    ):
        """
        Return a description of a database table.
        """
        head = self.read_query(
            q="SELECT * FROM "
            + self.db_model.quote_table_name(table_name)
            + " LIMIT "
            + str(row_limit),
        )
        return data_algebra.data_ops.describe_table(
            head, table_name=table_name, qualifiers=qualifiers, row_limit=row_limit
        )

    def execute(self, q) -> None:
        """
        Execute a SQL query or operator dag.
        """
        self.db_model.execute(conn=self.conn, q=q)

    def drop_table(self, table_name: str) -> None:
        """
        Remove a table.
        """
        self.db_model.drop_table(self.conn, table_name)

    def insert_table(
        self, d, *, table_name: str, allow_overwrite: bool = False
    ) -> data_algebra.data_ops.TableDescription:
        """
        Insert a table into the database.
        """
        incoming_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
        d = incoming_data_model.to_pandas(d)
        self.db_model.insert_table(
            conn=self.conn, d=d, table_name=table_name, allow_overwrite=allow_overwrite
        )
        res = self.describe_table(table_name)
        return res

    def to_sql(
        self,
        ops: data_algebra.data_ops.ViewRepresentation,
        *,
        sql_format_options: Optional[SQLFormatOptions] = None,
    ) -> str:
        """
        Convert operations into SQL
        """
        return self.db_model.to_sql(ops=ops, sql_format_options=sql_format_options)

    def query_to_csv(self, q, *, res_name: str) -> None:
        """
        Execute a query and save the results as a CSV file.
        """
        d = self.read_query(q)
        d.to_csv(res_name, index=False)

    def table_values_to_sql_str_list(
        self, v, *, result_name: str = "table_values"
    ) -> List[str]:
        """
        Convert a table of values to a SQL. Only for small tables.
        """
        return self.db_model.table_values_to_sql_str_list(v, result_name=result_name)

    def __str__(self):
        return (
            str(type(self).__name__)
            + "("
            + "db_model="
            + str(self.db_model)
            + ", conn="
            + str(self.conn)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def close(self) -> None:
        """
        Dispose of engine, or close connection.
        """
        if self.conn is not None:
            caught = None
            if self.db_engine is not None:
                # sqlalchemy style handle
                # noinspection PyBroadException
                try:
                    self.db_engine.dispose()
                except Exception as ex:
                    caught = ex
            else:
                # noinspection PyBroadException
                try:
                    self.conn.close()
                except Exception as ex:
                    caught = ex
            self.db_engine = None
            self.conn = None
            if caught is not None:
                raise ValueError("close caught: " + str(caught))
