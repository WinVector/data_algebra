
"""test data model isolation"""

from typing import Dict, Optional
import data_algebra
import data_algebra.data_ops
import data_algebra.data_model
import data_algebra.db_model
import data_algebra.SQLite
import data_algebra.test_util


class SQLiteDFModel(data_algebra.data_model.DataModel):
    """Implement tables in a private db. Like sqldf from R."""
    _next_id: int
    _db_handle: data_algebra.db_model.DBHandle

    def __init__(self):
        data_algebra.data_model.DataModel.__init__(self, 'SQLiteDFModel')
        self._next_id = 0
        self._db_handle = data_algebra.SQLite.example_handle()

    def data_frame(self, arg=None) -> data_algebra.data_ops.TableDescription:
        """
        Build a new empty data frame. Inserts table as a side effect.

        :param arg: optional argument passed to constructor.
        :return: data frame
        """

        table_name = f'temp_{self._next_id}'
        self._next_id = self._next_id + 1
        return self._db_handle.insert_table(d=arg, table_name=table_name, allow_overwrite=True)

    def is_appropriate_data_instance(self, df) -> bool:
        """
        Check if df is our type of data frame.
        """
        assert isinstance(df, data_algebra.data_ops.TableDescription)
        quote_name = self._db_handle.db_model.quote_table_name(df.table_name)
        # noinspection PyBroadException
        try:
            self._db_handle.read_query(f'SELECT * FROM {quote_name} LIMIT 1')
            return True
        except Exception:
            return False

    def eval(self, op, *, data_map: Optional[Dict] = None, narrow: bool = False):
        """
        Implementation of Pandas evaluation of operators. Inserts tables as a side-effect.

        :param op: ViewRepresentation to evaluate
        :param data_map: must map names to existing table descriptions with same name
        :param narrow: must be False
        :return: data frame result
        """
        assert not narrow

        tables_used = op.get_tables()
        if data_map is not None:
            for k in tables_used.keys():
                v = data_map[k]
                assert isinstance(v, data_algebra.data_ops.TableDescription)
                assert k == v.table_name
                assert self.is_appropriate_data_instance(v)
            for k, v in data_map.items():
                self._db_handle.insert_table(d=v, table_name=k, allow_overwrite=True)
        for k, v in tables_used.items():
            assert isinstance(v, data_algebra.data_ops.TableDescription)
            assert k == v.table_name
            assert self.is_appropriate_data_instance(v)

        return self._db_handle.read_query(op)


def test_data_model_isolation():
    pd = data_algebra.pandas_model.default_data_model.pd
    sql_df_model = SQLiteDFModel()
    t0 = sql_df_model.data_frame(pd.DataFrame({'x': [1, 2, 3]}))
    ops = (
        t0.extend({'y': 'x + 1'})
    )
    res = sql_df_model.eval(ops)
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
