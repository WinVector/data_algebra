
"""test data model isolation"""

import data_algebra
import data_algebra.data_model
import data_algebra.db_model
import data_algebra.SQLite
import data_algebra.test_util


class SQLiteDFModel(data_algebra.data_model.DataModel):
    """Implement tables in a private db"""
    _next_id: int
    _db_handle: data_algebra.db_model.DBHandle

    def __init__(self):
        data_algebra.data_model.DataModel.__init__(self, 'SQLiteDFModel')
        self._next_id = 0
        self._db_handle = data_algebra.SQLite.example_handle()

    def data_frame(self, arg=None):
        """
        Build a new emtpy data frame.

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
        assert isinstance(df, str)
        quote_name = self._db_handle.db_model.quote_table_name(df)
        # noinspection PyBroadException
        try:
            self._db_handle.read_query(f'SELECT * FROM {quote_name} LIMIT 1')
            return True
        except Exception:
            return False

    def eval(self, *, op, data_map: dict, narrow: bool):
        """
        Implementation of Pandas evaluation of operators

        :param op: ViewRepresentation to evaluate
        :param data_map: (ignored)
        :param narrow: (ignored)
        :return: data frame result
        """
        return self._db_handle.read_query(op)


def test_data_model_isolation():
    pd = data_algebra.default_data_model.pd
    sql_df_model = SQLiteDFModel()
    t0 = sql_df_model.data_frame(pd.DataFrame({'x': [1, 2, 3]}))
    ops = (
        t0.extend({'y': 'x + 1'})
    )
    res = sql_df_model.eval(op=ops, data_map=None, narrow=False)
    expect = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [2, 3, 4],
    })
    assert data_algebra.test_util.equivalent_frames(res, expect)
