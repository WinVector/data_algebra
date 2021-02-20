from data_algebra.eval_model import EvalModel
from data_algebra.pandas_base import PandasModelBase


class PandasModel(EvalModel, PandasModelBase):
    def __init__(self, *, pd, presentation_model_name="pandas"):
        EvalModel.__init__(self)
        PandasModelBase.__init__(
            self, pd=pd, presentation_model_name=presentation_model_name
        )

    # EvalModel interface

    def to_pandas(self, handle, *, data_map=None):
        if isinstance(data_map, str):
            res = data_map[handle]
        else:
            res = handle
        # noinspection PyUnresolvedReferences
        if not isinstance(res, self.pd.DataFrame):
            raise TypeError("expected handle to be of type self.pd.DataFrame")
        return res

    def eval(self, ops, *, data_map=None, result_name=None, eval_env=None, narrow=True):
        tables_needed = [k for k in ops.get_tables().keys()]
        missing_tables = set(tables_needed) - set(data_map.keys())
        if len(missing_tables) > 0:
            raise ValueError("missing required tables: " + str(missing_tables))
        if result_name is not None:
            result_name = self.mk_tmp_name(data_map)
        if result_name in tables_needed:
            raise ValueError("Can not write over an input table")
        res = ops.eval(data_map, eval_env=eval_env, data_model=self, narrow=narrow)
        data_map[result_name] = res
        return result_name
