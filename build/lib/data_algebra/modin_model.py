import os

import data_algebra.eval_model
from data_algebra.pandas_base import PandasModelBase


MODIN_ENGINE = None
MODIN_PANDAS = None


class ModinModel(data_algebra.eval_model.EvalModel):
    def __init__(self, modin_engine=None):
        # can't change engine, so track it as a global
        # https://github.com/modin-project/modin
        global MODIN_ENGINE
        global MODIN_PANDAS
        if MODIN_PANDAS is None:
            if modin_engine is None:
                raise ValueError("modin_engine not set")
            MODIN_ENGINE = modin_engine
            # https://github.com/modin-project/modin
            os.environ["MODIN_ENGINE"] = MODIN_ENGINE
            import modin.pandas

            MODIN_PANDAS = modin.pandas
        else:
            if (modin_engine is not None) and (modin_engine != MODIN_ENGINE):
                raise ValueError(
                    "MODIN_ENGINE already set to "
                    + MODIN_ENGINE
                    + ", and called with modin_engine=="
                    + modin_engine
                )
        data_algebra.eval_model.EvalModel.__init__(self)
        self.impl = PandasModelBase(pd=MODIN_PANDAS, presentation_model_name="modin")

    def to_pandas(self, handle, *, data_map=None):
        if isinstance(handle, str):
            res = data_map[handle]
        else:
            res = handle
        # noinspection PyProtectedMember
        res = res._to_pandas()  # https://github.com/modin-project/modin/issues/896
        return res

    def eval(self, ops, *, data_map=None, result_name=None, eval_env=None, narrow=True):
        tables_needed = [k for k in ops.get_tables().keys()]
        missing_tables = set(tables_needed) - set(data_map.keys())
        if len(missing_tables) > 0:
            raise ValueError("missing required tables: " + str(missing_tables))
        if result_name is None:
            result_name = self.mk_tmp_name(data_map)
        if result_name in tables_needed:
            raise ValueError("Can not write over an input table")
        res = ops.eval(data_map, eval_env=eval_env, data_model=self.impl, narrow=narrow)
        data_map[result_name] = res
        return result_name
