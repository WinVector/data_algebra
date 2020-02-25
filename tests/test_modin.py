
import importlib
import pytest
import warnings

import data_algebra
from data_algebra.data_ops import *
from data_algebra.modin_model import ModinModel
import data_algebra.test_util

@pytest.mark.filterwarnings("ignore:")
def test_modin():
    modin_pandas = None
    data_model = None
    try:
        modin_pandas = importlib.import_module("modin.pandas")
        data_model = ModinModel(modin_engine='ray')
    except:
        pass

    if (modin_pandas is not None) and (data_model is not None):
        d = modin_pandas.DataFrame({'x': [1, 2]})

        ops = describe_table(d, table_name='d'). \
            extend({'y': '2 * x'})

        data_map = {'d': d}

        res_name = data_model.eval(ops, data_map=data_map)
        res = data_map[res_name]

        assert isinstance(res, modin_pandas.DataFrame)

        res_p = data_model.to_pandas(res, data_map=data_map)

        assert isinstance(res_p, data_algebra.default_data_model.pd.DataFrame)

        expect = data_algebra.default_data_model.pd.DataFrame({
            'x': [1, 2],
            'y': [2, 4],
            })
        assert data_algebra.test_util.equivalent_frames(res_p, expect)