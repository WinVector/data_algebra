
import pytest
import warnings

import data_algebra
from data_algebra.data_ops import *
import data_algebra.modin_model
import data_algebra.test_util

modin_engine_choice = 'ray'

@pytest.mark.filterwarnings("ignore:")
def test_modin():
    modin_pandas = None
    data_model = None
    try:
        data_model = data_algebra.modin_model.ModinModel(modin_engine=modin_engine_choice)
        modin_pandas = data_algebra.modin_model.MODIN_PANDAS
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

        # check original frame is unchanged
        start_modin = data_map['d']
        start_p = data_model.to_pandas(start_modin, data_map=data_map)
        expect_start = data_algebra.default_data_model.pd.DataFrame({
            'x': [1, 2],
            })
        assert data_algebra.test_util.equivalent_frames(start_p, expect_start)


@pytest.mark.filterwarnings("ignore:")
def test_modin_gcalc():
    modin_pandas = None
    data_model = None
    try:
        data_model = data_algebra.modin_model.ModinModel(modin_engine=modin_engine_choice)
        modin_pandas = data_algebra.modin_model.MODIN_PANDAS
    except:
        pass

    if (modin_pandas is not None) and (data_model is not None):
        d = modin_pandas.DataFrame({
            'x': [1, 2, 3, 5],
            'g': ['a', 'a', 'b', 'b']})

        ops = describe_table(d, table_name='d'). \
            extend({'x_mean': 'x.mean()'},
                   partition_by = ['g'])

        data_map = {'d': d}
        res_name = data_model.eval(ops, data_map=data_map)
        res = data_map[res_name]
        res_p = data_model.to_pandas(res, data_map=data_map)

        expect = data_algebra.default_data_model.pd.DataFrame({
            'x': [1, 2, 3, 5],
            'g': ['a', 'a', 'b', 'b'],
            'x_mean': [1.5, 1.5, 4.0, 4.0],
            })
        assert data_algebra.test_util.equivalent_frames(res_p, expect)

        # check original frame is unchanged
        start_modin = data_map['d']
        start_p = data_model.to_pandas(start_modin, data_map=data_map)
        expect_start = data_algebra.default_data_model.pd.DataFrame({
            'x': [1, 2, 3, 5],
            'g': ['a', 'a', 'b', 'b']})
        assert data_algebra.test_util.equivalent_frames(start_p, expect_start)
