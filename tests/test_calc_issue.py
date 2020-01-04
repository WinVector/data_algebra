import numpy
import data_algebra

def test_calc_issue():
    # https://github.com/pandas-dev/pandas/issues/29819
    d = data_algebra.pd.DataFrame({
        "a": [True, False],
        "b": [1, 2],
        "c": [3, 4]})

    pandas_eval_env = {
        "if_else": lambda c, x, y: numpy.where(c, x, y),
    }
    d.eval("@if_else(a, 1, c)", local_dict=pandas_eval_env, global_dict=None)

    pandas_eval_env = {
        "if_else": lambda c, x, y: data_algebra.pd.Series(numpy.where(c, x, y), name='if_else')
    }
    d.eval("@if_else(a, 1, c)", local_dict=pandas_eval_env, global_dict=None)
