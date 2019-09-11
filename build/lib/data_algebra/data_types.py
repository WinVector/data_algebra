
import pandas
import numpy

import data_algebra

try:
    # noinspection PyUnresolvedReferences
    import dask.dataframe
except ImportError:
    pass


def is_acceptable_data_frame(d):
    if isinstance(d, pandas.DataFrame):
        return True
    if data_algebra.have_dask:
        if isinstance(d, dask.dataframe.DataFrame):
            return True
    return False


def is_dask_data_frame(d):
    if isinstance(d, pandas.DataFrame):
        return False
    if data_algebra.have_dask:
        if isinstance(d, dask.dataframe.DataFrame):
            return True
    return False


def assert_is_acceptable_data_frame(d, arg_name=None):
    if not is_acceptable_data_frame(d):
        if arg_name is not None:
            raise TypeError("argument " + str(arg_name) +
                            " should be a pandas.DataFrame or dask.dataframe.DataFrame (was " +
                            str(type(d)) + ")")
        else:
            raise TypeError("argument" +
                            " should be a pandas.DataFrame or dask.dataframe.DataFrame (was " +
                            str(type(d)) + ")")


def convert_to_pandas_dataframe(d, arg_name=None):
    if isinstance(d, pandas.DataFrame):
        return d
    if data_algebra.have_dask:
        if isinstance(d, dask.dataframe.DataFrame):
            d = d.compute()
            if not isinstance(d, pandas.DataFrame):
                raise Exception("conversion from " + str(type(d)) + " to pandas.DataFrame failed")
            return d
    if isinstance(d, numpy.ndarray):
        if d.dtype.names is None:
            d = pandas.DataFrame(d)
            d.columns = ["col_" + str(i) for i in range(d.shape[1])]
        else:
            d = pandas.DataFrame(d)
        return d
    if arg_name is not None:
        raise TypeError("can't convert argument " + str(arg_name) +
                        " to a pandas.DataFrame (was " +
                        str(type(d)) + ")")
    else:
        raise TypeError("can't convert argument" +
                        " to a pandas.DataFrame (was " +
                        str(type(d)) + ")")
