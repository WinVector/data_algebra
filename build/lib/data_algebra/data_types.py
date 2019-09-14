import pandas
import numpy

import data_algebra

try:
    # noinspection PyUnresolvedReferences
    import dask

    # noinspection PyUnresolvedReferences
    import dask.dataframe
except ImportError:
    pass


try:
    # noinspection PyUnresolvedReferences
    import datatable
except ImportError:
    pass


def is_dask_data_frame(d):
    if data_algebra.have_dask:
        if isinstance(d, dask.dataframe.DataFrame):
            return True
    return False


def is_datatable_frame(d):
    if data_algebra.have_datatable:
        if isinstance(d, datatable.Frame):
            return True
    return False


def convert_to_pandas_dataframe(d, arg_name=None):
    if isinstance(d, pandas.DataFrame):
        return d
    if isinstance(d, numpy.ndarray):
        return pandas.DataFrame(d)  # not likely to have column names
    if data_algebra.have_dask:
        if isinstance(d, dask.dataframe.DataFrame):
            d = d.compute()
            if not isinstance(d, pandas.DataFrame):
                raise RuntimeError(
                    "conversion from " + str(type(d)) + " to pandas.DataFrame failed"
                )
            return d
    if data_algebra.have_datatable:
        if isinstance(d, datatable.Frame):
            return d.to_pandas()
    if arg_name is not None:
        raise TypeError(
            "can't convert argument "
            + str(arg_name)
            + " to a pandas.DataFrame (was "
            + str(type(d))
            + ")"
        )
    else:
        raise TypeError(
            "can't convert argument"
            + " to a pandas.DataFrame (was "
            + str(type(d))
            + ")"
        )
