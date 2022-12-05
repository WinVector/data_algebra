"""
Package for data processing in Python: https://github.com/WinVector/data_algebra
"""

__docformat__ = "restructuredtext"
__version__ = "1.4.6"

__doc__ = """
`data_algebra`<https://github.com/WinVector/data_algebra> is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale.  The primary 
purpose of the package is to support an easy to compose and maintain grammar of data processing
steps that in turn can be used to generate database specific SQL.  The package also implements
the same transforms for Pandas DataFrames. 

`R`<https://www.r-project.org> versions of the system are available as 
the `rquery`<https://github.com/WinVector/rquery> and `rqdatatable`<https://github.com/WinVector/rqdatatable> packages.
"""

import data_algebra.data_model
import data_algebra.data_ops
# import for easy access for package users
from data_algebra.data_ops import TableDescription, SQLNode, describe_table, descr, data, ex


# ready our data models

def _register_data_models():
    import data_algebra.pandas_model
    data_algebra.pandas_model.register_pandas_model()
    try:
        import data_algebra.polars_model
        data_algebra.polars_model.register_polars_model()
    except ModuleNotFoundError:
        pass

_register_data_models()

assert isinstance(data_algebra.data_model.data_model_type_map["default_data_model"], data_algebra.pandas_model.PandasModel)
