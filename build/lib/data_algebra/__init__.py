"""
Package for data processing in Python: https://github.com/WinVector/data_algebra
"""

__docformat__ = "restructuredtext"
__version__ = "1.3.0"

__doc__ = """
`data_algebra`<https://github.com/WinVector/data_algebra> is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale.  The primary 
purpose of the package is to support an easy to compose and maintain grammar of data processing
steps that in turn can be used to generate database specific SQL.  The package also implements
the same transforms for Pandas DataFrames. 

This package is still under initial development, so some parts are not yet implemented or tested, and APIs
are subject to change.

Mature, production ready `R`<https://www.r-project.org> versions of the system are available as 
the `rquery`<https://github.com/WinVector/rquery> and `rqdatatable`<https://github.com/WinVector/rqdatatable> packages.
"""

import data_algebra.pandas_model


# set up what pandas supplier we are using
default_data_model = data_algebra.pandas_model.PandasModel()
