# noinspection PyUnresolvedReferences
import numpy

# noinspection PyUnresolvedReferences
import pandas


have_yaml = False
try:
    # noinspection PyUnresolvedReferences
    import yaml  # supplied by PyYAML

    have_yaml = True
except ImportError:
    pass


have_black = False
try:
    # noinspection PyUnresolvedReferences
    import black

    have_black = True
except ImportError:
    pass


have_sqlparse = False
try:
    # noinspection PyUnresolvedReferences
    import sqlparse

    have_sqlparse = True
except ImportError:
    pass


have_dask = False
try:
    # noinspection PyUnresolvedReferences
    import dask
    import dask.dataframe

    have_dask = True
except ImportError:
    pass


have_datatable = False
try:
    # noinspection PyUnresolvedReferences
    import datatable

    have_datatable = True
except ImportError:
    pass


__docformat__ = "restructuredtext"
__version__ = "0.2.9"

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

Recommended packages include: Pandas, PyYAML (supplies yaml), sqlparse, and black. 
"""
