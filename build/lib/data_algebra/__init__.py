import importlib


__docformat__ = "restructuredtext"
__version__ = "0.3.9"

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

pd = importlib.import_module("pandas")  # https://pandas.pydata.org

# TODO: possibly import modin instead
# can't do that now
# https://github.com/modin-project/modin/issues/865
# pd = importlib.import_module("modin.pandas")  # https://github.com/modin-project/modin
