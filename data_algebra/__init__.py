"""
Package for data processing in Python: https://github.com/WinVector/data_algebra
"""

__docformat__ = "restructuredtext"
__version__ = "1.4.5"

__doc__ = """
`data_algebra`<https://github.com/WinVector/data_algebra> is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale.  The primary 
purpose of the package is to support an easy to compose and maintain grammar of data processing
steps that in turn can be used to generate database specific SQL.  The package also implements
the same transforms for Pandas DataFrames. 

`R`<https://www.r-project.org> versions of the system are available as 
the `rquery`<https://github.com/WinVector/rquery> and `rqdatatable`<https://github.com/WinVector/rqdatatable> packages.
"""

# import for easy access for package users
from data_algebra.data_ops import TableDescription, SQLNode, describe_table, descr, data, ex
