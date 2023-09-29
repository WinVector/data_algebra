"""
Package for data processing in Python: https://github.com/WinVector/data_algebra
"""

__docformat__ = "restructuredtext"
__version__ = "1.7.1"

__doc__ = """
`data_algebra`<https://github.com/WinVector/data_algebra> is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale.  The primary 
purpose of the package is to support an easy to compose and maintain grammar of data processing
steps that in turn can be used to generate database specific SQL.  The package also implements
the same transforms for Pandas and Polars DataFrames. 

`R`<https://www.r-project.org> versions of the system are available as 
the `rquery`<https://github.com/WinVector/rquery> and `rqdatatable`<https://github.com/WinVector/rqdatatable> packages.
"""


import data_algebra.data_model
import data_algebra.data_ops

# import for easy access for package users
from data_algebra.data_ops import (
    describe_table, 
    descr, 
    data, 
    ex, 
    table
)
import data_algebra.view_representations
from data_algebra.view_representations import (
    TableDescription,
    ViewRepresentation,
    SQLNode,
)
import data_algebra.expr_rep
from data_algebra.expr_rep import (
    lit, col, d_, one
)

import data_algebra.data_schema
import data_algebra.cdata
from data_algebra.cdata import (
    RecordMap, 
    RecordSpecification,
    pivot_blocks_to_rowrecs,
    pivot_rowrecs_to_blocks,
    pivot_specification,
    unpivot_specification
)

import data_algebra.pandas_model
# do this last so everything is loaded/define
data_algebra.pandas_model.register_pandas_model()
