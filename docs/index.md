Module data_algebra
===================
`data_algebra`<https://github.com/WinVector/data_algebra> is a piped data wrangling system
based on Codd's relational algebra and experience working with dplyr at scale.  The primary 
purpose of the package is to support an easy to compose and maintain grammar of data processing
steps that in turn can be used to generate database specific SQL.  The package also implements
the same transforms for Pandas and Polars DataFrames. 

`R`<https://www.r-project.org> versions of the system are available as 
the `rquery`<https://github.com/WinVector/rquery> and `rqdatatable`<https://github.com/WinVector/rqdatatable> packages.

Sub-modules
-----------
* data_algebra.BigQuery
* data_algebra.MySQL
* data_algebra.OrderedSet
* data_algebra.PolarsSQL
* data_algebra.PostgreSQL
* data_algebra.SQLite
* data_algebra.SparkSQL
* data_algebra.arrow
* data_algebra.cdata
* data_algebra.connected_components
* data_algebra.data_model
* data_algebra.data_model_space
* data_algebra.data_ops
* data_algebra.data_ops_types
* data_algebra.data_ops_utils
* data_algebra.data_schema
* data_algebra.data_space
* data_algebra.db_model
* data_algebra.db_space
* data_algebra.eval_cache
* data_algebra.expr_parse
* data_algebra.expr_parse_fn
* data_algebra.expr_rep
* data_algebra.expression_walker
* data_algebra.flow_text
* data_algebra.fmt_python
* data_algebra.near_sql
* data_algebra.op_catalog
* data_algebra.pandas_base
* data_algebra.pandas_model
* data_algebra.parse_by_lark
* data_algebra.polars_model
* data_algebra.python3_lark
* data_algebra.shift_pipe_action
* data_algebra.solutions
* data_algebra.sql_format_options
* data_algebra.sql_model
* data_algebra.test_util
* data_algebra.util
* data_algebra.view_representations