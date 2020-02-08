John Mount and Nina Zumel

## Introduction

This is an example of an interesting data-wrangling/data-reshaping problem and how to solve it in [`Python`](https://www.python.org)/[`Pandas`](https://pandas.pydata.org) using [`data_algebra`](https://github.com/WinVector/data_algebra). The `R` version of this example can be found [here](https://github.com/WinVector/cdata/blob/master/Examples/OrderedGrouping/OrderedGrouping.md).

## The problem

In an RStudio Community note, user <code>hklovs</code> asked [how to re-organize some data](https://community.rstudio.com/t/tidying-data-reorganizing-tibble/48292). The ask was essentially to

<blockquote>
transform data from this format: 

| ID | OP | DATE                |
| -: | :- | :------------------ |
|  1 | A  | 2001-01-02 00:00:00 |
|  1 | B  | 2015-04-25 00:00:00 |
|  2 | A  | 2000-04-01 00:00:00 |
|  3 | D  | 2014-04-07 00:00:00 |
|  4 | C  | 2012-12-01 00:00:00 |
|  4 | A  | 2005-06-16 00:00:00 |
|  4 | D  | 2009-01-20 00:00:00 |
|  4 | B  | 2009-01-20 00:00:00 |
|  5 | A  | 2010-10-10 00:00:00 |
|  5 | B  | 2003-11-09 00:00:00 |
|  6 | B  | 2004-01-09 00:00:00 |

Into this format:

| ID | DATE1               | OP1 | DATE2               | OP2         | DATE3               | OP3 |
| -: | :------------------ | :-- | :------------------ | :---------- | :------------------ | :-- |
|  1 | 2001-01-02 00:00:00 | A   | 2015-04-25 00:00:00 | B           | None                  | None  |
|  2 | 2000-04-01 00:00:00 | A   | None                  | None          | None                  | None  |
|  3 | 2014-04-07 00:00:00 | D   | None                  | None          | None                  | None  |
|  4 | 2005-06-16 00:00:00 | A   | 2009-01-20 00:00:00 | B, D        | 2012-12-01 00:00:00 | C   |
|  5 | 2003-11-09 00:00:00 | B   | 2010-10-10 00:00:00 | A           | None                  | None  |
|  6 | 2004-01-09 00:00:00 | B   | None                  | None          | None                  | None  |
</blockquote>

That is: for each `ID` pick the first three operations ordered by date, merging operations with the same timestamp.  Then write these results into a single row for each `ID`.  As is common in working with `Pandas`, we will have to consider `NaN` and `None` as fairly interchangeable.

## The solution

A good way to solve any data-wrangling problem is to:

  * Get a good definition of what is wanted
  * Re-process the data so any advisory column you wished you had is actually there
  * And finish the problem.

Let's apply this process to our example problem.

### Adding an advisory rank column

The first step isn't to worry about the data format, as it is an inessential or solvable difficulty. Instead make any extra descriptions or controls you need explicit.  In this case we need to date-rank and to merge the operations (per `ID`). So let's do that first.


```python
# bring in all of our modues/packages
import io
import re
import sqlite3

import pandas

import data_algebra.util
from data_algebra.cdata import *
from data_algebra.data_ops import *
import data_algebra.SQLite
```


```python
# some example data
d = pandas.DataFrame({
    'ID': [1, 1, 2, 3, 4, 4, 4, 4, 5, 5, 6],
    'OP': ['A', 'B', 'A', 'D', 'C', 'A', 'D', 'B', 'A', 'B', 'B'],
    'DATE': ['2001-01-02 00:00:00', '2015-04-25 00:00:00', '2000-04-01 00:00:00', 
             '2014-04-07 00:00:00', '2012-12-01 00:00:00', '2005-06-16 00:00:00', 
             '2009-01-20 00:00:00', '2009-01-20 00:00:00', '2010-10-10 00:00:00', 
             '2003-11-09 00:00:00', '2004-01-09 00:00:00'],
    })

d
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>OP</th>
      <th>DATE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>2001-01-02 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>B</td>
      <td>2015-04-25 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>A</td>
      <td>2000-04-01 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>D</td>
      <td>2014-04-07 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>C</td>
      <td>2012-12-01 00:00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>A</td>
      <td>2005-06-16 00:00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>D</td>
      <td>2009-01-20 00:00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>B</td>
      <td>2009-01-20 00:00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>A</td>
      <td>2010-10-10 00:00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>B</td>
      <td>2003-11-09 00:00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>B</td>
      <td>2004-01-09 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# define a user aggregation function

# # a function to paste a vector of strings together
# def sorted_concat(vals):
#     return ', '.join(sorted([str(vi) for vi in set(vals)]))

# just a class that doesn't declare an iterable interface
# so Pandas thinks of these values as scalars.
class Container:
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return self.value.__repr__()
    
    def __str__(self):
        return self.value.__repr__()


def sorted_concat(vals):
    return Container(sorted([str(vi) for vi in set(vals)]))

# merge the operations to get one row per ID and DATE
# then rank the rows for each ID by DATE
ops = describe_table(d, table_name='d'). \
        project({'OP': user_fn(sorted_concat, 'OP')},  # fuse all the ops on same date/id into one string
                group_by=['ID', 'DATE']). \
        extend({'rank': '_row_number()'},  # rank each ID group in order of date
               partition_by=['ID'],
               order_by=['DATE'])

# specify the first few data processing steps
d2 = ops.transform(d)

d2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE</th>
      <th>OP</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>['A']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2015-04-25 00:00:00</td>
      <td>['B']</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>['A']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>['D']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>['A']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>2009-01-20 00:00:00</td>
      <td>['B', 'D']</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>2012-12-01 00:00:00</td>
      <td>['C']</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>['B']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>2010-10-10 00:00:00</td>
      <td>['A']</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>['B']</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In the above code we used the `project()` operator to merge rows with duplicate `ID` and `DATE` into a single string listing all the operations that occurred, for example "B, D".  Then we added a rank column.  This gives us all the information we need for a complete solution to the original problem.   Now all we have to do is re-arrange the data.

### Reshaping the data

To reshape the data, we strongly encourage drawing out what one wants it terms of one input record and one output record.  With `data_algebra` doing so essentially solves the problem.

So let's look at what happens only to the rows with `ID == 1`.  In this case we expect input rows that look like this:

| ID | DATE                | OP | rank |
| -: | :------------------ | :- | ---: |
|  1 | 2001-01-02 00:00:00 | A  |    1 |
|  1 | 2015-04-25 00:00:00 | B  |    2 |
|  1 |  None               | None | 3  |

And we want this record transformed into this:

| ID | DATE1               | OP1 | DATE2               | OP2 | DATE3 | OP3 |
| -: | :------------------ | :-- | :------------------ | :-- | :---- | :-- |
|  1 | 2001-01-02 00:00:00 | A   | 2015-04-25 00:00:00 | B   | None  | None  |


We call the above record form a *row record*, because all the data for a given `ID` is in a single row. When the data for a given `ID` is not in a single row, we say it is in a *block*. In addition to having a per-record key (`ID` in our example), each row of a block is uniquely identified by an in-record structure key (in this case, `rank`).

`data_algebra` moves records from row shaped to block shaped, and vice-versa (It can also move data from one block shape to another, by going through a row).

To use `data_algebra`, draw a picture of any block record in its full generality. 
In our case the interesting record is the input shape, which looks like the following (with the record `ID` columns suppressed for conciseness).


```python
def diagram_to_pandas(s):
    s = s.strip()
    s = re.sub(r'"', '', s)
    return pandas.read_table(sep='\\s*,\\s*', engine='python', filepath_or_buffer=io.StringIO(s))

# draw a picture of the record format
diagram = diagram_to_pandas("""


    "rank",    "DATE",    "OP"
    1,         DATE1,     OP1
    2,         DATE2,     OP2
    3,         DATE3,     OP3


""")

diagram
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>DATE</th>
      <th>OP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>DATE1</td>
      <td>OP1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>DATE2</td>
      <td>OP2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>DATE3</td>
      <td>OP3</td>
    </tr>
  </tbody>
</table>
</div>



The column names `rank`, `DATE`, and `OP` are all column names of the table we are starting with.  The values `1`, `2`, and `3` are all values we expect to see in the `rank` column of the working data frame.  And the symbols `DATE1`, `DATE2`, `DATE3`, `OP1`, `OP2`, and `OP3` are all stand-in names for values we see in our data.  These symbols will be the column names of our new row-records.

By default, the first column of a diagram is the in-record key (that is why we put `rank` first).  However, any set of columns can be specified as the in-record keys through the package interfaces. 

We have tutorials on how to build these diagrams [here](https://winvector.github.io/cdata/articles/design.html) and [here](https://winvector.github.io/cdata/articles/blocksrecs.html).  Essentially we draw one record of the input and output and match column names to stand-in interior values of the other.  The output record is a single row, so we don't have to explicitly pass it in.  However it looks like the following.


```python
row_record = diagram_to_pandas("""

  "DATE1", "OP1", "DATE2", "OP2", "DATE3", "OP3"
   DATE1 ,  OP1 ,  DATE2 ,  OP2 ,  DATE3 ,  OP3

""")

row_record
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE1</th>
      <th>OP1</th>
      <th>DATE2</th>
      <th>OP2</th>
      <th>DATE3</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DATE1</td>
      <td>OP1</td>
      <td>DATE2</td>
      <td>OP2</td>
      <td>DATE3</td>
      <td>OP3</td>
    </tr>
  </tbody>
</table>
</div>



Notice the interior-data portions (the parts we wrote in the inputs as unquoted) of each table input are the cells that are matched from one record to the other.  These are in fact just the earlier sample inputs and outputs with the values replaced with the placeholders `DATE1`, `DATE2`, `DATE3`, `OP1`, `OP2`, and `OP3`.

We have a two minute video of how to work out the placeholder labels and build the diagram [here](https://youtu.be/8TVoQdnvfqQ).

With the diagram in hand we can specify the data reshaping step. Since we are moving the data from blocks to row records, we use the class `RecordMap` with a `blocks_in` argument to create the reshaping transform.


```python
record_map = RecordMap(
    blocks_in=RecordSpecification(
        control_table=diagram,
        record_keys=['ID']
    ))
```

The transform specifies that records are found in the format shown in `diagram`, and are to be converted to rows.  We can confirm the intent by printing the transform.


```python
print(str(record_map))
```

    Transform block records of structure:
    RecordSpecification
       record_keys: ['ID']
       control_table_keys: ['rank']
       control_table:
          rank   DATE   OP
       0     1  DATE1  OP1
       1     2  DATE2  OP2
       2     3  DATE3  OP3
    to row records of the form:
      record_keys: ['ID']
     ['DATE1', 'DATE2', 'DATE3', 'OP1', 'OP2', 'OP3']
    


If we apply this transform to the intermediate table `d2`, we have the data in the format we need (except possibly for the order of `ID`).


```python
# specify the first few data processing steps
ops2 = describe_table(d2, table_name='d2'). \
        convert_records(  # transform the record shape
                        record_map). \
        order_rows(['ID'])  # ensure presentation is ordered by ID

# apply the operations to the dat
res = ops2.transform(d2)

res
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE1</th>
      <th>OP1</th>
      <th>DATE2</th>
      <th>OP2</th>
      <th>DATE3</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>['A']</td>
      <td>2015-04-25 00:00:00</td>
      <td>['B']</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>['A']</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>['D']</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>['A']</td>
      <td>2009-01-20 00:00:00</td>
      <td>['B', 'D']</td>
      <td>2012-12-01 00:00:00</td>
      <td>['C']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>['B']</td>
      <td>2010-10-10 00:00:00</td>
      <td>['A']</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>['B']</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### The full transformation

We are now ready to put all of our operations together into one composite pipeline, starting from a specification of the original data `d`.


```python
# specify the first few data processing steps
ops = describe_table(d, table_name='d'). \
        project({'OP': user_fn(sorted_concat, 'OP')},  # fuse all the ops on same date/id into one string
                group_by=['ID', 'DATE']). \
        extend({'rank': '_row_number()'},  # rank each ID group in order of date
               partition_by=['ID'],
               order_by=['DATE']). \
        convert_records(  # transform the record shape
                        record_map). \
        order_rows(['ID'])   # ensure presentation is ordered by ID

# apply the operations to the data
res = ops.transform(d)

# present the results
res
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE1</th>
      <th>OP1</th>
      <th>DATE2</th>
      <th>OP2</th>
      <th>DATE3</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>['A']</td>
      <td>2015-04-25 00:00:00</td>
      <td>['B']</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>['A']</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>['D']</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>['A']</td>
      <td>2009-01-20 00:00:00</td>
      <td>['B', 'D']</td>
      <td>2012-12-01 00:00:00</td>
      <td>['C']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>['B']</td>
      <td>2010-10-10 00:00:00</td>
      <td>['A']</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>['B']</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



And we are done.

## SQL

All of the steps are easy to translate into SQL for running in databases, which we will show here.


```python
# prepare our database connection
db_model = data_algebra.SQLite.SQLiteModel()
con = sqlite3.connect(":memory:")
db_model.prepare_connection(con)

# copy our data to the database
d.to_sql('d', con, if_exists='replace')

# define the aggregator
# Note: this will move data between the database and SQL
# in general we would use a stored procedure
# Also, SQLite has its own: GROUP_CONCAT() aggegator
class SortedConcat:
    def __init__(self):
        self.accum = set()

    def step(self, value):
        self.accum.add(str(value))

    def finalize(self):
        return ', '.join(sorted([v for v in self.accum]))

# registoer our aggregator with SQLite
con.create_aggregate("sorted_concat", 1, SortedConcat)

# convert ops to SQL
sql_code = ops.to_sql(db_model)

# run the query, and bring the results back
res_db = data_algebra.default_data_model.pd.read_sql_query(sql_code, con)

res_db
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE1</th>
      <th>DATE2</th>
      <th>DATE3</th>
      <th>OP1</th>
      <th>OP2</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>2015-04-25 00:00:00</td>
      <td>None</td>
      <td>A</td>
      <td>B</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>None</td>
      <td>None</td>
      <td>A</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>None</td>
      <td>None</td>
      <td>D</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>2009-01-20 00:00:00</td>
      <td>2012-12-01 00:00:00</td>
      <td>A</td>
      <td>B, D</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>2010-10-10 00:00:00</td>
      <td>None</td>
      <td>B</td>
      <td>A</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>None</td>
      <td>None</td>
      <td>B</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



Note: column order is not considered essential in `data_algebra` pipelines (though it is easy to fix once you are in `Python`).


```python
res_db[['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE1</th>
      <th>OP1</th>
      <th>DATE2</th>
      <th>OP2</th>
      <th>DATE3</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>A</td>
      <td>2015-04-25 00:00:00</td>
      <td>B</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>A</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>D</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>A</td>
      <td>2009-01-20 00:00:00</td>
      <td>B, D</td>
      <td>2012-12-01 00:00:00</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>B</td>
      <td>2010-10-10 00:00:00</td>
      <td>A</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>B</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## A variation

If we had not wanted to merge ties, the solution would look like this:



```python
# specify the first few data processing steps
ops = describe_table(d, table_name='d'). \
        extend({'rank': '_row_number()'},  # now we have to order by Date AND op
               partition_by=['ID'],
               order_by=['DATE', 'OP']). \
        convert_records(  # transform the record shape
                        record_map). \
        select_columns(   # order the columns
                       ['ID', 'DATE1', 'OP1', 'DATE2', 'OP2', 'DATE3', 'OP3']). \
        order_rows(   # ensure presentation is ordered by ID
                   ['ID'])

# apply the operations to the data
res = ops.transform(d)

# present the results
res
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE1</th>
      <th>OP1</th>
      <th>DATE2</th>
      <th>OP2</th>
      <th>DATE3</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>A</td>
      <td>2015-04-25 00:00:00</td>
      <td>B</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>D</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>A</td>
      <td>2009-01-20 00:00:00</td>
      <td>B</td>
      <td>2009-01-20 00:00:00</td>
      <td>D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>B</td>
      <td>2010-10-10 00:00:00</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>B</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



This time we included re-ordering the columns as part of the reusable pipeline.

And this is again, easy to run in a database.


```python
# convert ops to SQL
sql_code = ops.to_sql(db_model)

# run the query, and bring the results back
res_db = data_algebra.default_data_model.pd.read_sql_query(sql_code, con)

res_db
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>DATE1</th>
      <th>OP1</th>
      <th>DATE2</th>
      <th>OP2</th>
      <th>DATE3</th>
      <th>OP3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001-01-02 00:00:00</td>
      <td>A</td>
      <td>2015-04-25 00:00:00</td>
      <td>B</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2000-04-01 00:00:00</td>
      <td>A</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2014-04-07 00:00:00</td>
      <td>D</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2005-06-16 00:00:00</td>
      <td>A</td>
      <td>2009-01-20 00:00:00</td>
      <td>B</td>
      <td>2009-01-20 00:00:00</td>
      <td>D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2003-11-09 00:00:00</td>
      <td>B</td>
      <td>2010-10-10 00:00:00</td>
      <td>A</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2004-01-09 00:00:00</td>
      <td>B</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# close the connection
con.close()
```


```python

```

