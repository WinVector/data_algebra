
## Introduction to the `data_algebra`

[The `data_algebra`](https://github.com/WinVector/data_algebra) is a data wrangling system designed to express complex data manipulation as a series of simple data transforms. This is in the spirit of `R`'s `base::transform()`, `dplyr`'s `dplyr::mutate()`, or [`rquery`](https://github.com/WinVector/rquery)'s `rquery::extend()` and uses a method chaining notation. The operators themselves follow the selections in Codd's relational algebra, with the addition of the traditional `SQL` "window functions."  More on the background and context of `data_algebra` can be found [here](https://github.com/WinVector/data_algebra/).

The `Python`/`data_algebra` version of this introduction is [here](https://github.com/WinVector/data_algebra/blob/master/Examples/Introduction/data_algebra_Introduction.md), and the`R`/`rquery` version of this introduction is [here](https://github.com/WinVector/rquery/blob/master/Examples/Introduction/rquery_Introduction.md).

In transform formulations data manipulation is written as transformations that produce new `DataFrame`s, instead of as alterations of a primary data structure (as is the case with `data.table`).  Transform system *can* use more space and time than in-place methods. However, in our opinion, transform systems have a number of pedagogical advantages.

In `data_algebra`'s case the primary set of data operators is as follows:

 * `drop_columns` 
 * `select_columns`
 * `rename_columns`
 * `select_rows`
 * `order_rows`
 * `extend`
 * `project`
 * `natural_join`
 * `concat_rows`
 * `convert_records`.

These operations break into a small number of themes:

 * Simple column operations (selecting and re-naming columns).
 * Simple row operations (selecting and re-ordering rows).
 * Creating new columns or replacing columns with new calculated values.
 * Aggregating or summarizing data.
 * Combining results between two `DataFrame`s.
 * General conversion of record layouts.
 
The point is: Codd worked out that a great number of data transformations can be decomposed into a small number of the above steps.  `data_algebra` supplies a high performance implementation of these methods that scales from in-memory scale up through big data scale (to just about anything that supplies a sufficiently powerful `SQL` interface, such as PostgreSQL, Apache Spark, or Google BigQuery).

We will work through simple examples/demonstrations of the `data_algebra` data manipulation operators.

## `data_algebra` operators

### Simple column operations (selecting and re-naming columns)

The simple column operations are as follows.

 * `drop_columns` 
 * `select_columns`
 * `rename_columns`
 
These operations are easy to demonstrate.

We set up some simple data.


```python
import pandas

d = pandas.DataFrame({
  'x': [1, 1, 2],
  'y': [5, 4, 3],
  'z': [6, 7, 8],
})

d
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



For example: `drop_columns` works as follows. `drop_columns` creates a new `DataFrame` without certain columns. We can start by wrapping our `DataFrame` `d` for processing, then applying the `drop_columns()` operators, and finally ending and executing the chain with the `ex()` method.


```python
from data_algebra.data_ops import *

wrap(d). \
    drop_columns(['y', 'z']). \
    ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



In all cases the first argument of a `data_algebra` operator is either the data to be processed, or an earlier `data_algebra` pipeline to be extended.  We will take about composing `data_algebra` operations after we work through examples of all of the basic operations.

`select_columns`'s action is also obvious from example.


```python
wrap(d). \
  select_columns(['x', 'y']). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



`rename_columns` is given as name-assignments of the form `'new_name': 'old_name'`:


```python
wrap(d). \
  rename_columns({
     'x_new_name': 'x',
     'y_new_name': 'y'
    }). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_new_name</th>
      <th>y_new_name</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Simple row operations (selecting and re-ordering rows)

The simple row operations are:

 * `select_rows`
 * `order_rows`

`select_rows` keeps the set of rows that meet a given predicate expression.


```python
wrap(d). \
  select_rows('x == 1'). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




`order_rows` re-orders rows by a selection of column names (and allows reverse ordering by naming which columns to reverse in the optional `reverse` argument).  Multiple columns can be selected in the order, each column breaking ties in the earlier comparisons.


```python
wrap(d). \
  order_rows(
             ['x', 'y'],
             reverse = ['x']). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



General `data_algebra` operations do not depend on row-order and are not guaranteed to preserve row-order, so if you do want to order rows you should make it the last step of your pipeline.

### Creating new columns or replacing columns with new calculated values

The important create or replace column operation is:

 * `extend`

`extend` accepts arbitrary expressions to create new columns (or replace existing ones).  For example:


```python
wrap(d). \
  extend({'zzz': 'y / x'}). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>zzz</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>



We can use `=` or `:=` for column assignment.  In these examples we will use `:=` to keep column assignment clearly distinguishable from argument binding.

`extend` allows for very powerful per-group operations akin to what [`SQL`](https://en.wikipedia.org/wiki/SQL) calls ["window functions"](https://en.wikipedia.org/wiki/SQL_window_function).  When the optional `partitionby` argument is set to a vector of column names then aggregate calculations can be performed per-group.  For example.


```python
wrap(d). \
  extend({
         'max_y': 'y.max()',
         'shift_z': 'z.shift()',
         'row_number': '_row_number()',
         'cumsum_z': 'z.cumsum()',},
         partition_by = 'x',
         order_by = ['y', 'z']). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>max_y</th>
      <th>shift_z</th>
      <th>row_number</th>
      <th>cumsum_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>7.0</td>
      <td>2</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>NaN</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>3</td>
      <td>NaN</td>
      <td>1</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



Notice the aggregates were performed per-partition (a set of rows with matching partition key values, specified by `partitionby`) and in the order determined by the `orderby` argument (without the `orderby` argument order is not guaranteed, so always set `orderby` for windowed operations that depend on row order!).

More on the window functions can be found [here](https://github.com/WinVector/data_algebra/blob/master/Examples/WindowFunctions/WindowFunctions.md).


### Aggregating or summarizing data

The main aggregation method for `data_algebra` is:

 * `project`
 
`project` performs per-group calculations, and returns only the grouping columns (specified by `groupby`) and derived aggregates.  For example:


```python
wrap(d). \
  project({
         'max_y': 'y.max()',
         'count': '_size()',},
         group_by = ['x']). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>max_y</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Notice we only get one row for each unique combination of the grouping variables.  We can also aggregate into a single row by not specifying any `groupby` columns.


```python
wrap(d). \
  project({
         'max_y': 'y.max()',
         'count': '_size()',
          }). \
  ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_y</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




### Combining results between two `DataFrame`s

To combine multiple tables in `data_algebra` one uses one of:
  
  * `natural_join` 
  * `concat_rows`

#### `natural_join`

In the `data_algebra` `natural_join`, rows are matched by column keys and any two columns with the same name are *coalesced* (meaning the first table with a non-missing values supplies the answer).  This is easiest to demonstrate with an example.

Let's set up new example tables.


```python
d_left = pandas.DataFrame({
  'k': ['a', 'a', 'b'],
  'x': [1, None, 3],
  'y': [1, None, None],
})

d_left
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
d_right = pandas.DataFrame({
  'k': ['a', 'b', 'q'],
  'y': [10, 20, 30],
})

d_right
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>q</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



To perform a join we specify which set of columns our our row-matching conditions (using the `by` argument) and what type of join we want (using the `jointype` argument).  For example we can use `jointype = 'LEFT'` to augment our `d_left` table with additional values from `d_right`.


```python
ops = describe_table(d_left, table_name = 'd_left'). \
  natural_join(b = describe_table(d_right, table_name = 'd_right'),
               by = 'k',
               jointype = 'LEFT')

ops.eval({'d_left': d_left, 'd_right': d_right})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>NaN</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>3.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



In a left-join (as above) if the right-table has unique keys then we get a table with the same structure as the left-table- but with more information per row.  This is a very useful type of join in data science projects.  Notice columns with matching names are coalesced into each other, which we interpret as "take the value from the left table, unless it is missing."

#### `concat_rows`

Another way to combine compatible tables is to concatinate rows.


```python
d_a = pandas.DataFrame({
  'k': ['a', 'a', 'b'],
  'x': [1, None, 3],
  'y': [1, None, None],
})

d_a
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
d_b = pandas.DataFrame({
  'k': [None, 'a', 'b'],
  'x': [None, 9, 3],
  'y': [None, 2, None],
})

d_b
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>9.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
ops = describe_table(d_a, table_name = 'd_a'). \
  concat_rows(b = describe_table(d_b, table_name = 'd_b'))

ops.eval({'d_a': d_a, 'd_b': d_b})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>x</th>
      <th>y</th>
      <th>source_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



### General conversion of record layouts

Record transformation is re-shaping one (possibly multi-row) record layout to another (possibly multi-row) record layout.


```python
iris_small = pandas.DataFrame({
    'Sepal.Length': [5.1, 4.9, 4.7],
    'Sepal.Width': [3.5, 3.0, 3.2],
    'Petal.Length': [1.4, 1.4, 1.3],
    'Petal.Width': [0.2, 0.2, 0.2],
    'Species': ['setosa', 'setosa', 'setosa'],
    'id': [0, 1, 2],
    })

iris_small
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from data_algebra.cdata import *

control_table = pandas.DataFrame(
    {
        "Part": ["Sepal", "Sepal", "Petal", "Petal"],
        "Measure": ["Length", "Width", "Length", "Width"],
        "Value": ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    }
)

record_spec = data_algebra.cdata.RecordSpecification(
    control_table,
    control_table_keys = ['Part', 'Measure'],
    record_keys = ['id', 'Species']
    )

map = RecordMap(blocks_out=record_spec)

print(str(map))
```

    Transform row records of the form:
      record_keys: ['id', 'Species']
     ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    to block records of structure:
    RecordSpecification
       record_keys: ['id', 'Species']
       control_table_keys: ['Part', 'Measure']
       control_table:
           Part Measure         Value
       0  Sepal  Length  Sepal.Length
       1  Sepal   Width   Sepal.Width
       2  Petal  Length  Petal.Length
       3  Petal   Width   Petal.Width
    



```python
ops = describe_table(iris_small, 'iris_small'). \
    convert_records(record_map=map)

ops.eval({'iris_small': iris_small, 
          'cdata_temp_record': ops.record_map.blocks_out.control_table})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Species</th>
      <th>Part</th>
      <th>Measure</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>setosa</td>
      <td>Petal</td>
      <td>Length</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>setosa</td>
      <td>Petal</td>
      <td>Width</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>setosa</td>
      <td>Sepal</td>
      <td>Length</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>setosa</td>
      <td>Sepal</td>
      <td>Width</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>setosa</td>
      <td>Petal</td>
      <td>Length</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>setosa</td>
      <td>Petal</td>
      <td>Width</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>setosa</td>
      <td>Sepal</td>
      <td>Length</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>setosa</td>
      <td>Sepal</td>
      <td>Width</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>setosa</td>
      <td>Petal</td>
      <td>Length</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>setosa</td>
      <td>Petal</td>
      <td>Width</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>setosa</td>
      <td>Sepal</td>
      <td>Length</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>setosa</td>
      <td>Sepal</td>
      <td>Width</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



Record transformation is "simple once you get it".  However, we suggest reading up on that as a separate topic [here](https://github.com/WinVector/data_algebra/blob/master/Examples/cdata/cdata.md).

## Composing operations

We could, of course, perform complicated data manipulation by sequencing `data_algebra` operations, and saving intermediate values. 
`data_algebra` operators can also act on `data_algebra` pipelines instead of acting on data. We can write our operations as follows.

We can use the `wrap()`/`ex()` pattern to capture both the operator pipeline and to apply it.


```python
wrapped_ops = wrap(d). \
  extend({
         'row_number': '_row_number()',
         },
         partition_by = ['x'],
         order_by = ['y', 'z']). \
  select_rows(
              'row_number == 1') . \
  drop_columns(
               "row_number")

wrapped_ops.underlying
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y', 'z']) .\
       extend({
        'row_number': '_row_number()'},
       partition_by=['x'],
       order_by=['y', 'z']) .\
       select_rows('row_number == 1') .\
       drop_columns(['row_number'])




```python
wrapped_ops.ex()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



`data_algebra` operators can also act on `data_algebra` pipelines instead of acting on data. We can write our operations as follows:


```python
ops = describe_table(d). \
  extend({
         'row_number': '_row_number()',
         },
         partition_by = ['x'],
         order_by = ['y', 'z']). \
  select_rows(
              'row_number == 1') . \
  drop_columns(
               "row_number")

ops
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y', 'z']) .\
       extend({
        'row_number': '_row_number()'},
       partition_by=['x'],
       order_by=['y', 'z']) .\
       select_rows('row_number == 1') .\
       drop_columns(['row_number'])



And we can re-use this pipeline, both on local data and to generate `SQL` to be run in remote databases. Applying this operator pipeline to our `DataFrame` `d` is performed as follows.


```python
ops.transform(d)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



What we are trying to illustrate above: there is a continuum of notations possible between:

 * Working over values with explicit intermediate variables.
 * Working over values with a pipeline.
 * Working over operators with a pipeline.

Being able to see these as all related gives some flexibility in decomposing problems into solutions.  We have some more advanced notes on the differences in working modalities [here](https://github.com/WinVector/data_algebra/blob/master/Examples/Modes/Modes.md) and [here](https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/Arrow.md).


## Conclusion

`data_algebra` supplies a very teachable grammar of data manipulation based on Codd's relational algebra and experience with pipelined data transforms (such as `base::transform()`, `dplyr`, `data.table`, `Pandas`, and `rquery`).  

For in-memory situations `data_algebra` uses `Pandas` as the implementation provider.

For bigger than memory situations `data_algebra` can translate to any sufficiently powerful `SQL` dialect, allowing `data_algebra` pipelines to be executed on PostgreSQL, Apache Spark, or Google BigQuery.

In addition the [`rquery`](https://github.com/WinVector/rquery) R package supplies a nearly identical system for working with data in R. The two systems can even [share data manipulation code between each other](https://github.com/WinVector/data_algebra/blob/master/Examples/LogisticExample/ScoringExample.md) (allowing very powerful R/Python inter-operation or helping port projects from one to the other).






```python

```
