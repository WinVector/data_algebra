[`data_algebra`](https://github.com/WinVector/data_algebra) version of this [`rquery` example](http://www.win-vector.com/blog/2019/12/what-is-new-for-rquery-december-2019/).

First lets import our modules and set up our operator pipeline.


```python
import sqlite3

import pandas

from data_algebra.data_ops import *
import data_algebra.PostgreSQL
import data_algebra.SQLite

ops = TableDescription(
    table_name='d', 
    column_names=['col1', 'col2', 'col3']). \
    extend({
        'sum23': 'col2 + col3'
    }). \
    extend({
        'x': 1
    }). \
        extend({
        'x': 2
    }). \
        extend({
        'x': 3
    }). \
        extend({
        'x': 4
    }). \
        extend({
        'x': 5
    }). \
    select_columns(['x', 'sum23', 'col3'])


print(ops)

```

    TableDescription(
     table_name='d',
     column_names=[
       'col1', 'col2', 'col3']) .\
       extend({
        'sum23': 'col2 + col3',
        'x': '5'}) .\
       select_columns(['x', 'sum23', 'col3'])


Notice even setting up the pipeline involves some optimizations.  This is simple feature of the `data_algebra`, 
made safe and easy to manage by the [category-theoretical design](http://www.win-vector.com/blog/2019/12/data_algebra-rquery-as-a-category-over-table-descriptions/).

These operations can be applied to data.


```python
d = pandas.DataFrame({
    'col1': [1, 2],
    'col2': [3, 4],
    'col3': [4, 5]
})

ops.transform(d)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>sum23</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>9</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



We are working on adapters for near-`Pandas` systems such as `modin` and others.

We can also convert the query into `SQL` query.


```python
sql_model = data_algebra.SQLite.SQLiteModel()

print(ops.to_sql(db_model=sql_model, pretty=True))

```

    SELECT 5 AS "x",
           "col2" + "col3" AS "sum23",
           "col3"
    FROM "d"


Notice this query is compact.  `data_algebra` optimizations do not combine steps with different concerns, but they do have some nice features:

  * Queries are shortened: some steps that are not used are not preserved.
  * Queries are narrowed: values not used in the result are not brought through intermediate queries.
  * Non-terminal row-orders are thrown away (as they are not semantic in many data-stores).
  * `select_column()` steps are implicit, change other steps but not translated as explicit queries.
  * Tables are used by name when deeper in queries.
 
This make for tighter query generation than the current version of [`rquery`](https://github.com/WinVector/rquery/) (which [itself one of the best query generators in `R`](http://www.win-vector.com/blog/2019/12/what-is-new-for-rquery-december-2019/)).

And we can easily demonstrate the query in action.


```python
conn = sqlite3.connect(':memory:')
sql_model.prepare_connection(conn)
sql_model.insert_table(conn, d, table_name='d')

conn.execute('CREATE TABLE res AS ' + ops.to_sql(db_model=sql_model))
sql_model.read_table(conn, 'res')
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>sum23</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>9</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
conn.close()

```

