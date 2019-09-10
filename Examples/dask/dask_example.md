
Quick demo showing data_alebra working on three different potential scales of data: pandas.DataFrame, dask.dataframe.DataFrame, and SQL.


First set up our (trivial) example.


```python
import pandas
import dask.dataframe
import psycopg2

from data_algebra.data_ops import *
import data_algebra.PostgreSQL


d_pandas = pandas.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [5, 6, 7, 8]
})
d_pandas
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



Define our (trivial) operator pipeline.


```python
ops = TableDescription('d', ['x', 'y']) .\
    extend({'z': 'x + y'})
ops
```




    TableDescription(table_name='d', column_names=['x', 'y']) .\
       extend({'z': 'x + y'})



Apply operators to pandas.DataFrame


```python
ops.transform(d_pandas)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



Set up a dask example.


```python
d_dask = dask.dataframe.from_pandas(d_pandas, npartitions=2)
```

Apply the same operators to the dask data structure.



```python
r_dask = ops.transform(d_dask)
r_dask
```




<div><strong>Dask DataFrame Structure:</strong></div>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
    <tr>
      <th>npartitions=2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th></th>
      <td>int64</td>
      <td>int64</td>
      <td>int64</td>
    </tr>
    <tr>
      <th></th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th></th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
</div>
<div>Dask Name: assign, 10 tasks</div>



Call .compute() to get the result back.


```python
r_dask.compute()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>7</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>8</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



Now the same thing in SQL with PostgreSQL.

First set up our database, and simulate having the data already in the database by copying the data over.


```python
conn_p = psycopg2.connect(
    database="johnmount",
    user="johnmount",
    host="localhost",
    password=""
)
conn_p.autocommit=True

db_model_p = data_algebra.PostgreSQL.PostgreSQLModel()

db_model_p.insert_table(conn_p, d_pandas, 'd')

sql = ops.to_sql(db_model_p, pretty=True)
print(sql)
```

    SELECT "y",
           "x",
           "x" + "y" AS "z"
    FROM
      (SELECT "y",
              "x"
       FROM "d") "sq_0"


And execute the SQL


```python
db_model_p.read_query(conn_p, sql)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>x</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>12.0</td>
    </tr>
  </tbody>
</table>
</div>



Clean up.


```python
conn_p.close()
```
