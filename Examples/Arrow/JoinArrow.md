
Converting a join to a an arrow ([`R` version](https://github.com/WinVector/rquery/blob/master/Examples/Arrow/JoinArrow.md), [`Python` version](https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/JoinArrow.md)).


```python
import pandas

from data_algebra.data_ops import *
from data_algebra.arrow import *

d1 = pandas.DataFrame({
    'key': ['a', 'b'],
    'x': [1, 2],
})

d1
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
table_1_description = describe_table(d1, table_name='d1')

d2 = pandas.DataFrame({
    'key': ['b', 'c'],
    'y': [3, 4],
})

d2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
table_2_description = describe_table(d2, table_name='d2')

ops =  table_1_description.\
    natural_join(
        b=table_2_description, 
        by=['key'],
        jointype='FULL')
```


```python
arrow_1 = DataOpArrow(ops, free_table_key=table_1_description.key)
#arrow_1.fit(d1)

print(arrow_1)
```

    [
     'd1':
      [ key: <class 'str'>, x: <class 'numpy.int64'> ]
       ->
      [ key, x, y ]
    ]
    



```python
arrow_2 = DataOpArrow(ops, free_table_key=table_2_description.key)

print(arrow_2)
```

    [
     'd2':
      [ key: <class 'str'>, y: <class 'numpy.int64'> ]
       ->
      [ key, x, y ]
    ]
    



```python
arrow_1.pipeline.eval_pandas({
    'd1': d1,
    'd2': d2,
})

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



More on the categorical arrow treatment of data transformations can be found [here](https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/Arrow.md).

(Examples of advanced inter-operation between the [`R` `rquery` package](https://github.com/WinVector/rquery/) and the [`Python` `data_algebra` package](https://github.com/WinVector/data_algebra) and `SQL` can be found [here](https://github.com/WinVector/data_algebra/blob/master/Examples/LogisticExample/ScoringExample.md).)

