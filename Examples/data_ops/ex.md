

```python
import pandas
from data_algebra.data_ops import *

d = pandas.DataFrame({
    'x': [1, 2, 3],
    'y': [4, 5, 6]
})

ops = wrap(d). \
    extend({'z': 'x + y'})

ops
```




    [TableDescription(
     table_name='data_frame',
     column_names=[
       'x', 'y']) .\
       extend({
        'z': 'x + y'})](
     {'data_frame'})




```python
ops.ex()
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
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>6</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
d
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
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



