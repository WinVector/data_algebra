The [`data_algebra`](https://github.com/WinVector/data_algebra) is designed to have a number of different modes of use.  The primary intended one the considered mode of building up a pipelines from a description of the tables to be acted on.

(Note the `Python`/`data_algebra` version can be found [here](https://github.com/WinVector/data_algebra/blob/master/Examples/Modes/Modes.md), and the `R`/`rquery` version of this example can be found [here](https://github.com/WinVector/rquery/blob/master/Examples/Modes/Modes.md).)

For our example, lets start with the following example data.


```python

```


```python
import pandas

d = pandas.DataFrame({
  'x': [1, 2, 3, 4, 5, 6],
  'y': [2, 2, 2, 3, 7, 10],
  'g': ['a', 'a', 'a', 'b', 'b' ,'b'],
})

d
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
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>b</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>10</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



For our task: let's find a row with the largest ratio of 'y' to 'x', per group 'g'.

The `data_algebra` concept is to break this into small sub-goals and steps:

 * Find the ratio of 'y' to 'x'.
 * Rank the rows by this ratio.
 * Mark our chosen rows.

In the standard `data_algebra` practice we build up our processing pipeline to follow our above plan.  The translation involves some familiarity with the `data_algebra` steps, including the row-numbering command [`row_number()`](https://github.com/WinVector/data_algebra/blob/master/Examples/WindowFunctions/WindowFunctions.md).


```python
from data_algebra.data_ops import *

ops = describe_table(d, table_name='d'). \
  extend(       # add a new column
         {'ratio': 'y / x'}). \
  extend(       # rank the rows by group and order
         {'simple_rank': '_row_number()'},
         partition_by = ['g'],
         order_by = ['ratio'],
         reverse = ['ratio']). \
  extend(       # mark the rows we want
         {'choice': 'simple_rank == 1'})
```

The `ops` operator pipeline can than be used to process data.

Either through the `eval()` method.


```python
ops.eval({'d': d})
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
      <th>g</th>
      <th>ratio</th>
      <th>simple_rank</th>
      <th>choice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>a</td>
      <td>2.000000</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>a</td>
      <td>1.000000</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
      <td>a</td>
      <td>0.666667</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>b</td>
      <td>0.750000</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>b</td>
      <td>1.400000</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>10</td>
      <td>b</td>
      <td>1.666667</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Or, if there is only one table defined, through the `transform()` method.


```python
ops.transform(d)
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
      <th>g</th>
      <th>ratio</th>
      <th>simple_rank</th>
      <th>choice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>a</td>
      <td>2.000000</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>a</td>
      <td>1.000000</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2</td>
      <td>a</td>
      <td>0.666667</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>b</td>
      <td>0.750000</td>
      <td>3</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7</td>
      <td>b</td>
      <td>1.400000</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>10</td>
      <td>b</td>
      <td>1.666667</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Another point is: this form documents check-able (and enforceable) pre and post conditions on the calculation.  For example such a calculation documents what columns are required by the calculation, and which ones are produced.


```python
# columns produced
ops.column_names
```




    ['x', 'y', 'g', 'ratio', 'simple_rank', 'choice']




```python
# columns used
ops.columns_used_from_sources()
```




    [{'g', 'ratio', 'simple_rank', 'x', 'y'}]



We can in fact make these conditions the explicit basis of [an interpretation of these data transforms as category theory arrows](https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/Arrow.md).


```python
from data_algebra.arrow import *

print(str(DataOpArrow(ops)))
```

    [
     'd':
      at least [ x: <class 'numpy.int64'>, y: <class 'numpy.int64'>, g: <class 'str'> ]
       ->
      at least [ choice, g, ratio, simple_rank, x, y ]
    ]
    



```python

```


```python

```
