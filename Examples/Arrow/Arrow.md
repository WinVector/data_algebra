
Example of data transforms as categorical arrows ([`R` version](https://github.com/WinVector/rquery/blob/master/Examples/Arrow/Arrow.md) [`Python` version](https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/Arrow.md)).

(For ideas on applying category theory to science and data, please see David I Spivak, *Category Theory for the Sciences*, MIT Press, 2014.)

The [Python `data_algebra` package](https://github.com/WinVector/data_algebra) supplies a number of operators for working with tabular data.  The operators are picked in reference to [Codd's relational algebra](https://en.wikipedia.org/wiki/Relational_algebra), though (as with [`SQL`](https://en.wikipedia.org/wiki/SQL)) we do not insist on table rows being unique. Many of the operations are simple: selecting rows, selecting columns, joining tables.  Two of the operations stand out: projecting or aggregating rows, and extending tables with new derived columns.

An interesting point is: while the `data_algebra` operators are fairly generic: the operator pipelines that map a single table to a single table form a category over a nice set of objects.

The objects of this category can be either of:

 * Sets of column names.
 * Maps of column names to column types (schema-like objects).
 
I will take a liberty and call these objects (with or without types) "schemas."

Our setup is easiest to explain with an example.  Let's work an example in `Python`.

First we import our packages and instantiate an example data frame.


```python
import pandas
import graphviz

import data_algebra.diagram
from data_algebra.data_ops import *  # https://github.com/WinVector/data_algebra
import data_algebra.util
import data_algebra.arrow

d = pandas.DataFrame({
    'g': ['a', 'b', 'b', 'c', 'c', 'c'],
    'x': [1, 4, 5, 7, 8, 9],
    'v': [10.0, 40.0, 50.0, 70.0, 80.0, 90.0],
    'i': [True, True, False, False, False, False],
})

d
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>x</th>
      <th>v</th>
      <th>i</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>10.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



`data_algebra` operator pipelines are designed to transform data.  For example we can define the following operator pipeline which is designed count how many different values there are for `g`, and assign a unique integer id to each group.


```python
table_description = TableDescription('d', ['g', 'x', 'v', 'i'])

id_ops_a = table_description. \
    project(group_by=['g']). \
    extend({
        'ngroup': '_row_number()',
    },
    order_by=['g'])
```

The pipeline is saved in the variable `id_ops_a` which can then be applied to our data as follows.


```python
id_ops_a.transform(d)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>ngroup</th>
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
    <tr>
      <th>2</th>
      <td>c</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The pipelines are designed for composition in addition to application to data.  For example we can use the `id_ops_a` pipeline as part of a larger pipeline as follows.


```python
id_ops_b = table_description. \
    natural_join(id_ops_a, by=['g'], jointype='LEFT')
```

This pipeline specifies joining the integer group ids back into the original table as follows.


```python
id_ops_b.transform(d)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>x</th>
      <th>v</th>
      <th>i</th>
      <th>ngroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>10.0</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Notice the `ngroup` column is a function of the `g` column in this result.

I am now ready to state my big point.  These pipelines have documented pre and post conditions: what set of columns (and optionally types) they expect on their input, and what set of columns (optionally types) the pipeline produces.


```python
# needs
id_ops_b.columns_used()
```




    {'d': {'g', 'i', 'v', 'x'}}




```python
# produced
id_ops_b.column_names
```




    ['g', 'x', 'v', 'i', 'ngroup']



This is where we seem to have nice opportunity to use category theory to manage our pre-and post conditions.  Let's wrap this pipeline into a convenience class to make the categorical connection easier to see.


```python
a1 = data_algebra.arrow.DataOpArrow(id_ops_b)

print(a1)
```

    [
      [ g, x, i, v ]
       ->
      [ g, x, v, i, ngroup ]
    ]
    


`a1` is an arrow in a category whose objects are sets of column names (or alternately in a category whose objects are maps from column names to column types).


```python
a1.fit(d)

print(a1)
```

    [
      [ g: <class 'str'>, x: <class 'numpy.int64'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'> ]
    ]
    


As is typical in category theory, there can be more than one arrow from a given object to given object.  Our particular arrow is more fully described as follows.


```python
print(a1.__repr__())
```

    DataOpArrow(TableDescription(
     table_name='d',
     column_names=[
       'g', 'x', 'v', 'i']) .\
       natural_join(b=
          TableDescription(
           table_name='d',
           column_names=[
             'g', 'x', 'v', 'i']) .\
             project({
              },
             group_by=['g']) .\
             extend({
              'ngroup': '_row_number()'},
             order_by=['g']),
          by=['g'], jointype='LEFT'))


So our arrows are arrows in a category whose objects are sets of column names (or alternately maps from column names to column types).  These arrows also act on data frames that meet the required column pre conditions.


```python
a1.transform(d)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>x</th>
      <th>v</th>
      <th>i</th>
      <th>ngroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>10.0</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The arrows compose exactly when the pre-conditions meet the post conditions.  

Here are two examples of violating the pre and post conditions.


```python
cols2_too_small = [c for c in (set(id_ops_b.column_names) - set(['i']))]
ordered_ops = TableDescription('d2', cols2_too_small). \
    extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
    order_by=['x'],
    partition_by=['g'])
a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
print(a2)

```

    [
      [ g, ngroup, x, v ]
       ->
      [ g, ngroup, x, v, row_number, shift_v ]
    ]
    



```python
try:
    a1 >> a2
except ValueError as e:
    print(str(e))
    
```

    extra incoming columns: {'i'}



```python
cols2_too_large = id_ops_b.column_names + ['q']
ordered_ops = TableDescription('d2', cols2_too_large). \
    extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
    order_by=['x'],
    partition_by=['g'])
a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
print(a2)
```

    [
      [ q, ngroup, x, i, g, v ]
       ->
      [ g, x, v, i, ngroup, q, row_number, shift_v ]
    ]
    



```python
try:
    a1 >> a2
except ValueError as e:
    print(str(e))

```

    missing required columns: {'q'}


The point is: we will never see the above exceptions when we compose arrows that match on pre and post conditions (which in category theory are the only arrows you are allowed to compose).

When the pre and post conditions are met the arrows compose in a fully associative manner.


```python
ordered_ops = TableDescription('d2', id_ops_b.column_names). \
    extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
    order_by=['x'],
    partition_by=['g'])
a2 = data_algebra.arrow.DataOpArrow(ordered_ops)
print(a2)
```

    [
      [ ngroup, x, i, g, v ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v ]
    ]
    



```python
print(a2)
```

    [
      [ ngroup, x, i, g, v ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v ]
    ]
    



```python
print(a1 >> a2)
```

    [
      [ g: <class 'str'>, x: <class 'numpy.int64'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v ]
    ]
    


We can also enforce type invarients.


```python
wrong_example = pandas.DataFrame({
    'g': ['a'],
    'v': [1.0],
    'x': ['b'],
    'i': [True],
    'ngroup': [1]
})

a2.fit(wrong_example)
print(a2)
```

    [
      [ ngroup: <class 'numpy.int64'>, x: <class 'str'>,
        i: <class 'numpy.bool_'>, g: <class 'str'>, v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'str'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'> ]
    ]
    



```python
try:
    a1 >> a2
except Exception as ex:
    print(str(ex))
```

    column x self incoming type is <class 'str'>, while X outgoing type is <class 'numpy.int64'>



```python
print(a2.fit(a1.transform(d)))
```

    [
      [ ngroup: <class 'numpy.int64'>, x: <class 'numpy.int64'>,
        i: <class 'numpy.bool_'>, g: <class 'str'>, v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'> ]
    ]
    



```python
unordered_ops = TableDescription('d3', ordered_ops.column_names). \
    extend({
        'size': '_size()',
        'max_v': 'v.max()',
        'min_v': 'v.min()',
        'sum_v': 'v.sum()',
        'mean_v': 'v.mean()',
        'count_v': 'v.count()',
        'size_v': 'v.size()',
    },
    partition_by=['g'])
a3 = data_algebra.arrow.DataOpArrow(unordered_ops)
print(a3)
```

    [
      [ ngroup, x, shift_v, i, g, row_number, v ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v, size, max_v, min_v, sum_v,
        mean_v, count_v, size_v ]
    ]
    



```python
print(a3.fit(a2.transform(a1.transform(d))))
```

    [
      [ ngroup: <class 'numpy.int64'>, x: <class 'numpy.int64'>,
        shift_v: <class 'numpy.float64'>, i: <class 'numpy.bool_'>,
        g: <class 'str'>, row_number: <class 'numpy.int64'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        size: <class 'numpy.int64'>, max_v: <class 'numpy.float64'>,
        min_v: <class 'numpy.float64'>, sum_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'>, count_v: <class 'numpy.int64'>,
        size_v: <class 'numpy.int64'> ]
    ]
    



```python
print(a1 >> a2 >> a3)
```

    [
      [ g: <class 'str'>, x: <class 'numpy.int64'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        size: <class 'numpy.int64'>, max_v: <class 'numpy.float64'>,
        min_v: <class 'numpy.float64'>, sum_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'>, count_v: <class 'numpy.int64'>,
        size_v: <class 'numpy.int64'> ]
    ]
    



```python
print((a1 >> a2) >> a3)
```

    [
      [ g: <class 'str'>, x: <class 'numpy.int64'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        size: <class 'numpy.int64'>, max_v: <class 'numpy.float64'>,
        min_v: <class 'numpy.float64'>, sum_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'>, count_v: <class 'numpy.int64'>,
        size_v: <class 'numpy.int64'> ]
    ]
    



```python
print(a1 >> (a2 >> a3))
```

    [
      [ g: <class 'str'>, x: <class 'numpy.int64'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        size: <class 'numpy.int64'>, max_v: <class 'numpy.float64'>,
        min_v: <class 'numpy.float64'>, sum_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'>, count_v: <class 'numpy.int64'>,
        size_v: <class 'numpy.int64'> ]
    ]
    


All three compositions are in fact the same arrow.


```python
(a1 >> a2) >> a3
```




    DataOpArrow(TableDescription(
     table_name='d',
     column_names=[
       'g', 'x', 'v', 'i']) .\
       natural_join(b=
          TableDescription(
           table_name='d',
           column_names=[
             'g', 'x', 'v', 'i']) .\
             project({
              },
             group_by=['g']) .\
             extend({
              'ngroup': '_row_number()'},
             order_by=['g']),
          by=['g'], jointype='LEFT') .\
       extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()'},
       partition_by=['g'],
       order_by=['x']) .\
       extend({
        'size': '_size()',
        'max_v': 'v.max()',
        'min_v': 'v.min()',
        'sum_v': 'v.sum()',
        'mean_v': 'v.mean()',
        'count_v': 'v.count()',
        'size_v': 'v.size()'},
       partition_by=['g']))




```python
a1 >> (a2 >> a3)
```




    DataOpArrow(TableDescription(
     table_name='d',
     column_names=[
       'g', 'x', 'v', 'i']) .\
       natural_join(b=
          TableDescription(
           table_name='d',
           column_names=[
             'g', 'x', 'v', 'i']) .\
             project({
              },
             group_by=['g']) .\
             extend({
              'ngroup': '_row_number()'},
             order_by=['g']),
          by=['g'], jointype='LEFT') .\
       extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()'},
       partition_by=['g'],
       order_by=['x']) .\
       extend({
        'size': '_size()',
        'max_v': 'v.max()',
        'min_v': 'v.min()',
        'sum_v': 'v.sum()',
        'mean_v': 'v.mean()',
        'count_v': 'v.count()',
        'size_v': 'v.size()'},
       partition_by=['g']))



The payoff is: we can use this composite arrow on data.


```python
(a1 >> a2 >> a3).transform(d)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>x</th>
      <th>v</th>
      <th>i</th>
      <th>ngroup</th>
      <th>row_number</th>
      <th>shift_v</th>
      <th>size</th>
      <th>max_v</th>
      <th>min_v</th>
      <th>sum_v</th>
      <th>mean_v</th>
      <th>count_v</th>
      <th>size_v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>10.0</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
      <td>50.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>40.0</td>
      <td>2</td>
      <td>50.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
      <td>3</td>
      <td>2</td>
      <td>70.0</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>80.0</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The combination operator `>>` is fully associative over the combination of data and arrows.


```python
# Python default associates left to right so this is:
# ((d >> a1) >> a2) >> a3
d >> a1 >> a2 >> a3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>x</th>
      <th>v</th>
      <th>i</th>
      <th>ngroup</th>
      <th>row_number</th>
      <th>shift_v</th>
      <th>size</th>
      <th>max_v</th>
      <th>min_v</th>
      <th>sum_v</th>
      <th>mean_v</th>
      <th>count_v</th>
      <th>size_v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>10.0</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
      <td>50.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>40.0</td>
      <td>2</td>
      <td>50.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
      <td>3</td>
      <td>2</td>
      <td>70.0</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>80.0</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the preferred notation, work in operator space
d >> (a1 >> a2 >> a3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>g</th>
      <th>x</th>
      <th>v</th>
      <th>i</th>
      <th>ngroup</th>
      <th>row_number</th>
      <th>shift_v</th>
      <th>size</th>
      <th>max_v</th>
      <th>min_v</th>
      <th>sum_v</th>
      <th>mean_v</th>
      <th>count_v</th>
      <th>size_v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>10.0</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
      <td>2</td>
      <td>50.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
      <td>2</td>
      <td>2</td>
      <td>40.0</td>
      <td>2</td>
      <td>50.0</td>
      <td>40.0</td>
      <td>90.0</td>
      <td>45.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
      <td>3</td>
      <td>2</td>
      <td>70.0</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
      <td>3</td>
      <td>3</td>
      <td>80.0</td>
      <td>3</td>
      <td>90.0</td>
      <td>70.0</td>
      <td>240.0</td>
      <td>80.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The underlying `data_algebra` steps compute and check very similar pre and post conditions, the arrow class is just making this look more explicitly like arrows moving through objects in category.

There is more to be gotten from how the data relates to the schema descriptions.  I think we have that if we consider the arrows operating on data and the arrows operating on schemas we have a faithfull embedding (in the sense of Saunders Mac Lane *Categories for the Working Mathematician, 2nd Edition*, Springer, 1997, page 15) from data to schemas.
