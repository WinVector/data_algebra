
Example of data transforms as categorical arrows ([`R` version](https://github.com/WinVector/rquery/blob/master/Examples/Arrow/Arrow.md) [`Python` version](https://github.com/WinVector/data_algebra/blob/master/Examples/Arrow/Arrow.md)).

(For ideas on applying category theory to science and data, please see David I Spivak, *Category Theory for the Sciences*, MIT Press, 2014.)

The [Python `data_algebra` package](https://github.com/WinVector/data_algebra) supplies a number of operators for working with tabular data.  The operators are picked in reference to [Codd's relational algebra](https://en.wikipedia.org/wiki/Relational_algebra), though (as with [`SQL`](https://en.wikipedia.org/wiki/SQL)) we do not insist on table rows being unique. Many of the operations are simple: selecting rows, selecting columns, joining tables.  Two of the operations stand out: projecting or aggregating rows, and extending tables with new derived columns.

An interesting point is: while the `data_algebra` operators are fairly generic: the operator pipelines that map a single table to a single table form the arrows of a category over a nice set of objects.

The objects of this category can be either of:

 * Sets of column names.
 * Maps of column names to column types (schema-like objects).
 
I will take a liberty and call these objects (with or without types) "single table schemas."

Our setup is easiest to explain with an example.  Let's work an example in `Python`.

First we import our packages and instantiate an example data frame.


```python
import pandas

from data_algebra.data_ops import *
from data_algebra.arrow import *

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
a1 = DataOpArrow(id_ops_b)
```

`a1` is a categorical theory arrow, it has the usual domain (arrow base, or incoming object), and co-domain (arrow head, or outgoing object) in a category of single-table schemas.


```python
a1.dom()
```




    {'g', 'i', 'v', 'x'}




```python
a1.cod()
```




    ['g', 'x', 'v', 'i', 'ngroup']



These are what are presented in the succinct presentation of the arrow.


```python
print(a1)
```

    [
      [ x, g, i, v ]
       ->
      [ g, x, v, i, ngroup ]
    ]
    


The arrow has a more detailed presentation, which is the realization of the operator pipeline as code.


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


We can think of our arrows (or obvious mappings of them) as being able to be applied to:
  * More arrows of the same type (composition).
  * Data (action or application).
  * Single table schemas (managing pre and post conditions).
  
The arrow can be converted to a more detailed arrow that records both incoming and outgoing column types in the domain and co-domain by the `.fit()` function.


```python
a1.fit(d)

print(a1)
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'> ]
    ]
    



```python
identity_left = a1.identity_arrow(a1.dom())
print(identity_left)
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
    ]
    



```python
identity_right = a1.identity_arrow(a1.cod())
print(identity_right)
```

    [
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'> ]
    ]
    


Arrows can be composed or applied by using the notation `a1.transform(d)` or the equivalent notation `d >> a1`. Note: we are not thinking of `>>` itself as an arrow, but as a symbol for composition of arrows (we used `>>` as it is one of the few operators not used by `Pandas`, which means using this operator makes it easier for our notation to work with `Pandas`).


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




```python
d >> a1
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



Up until now we have been showing how we work to obey the category theory axioms.  From here on we look at what does category theory do for us.  What it does is check correct composition and ensure full associativity of operations.

As is typical in category theory, there can be more than one arrow from a given object to given object.  For example the following is a different arrow with the same start and end.


```python
a1b = DataOpArrow(
    table_description. \
        extend({
            'ngroup': 0
        }))
a1b.fit(d)
print(a1b)
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'> ]
    ]
    


However, the `a1b` arrow represents a different operation than `a1`:


```python
a1b.transform(d)
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
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>4</td>
      <td>40.0</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>5</td>
      <td>50.0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c</td>
      <td>7</td>
      <td>70.0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c</td>
      <td>8</td>
      <td>80.0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>9</td>
      <td>90.0</td>
      <td>False</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The arrows can be composed exactly when the pre-conditions meet the post conditions.  

Here are two examples of violating the pre and post conditions.  The point is, the categorical conditions enforce the checking for us.  We can't compose arrows that don't match domain and range.  Up until now we have been setting things up to make the categorical machinery work, now this machinery will work for us and make the job of managing complex data transformations easier.


```python
cols2_too_small = [c for c in (set(id_ops_b.column_names) - set(['i']))]
ordered_ops = TableDescription('d2', cols2_too_small). \
    extend({
        'row_number': '_row_number()',
        'shift_v': 'v.shift()',
    },
    order_by=['x'],
    partition_by=['g'])
a2 = DataOpArrow(ordered_ops)
print(a2)
```

    [
      [ x, g, ngroup, v ]
       ->
      [ x, g, ngroup, v, row_number, shift_v ]
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
a2 = DataOpArrow(ordered_ops)
print(a2)
```

    [
      [ q, ngroup, x, g, i, v ]
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
a2 = DataOpArrow(ordered_ops)
print(a2)
```

    [
      [ ngroup, x, g, i, v ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v ]
    ]
    



```python
print(a1 >> a2)
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v ]
    ]
    


We can also enforce type invariants.


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
      [ ngroup: <class 'numpy.int64'>, x: <class 'str'>, g: <class 'str'>,
        i: <class 'numpy.bool_'>, v: <class 'numpy.float64'> ]
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
        g: <class 'str'>, i: <class 'numpy.bool_'>, v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'> ]
    ]
    


We can add yet another set of operations to our pipeline: computing a per-group variable `mean`.


```python
unordered_ops = TableDescription('d3', ordered_ops.column_names). \
    extend({
        'mean_v': 'v.mean()',
    },
    partition_by=['g'])
a3 = DataOpArrow(unordered_ops)
print(a3)
```

    [
      [ ngroup, x, g, shift_v, i, v, row_number ]
       ->
      [ g, x, v, i, ngroup, row_number, shift_v, mean_v ]
    ]
    



```python
print(a3.fit(a2.transform(a1.transform(d))))
```

    [
      [ ngroup: <class 'numpy.int64'>, x: <class 'numpy.int64'>,
        g: <class 'str'>, shift_v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, v: <class 'numpy.float64'>,
        row_number: <class 'numpy.int64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'> ]
    ]
    


The three arrows can form a composite pipeline that computes a number of interesting per-group statistics all at once.


```python
print(a1 >> a2 >> a3)
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'> ]
    ]
    


And, we the methods are fully associative (can be grouped in any sequence that is still in the original order).


```python
print((a1 >> a2) >> a3)
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'> ]
    ]
    



```python
print(a1 >> (a2 >> a3))
```

    [
      [ x: <class 'numpy.int64'>, g: <class 'str'>, i: <class 'numpy.bool_'>,
        v: <class 'numpy.float64'> ]
       ->
      [ g: <class 'str'>, x: <class 'numpy.int64'>, v: <class 'numpy.float64'>,
        i: <class 'numpy.bool_'>, ngroup: <class 'numpy.int64'>,
        row_number: <class 'numpy.int64'>, shift_v: <class 'numpy.float64'>,
        mean_v: <class 'numpy.float64'> ]
    ]
    


All the compositions are in fact the same arrow, aswe can see by using it on data.


```python
((a1 >> a2) >> a3).transform(d)
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
      <th>mean_v</th>
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
      <td>10.0</td>
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
      <td>45.0</td>
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
      <td>45.0</td>
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
      <td>80.0</td>
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
      <td>80.0</td>
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
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
(a1 >> (a2 >> a3)).transform(d)
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
      <th>mean_v</th>
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
      <td>10.0</td>
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
      <td>45.0</td>
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
      <td>45.0</td>
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
      <td>80.0</td>
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
      <td>80.0</td>
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
      <td>80.0</td>
    </tr>
  </tbody>
</table>
</div>



The combination operator `>>` is fully associative over the combination of data and arrows.

The underlying `data_algebra` steps compute and check very similar pre and post conditions, the arrow class is just making this look more explicitly like arrows moving through objects in category.

The data arrows operate over three different value domains:

 * single table schemas (transforming single table schemas)
 * their own arrow space (i.e. composition)
 * data frames (transforming data as an action)

