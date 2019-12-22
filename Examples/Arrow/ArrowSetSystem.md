I've been writing a lot about a [category theory interpretations of data-processing pipelines](http://www.win-vector.com/blog/2019/12/data_algebra-rquery-as-a-category-over-table-descriptions/) and [some of the improvements we feel it is driving](http://www.win-vector.com/blog/2019/12/better-sql-generation-via-the-data_algebra/) in both the [`data_algebra`](https://github.com/WinVector/data_algebra) and in [`rquery`](https://github.com/WinVector/rquery)/[`rqdatatable`](https://github.com/WinVector/rqdatatable).

I think I've found an even better category theory re-formulation of the package, which I will describe here.

In the [earlier formalism](http://www.win-vector.com/blog/2019/12/data_algebra-rquery-as-a-category-over-table-descriptions/) our data transform pipelines were arrows over a category of sets of column names (sets of strings).  

These pipelines acted on `Pandas` tables or `SQL` tables, with one table marked as special.  Marking one table as special (or using a "pointed set" notation) lets us use a nice compositional notation, without having to appeal to something like operads.  The treating one table as the one of interest is fairly compatible with data science, as in data science often when working with many tables one is the primary model-frame and the rest are used to join in additional information.

The above formulation was really working well. But we have found a variation of the `data_algebra` with an even neater formalism.

The `data_algebra` objects have a very nice interpretation as arrows in a category whose objects are set families described by:

 * a set of required columns.
 * a set of forbidden columns.
 
The arrows `a` and `b` compose as `a >> b` as long as:

 * All of the columns required by `b` are produced by `a`.
 * None of the columns forbidden by `b` are produced by `a`.
 
This is still an equality check of domains and co-domains, so as long as we maintain associativity we still have a nice category.

We can illustrate the below.

First we import our modules.


```python
import sqlite3

import pandas

from data_algebra.data_ops import *
from data_algebra.arrow import fmt_as_arrow
import data_algebra.SQLite
```

We define our first arrow which is a transform that creates a new column `x` as the sum of the columns `a` and `b`.


```python
a = TableDescription(table_name='table_a', column_names=['a', 'b']). \
        extend({'c': 'a + b'})

a
```




    TableDescription(
     table_name='table_a',
     column_names=[
       'a', 'b']) .\
       extend({
        'c': 'a + b'})




```python
print(fmt_as_arrow(a))
```

    [
     'table_a':
      at least [ a, b ]
       ->
      at least [ a, b, c ]
    ]
    


And we define our second arrow, `b`, which renames the column `a` to a new column name `x`.


```python
b = TableDescription(table_name='table_b', column_names=['a']). \
        rename_columns({'x': 'a'})

b
```




    TableDescription(
     table_name='table_b',
     column_names=[
       'a']) .\
       rename_columns({'x': 'a'})




```python
print(fmt_as_arrow(b))
```

    [
     'table_b':
      at least [ a ] , and none of [ x ]
       ->
      at least [ x ]
    ]
    


The rules are met, so we can combine these two arrows.


```python
ab = a >> b

ab
```




    TableDescription(
     table_name='table_a',
     column_names=[
       'a', 'b']) .\
       extend({
        'c': 'a + b'}) .\
       rename_columns({'x': 'a'})




```python
print(fmt_as_arrow(ab))
```

    [
     'table_a':
      at least [ a, b ] , and none of [ x ]
       ->
      at least [ b, c, x ]
    ]
    


Notice this produces a new arrow `ab` with appropriate required and forbidden columns.  By associativity (one of the primary properties needed to be a category) we get that the arrow `ab` has an action on data frames the same as using the `a` action followed by the `b` action.

Let's illustrate that here.


```python
d = pandas.DataFrame({
    'a': [1, 2],
    'b': [30, 40]
})

d
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
b.act_on(a.act_on(d))
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>40</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
ab.act_on(d)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>40</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>



`.act_on()` copies forward all columns consistent with the transform specification and used at the output.  Missing columns are excess columns are checked for at the start of a calculation.


```python
excess_frame = pandas.DataFrame({
    'a': [1], 
    'b': [2], 
    'd': [3],
    'x': [4]})

try:
    ab.act_on(excess_frame)
except ValueError as ve:
    print("caught ValueError: " + str(ve))
```

    caught ValueError: Table table_a has forbidden columns: {'x'}


The `.transform()` method, on the other hand, copies forward only declared columns.


```python
ab.transform(excess_frame)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Notice in the above that the input `x` did not interfere with the calculation, and `d` was not copied forward.  The idea is behavior during composition is very close to behavior during action/application, so we find more issues during composition.

However, `.transform()` does not associate with composition as we have `b.transform(a.transform(d))` is not equal to `ab.transform(d)`.


```python
b.transform(a.transform(d))
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
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



In both cases we still have result-oriented narrowing.


```python
c = TableDescription(table_name='table_c', column_names=['a', 'b', 'c']). \
        extend({'x': 'a + b'}). \
        select_columns({'x'})

c
```




    TableDescription(
     table_name='table_c',
     column_names=[
       'a', 'b', 'c']) .\
       extend({
        'x': 'a + b'}) .\
       select_columns(['x'])




```python
print(fmt_as_arrow(c))
```

    [
     'table_c':
      at least [ a, b, c ]
       ->
      at least [ x ]
    ]
    



```python
table_c = pandas.DataFrame({
    'a': [1, 2],
    'b': [30, 40],
    'c': [500, 600],
    'd': [7000, 8000]
})

table_c
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30</td>
      <td>500</td>
      <td>7000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>40</td>
      <td>600</td>
      <td>8000</td>
    </tr>
  </tbody>
</table>
</div>




```python
c.act_on(table_c)
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
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
c.transform(table_c)
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
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>



`.select_columns()` conditions are propagated back through the calculation.

Another useful operator is `.drop_columns()` which drops columns if they are present, but does not raise an issue if the columns to be removed are already not present.  `.drop_columns()` can be used to guarantee forbidden columns are not present.  We could use `.act_on()` or `excess_frame` using `.drop_columns()` as follows.


```python
tdr = describe_table(excess_frame).drop_columns(['x'])

tdr
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'a', 'b', 'd', 'x']) .\
       drop_columns(['x'])




```python
rab = tdr >> ab

rab
```




    TableDescription(
     table_name='data_frame',
     column_names=[
       'a', 'b', 'd', 'x']) .\
       drop_columns(['x']) .\
       extend({
        'c': 'a + b'}) .\
       rename_columns({'x': 'a'})



The `>>` notation is composing the arrows.  `tdr >> ab` is syntactic sugar for `ab.apply_to(tdr)`. Both of these are the arrow composition operations.


```python
rab.act_on(excess_frame)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>b</th>
      <th>d</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Remember, the original `ab` operator rejects `excess_frame`.


```python
try:
    ab.act_on(excess_frame)
except ValueError as ve:
    print("caught ValueError: " + str(ve))
```

    caught ValueError: Table table_a has forbidden columns: {'x'}


We can also adjust the input-specification by composing pipelines with table descriptions.


```python
a
```




    TableDescription(
     table_name='table_a',
     column_names=[
       'a', 'b']) .\
       extend({
        'c': 'a + b'})




```python
bigger = TableDescription(table_name='bigger', column_names=['a', 'b', 'x', 'y', 'z'])

bigger
```




    TableDescription(
     table_name='bigger',
     column_names=[
       'a', 'b', 'x', 'y', 'z'])




```python
bigger_a = bigger >> a

bigger_a
```




    TableDescription(
     table_name='bigger',
     column_names=[
       'a', 'b', 'x', 'y', 'z']) .\
       extend({
        'c': 'a + b'})




```python
print(fmt_as_arrow(bigger_a))
```

    [
     'bigger':
      at least [ a, b, x, y, z ]
       ->
      at least [ a, b, c, x, y, z ]
    ]
    


Notice the new arrow (`bigger_a`) has a wider input specification. Appropriate checking is performed during the composition.

As always, we can also translate any of our operators to `SQL`.


```python
db_model = data_algebra.SQLite.SQLiteModel()

print(bigger_a.to_sql(db_model=db_model, pretty=True))
```

    SELECT "a" + "b" AS "c",
           "a",
           "b",
           "x",
           "y",
           "z"
    FROM "bigger"


The `SQL` translation is similar to `.transform()` in that it only refers to known columns by name.  This means we are safe from extra columns in the source tables. This means if we did derive an action acting on `SQL` or composition over `SQL` it would not associate with the `data_algebra` operator composition (just as `.transform()` did not).

Notice we no longer have to use the arrow-adapter classes (except for formatting), the `data_algebra` itself has been adjusted to a more direct categorical basis.

And that is some of how the `data_algebra` works on our new set-oriented category.  In this formulation much less annotation is required from the user, while still allowing very detailed record-keeping.  The detailed record-keeping lets us find issues while assembling the pipelines, not later when working with potentially large/slow data.


```python

```

