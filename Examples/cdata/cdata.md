
This note is a simple data wrangling example worked using both the Python [data_algebra](https://github.com/WinVector/data_algebra) package and the [R](https://www.r-project.org) [cdata](https://github.com/WinVector/cdata) package. Both of these packages make data wrangling easy through he use of [coordinatized data](http://www.win-vector.com/blog/tag/coordinatized-data/) concepts  (relying heavily on [Codd's "rule of access"](https://en.wikipedia.org/wiki/Codd%27s_12_rules)).

The advantages of [data_algebra](https://github.com/WinVector/data_algebra) and [cdata](https://github.com/WinVector/cdata) are:

 * The user specifies their desired transform declaratively *by example* and *in data*.  What one does is: work an example, and then write down what you want (we have a tutorial on this [here](https://winvector.github.io/cdata/articles/design.html)).
    
 * The transform systems can print what a transform is going to do.  This makes reasoning about data transforms *much* easier.
 
 * The transforms, as they themselves are written as data, can be easily shared between systems (such as R and Python).
 
Let's re-work a small [R cdata example](https://github.com/WinVector/cdata/blob/master/vignettes/control_table_keys.Rmd), using the Python package [data_algebra](https://github.com/WinVector/data_algebra).

## An Example

First we import some modules and packages, and import some notional data.


```python
# https://pandas.pydata.org
import pandas

# PyYAML from https://pyyaml.org
import yaml

# pip install data_algebra
# data_algebra from https://github.com/WinVector/data_algebra/
import data_algebra.cdata
import data_algebra.cdata_impl
import data_algebra.data_ops
import data_algebra.yaml
import data_algebra.SQLite

# ask YAML to write simpler structures
data_algebra.yaml.fix_ordered_dict_yaml_rep()

# read our example data
iris = pandas.read_csv('iris_small.csv')
# print it out
iris
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



Our goal is to move from this de-normalized or wide-form (or "model matrix"/"data matrix" form, where each record is exactly one row) into a tall form where records may span more than one row.

Or, more concretely, we want our data to look like the following.


```python
answer = pandas.read_csv("answer.csv")
answer
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



Notice each row of the original data set is now four rows of the derived one. This "tall form" is often useful for plotting.

This sort of conversion can be called an anti-pivot. In Python these sorts of transforms are specified with [pandas.DataFrame.pivot](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html), [pandas.pivot_table](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html), [pandas.melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html), 
and others. Many [R](https://www.r-project.org) packages supply related operators: [reshape]( https://CRAN.R-project.org/package=reshape), [reshape2](https://CRAN.R-project.org/package=reshape2), [data.table](https://CRAN.R-project.org/package=data.table), [tidyr](https://CRAN.R-project.org/package=tidyr), [cdata](https://CRAN.R-project.org/package=cdata), and more. We feel the [data_algebra](https://github.com/WinVector/data_algebra) and [cdata](https://CRAN.R-project.org/package=cdata) methods (which only perform data-reshaping, and not aggregation) offer a number of significant advantages (some of which we have already mentioned, and a few more of which we will see demonstrated in this note).

Back to our example.

To specify our desired transform, we build a structure describing what a data record looks like. We can crib this specification from the answer as follows.


```python
control_table = answer.loc[answer.id == 0, ['Part', 'Measure']]
control_table = control_table.reset_index(inplace=False, drop=True)
control_table["Value"] = [control_table['Part'][i] + '.' + control_table['Measure'][i] for 
                            i in range(control_table.shape[0])]
control_table
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
      <th>Part</th>
      <th>Measure</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Petal</td>
      <td>Length</td>
      <td>Petal.Length</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Petal</td>
      <td>Width</td>
      <td>Petal.Width</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sepal</td>
      <td>Length</td>
      <td>Sepal.Length</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sepal</td>
      <td>Width</td>
      <td>Sepal.Width</td>
    </tr>
  </tbody>
</table>
</div>



We can derive the control table from the answer, as we did here, or just type one in directly.  The idea is: we can use any method we want to derive the prototype record shape, we are not limited to a sub-query language or methodology from any one package.

For each record we take care to identify what keys identify records (the `record_keys`) and want parts identify rows within the record (the `control_table_keys`).  We suppress the `record_key` when writing the control table, as these are exactly the values that do not vary inside each record.

We combine all of these specification into a `RecordSpecification` as follows:


```python
record_spec = data_algebra.cdata.RecordSpecification(
    control_table,
    control_table_keys = ['Part', 'Measure'],
    record_keys = ['id', 'Species']
    )
record_spec
```




    RecordSpecification
       record_keys: ['id', 'Species']
       control_table_keys: ['Part', 'Measure']
       control_table:
           Part Measure         Value
       0  Petal  Length  Petal.Length
       1  Petal   Width   Petal.Width
       2  Sepal  Length  Sepal.Length
       3  Sepal   Width   Sepal.Width



In this notation any cells of the control table from columns that are not `control_table_keys` are "value stand ins".  During data transfrom these cells will be replaced by values coming from the columns named by these cells.

Now we can transform our original row-record oriented data into general block records.  To do this we specify a `RecordMap` using our record specification to describe the outgoing record structure. The incoming record structure is implicitly assumed to be single-row records, unless we specify otherwise (using the `blocks_in` argument).


```python
mp_to_blocks = data_algebra.cdata_impl.RecordMap(blocks_out=record_spec)
print(str(mp_to_blocks))
```

    Transform row records of the form:
      record_keys: ['id', 'Species']
     ['id', 'Species', 'Petal.Length', 'Petal.Width', 'Sepal.Length', 'Sepal.Width']
    to block records of structure:
    RecordSpecification
       record_keys: ['id', 'Species']
       control_table_keys: ['Part', 'Measure']
       control_table:
           Part Measure         Value
       0  Petal  Length  Petal.Length
       1  Petal   Width   Petal.Width
       2  Sepal  Length  Sepal.Length
       3  Sepal   Width   Sepal.Width
    


And we are ready to apply our specified transform.


```python
arranged_blocks = mp_to_blocks.transform(iris)
arranged_blocks
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



We see the operation has been performed for us. Notice we specify the transform *declaratively* with data structures carrying deceptions of what we want, instead of having to build a sequence of verbs that realize the transformation.

### Inverting the transform

An inverse transform is simply expressed by reversing the roles of the `blocks_out` and `blocks_in` arguments. In this case the output is row-records, as we didn't specify an outgoing block structure with `blocks_out`.


```python
mp_to_rows = data_algebra.cdata_impl.RecordMap(blocks_in=record_spec)
print(str(mp_to_rows))
```

    Transform block records of structure:
    RecordSpecification
       record_keys: ['id', 'Species']
       control_table_keys: ['Part', 'Measure']
       control_table:
           Part Measure         Value
       0  Petal  Length  Petal.Length
       1  Petal   Width   Petal.Width
       2  Sepal  Length  Sepal.Length
       3  Sepal   Width   Sepal.Width
    to row records of the form:
      record_keys: ['id', 'Species']
     ['id', 'Species', 'Petal.Length', 'Petal.Width', 'Sepal.Length', 'Sepal.Width']
    



```python
arranged_rows = mp_to_rows.transform(arranged_blocks)
arranged_rows
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
      <th>id</th>
      <th>Species</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>setosa</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>5.1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>setosa</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>4.9</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>setosa</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>4.7</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>



## Arbitrary transforms

Arbitrary record to record transforms can be specified by setting both `blocks_in` (to describe incoming structure) and `blocks_out` (to describe outgoing structure) at the same time.  

## Transforms in databases

`data_algebra` also implements all the transform steps in databases using [`SQL`](https://en.wikipedia.org/wiki/SQL) (via `row_recs_to_blocks_query()` and `blocks_to_row_recs_query()`).

These queries can be seen below.


```python
db_model = data_algebra.SQLite.SQLiteModel()
```


```python
print(db_model.row_recs_to_blocks_query(
    source_view=data_algebra.data_ops.describe_table(
        iris, 'iris'),
    record_spec=record_spec,
    record_view=data_algebra.data_ops.describe_table(
        record_spec.control_table, "control_table")
))
```

    SELECT
     a."id" AS "id",
     a."Species" AS "Species",
     b."Part" AS "Part",
     b."Measure" AS "Measure",
     CASE
      WHEN b."Value" = 'Petal.Length' THEN a."Petal.Length"
      WHEN b."Value" = 'Petal.Width' THEN a."Petal.Width"
      WHEN b."Value" = 'Sepal.Length' THEN a."Sepal.Length"
      WHEN b."Value" = 'Sepal.Width' THEN a."Sepal.Width"
      ELSE NULL END AS "Value"
    FROM (
      "iris" ) a
    CROSS JOIN (
      "control_table" ) b
     ORDER BY a."id", a."Species", b."Part", b."Measure"



```python
print(db_model.blocks_to_row_recs_query(
    source_view=data_algebra.data_ops.describe_table(
        iris, 'iris'),
    record_spec=record_spec
))
```

    SELECT
     "id" AS "id",
     "Species" AS "Species",
     MAX(CASE WHEN  ( "Part" = 'Petal' )  AND  ( "Measure" = 'Length' )  THEN "Value" ELSE NULL END) AS "Petal.Length",
     MAX(CASE WHEN  ( "Part" = 'Petal' )  AND  ( "Measure" = 'Width' )  THEN "Value" ELSE NULL END) AS "Petal.Width",
     MAX(CASE WHEN  ( "Part" = 'Sepal' )  AND  ( "Measure" = 'Length' )  THEN "Value" ELSE NULL END) AS "Sepal.Length",
     MAX(CASE WHEN  ( "Part" = 'Sepal' )  AND  ( "Measure" = 'Width' )  THEN "Value" ELSE NULL END) AS "Sepal.Width"
    FROM (
      "iris"
     )
     GROUP BY "id", "Species"
     ORDER BY "id", "Species"


The use case for `SQL` queries is: the queries can be used to with "`CREATE TABLE table_name AS`" to materialize transform results in a database, without round-tripping the data in and out of the database.

As complicated as the queries look, they actually expose some deep truths:

  * The `row_recs_to_blocks_query()` is essentially a cross-join of the data to the record description.  Each combination of data row and record description row builds a new result row.

  * The `blocks_to_row_recs_query()` is an aggregation.  Each set of rows corresponding to a given data record is aggregated into a single result row.
  
  * Just about any arbitrary record shape to arbitrary record shape can be written as a transform from the first record shape to row-records (record sets that have exactly one row per record), followed by a transform from the row-records to the new format. This transform can preserve column types as in the intermediate form each different record entry has its own column.  This is an advantage of using a "thin" intermediate form such as [RDF triples](https://en.wikipedia.org/wiki/Semantic_triple).

This leads us to believe that transforming to and from single-row records are in fact fundemental operations, and not just implementation details.

## The R `cdata` version

The `data_algebra` had been previously implemented in 
[R](https://www.r-project.org) in the [`cdata`](https://github.com/WinVector/cdata), [`rquery`](https://github.com/WinVector/rquery), and [`rqdatatable`](https://github.com/WinVector/rqdatatable) packages.
This packages support both in-memory operations and translation of the transforms to SQL.

We would perform the above transforms in R as follows.

First we write out a copy of the transform.


```python
# convert objects to a YAML string
xform_rep = yaml.dump(mp_to_blocks.to_simple_obj())
# write to file
with open("xform_yaml.txt", "wt") as f:
    print(xform_rep, file=f)
```

Now we we start up our R environment and read in the transform specification and data.


```python
%load_ext rpy2.ipython
```


```r
%%R

# install.packages("cdata")
library(cdata)
library(yaml)

iris <- read.csv('iris_small.csv')
print(iris)
```

      Sepal.Length Sepal.Width Petal.Length Petal.Width Species id
    1          5.1         3.5          1.4         0.2  setosa  0
    2          4.9         3.0          1.4         0.2  setosa  1
    3          4.7         3.2          1.3         0.2  setosa  2



```r
%%R

r_yaml <- yaml.load_file("xform_yaml.txt")
transform <- convert_yaml_to_cdata_spec(r_yaml)

print(transform)
```

    {
     row_record <- wrapr::qchar_frame(
       "id"  , "Species", "Petal.Length", "Petal.Width", "Sepal.Length", "Sepal.Width" |
         .   , .        , Petal.Length  , Petal.Width  , Sepal.Length  , Sepal.Width   )
     row_keys <- c('id', 'Species')
    
     # becomes
    
     block_record <- wrapr::qchar_frame(
       "id"  , "Species", "Part" , "Measure", "Value"      |
         .   , .        , "Petal", "Length" , Petal.Length |
         .   , .        , "Petal", "Width"  , Petal.Width  |
         .   , .        , "Sepal", "Length" , Sepal.Length |
         .   , .        , "Sepal", "Width"  , Sepal.Width  )
     block_keys <- c('id', 'Species', 'Part', 'Measure')
    
     # args: c(checkNames = TRUE, checkKeys = FALSE, strict = FALSE, allow_rqdatatable = TRUE)
    }
    


Now that we have recovered the transform, we can use it in R.


```r
%%R

iris %.>% transform
```

       id Species  Part Measure Value
    1   0  setosa Petal  Length   1.4
    2   0  setosa Petal   Width   0.2
    3   0  setosa Sepal  Length   5.1
    4   0  setosa Sepal   Width   3.5
    5   1  setosa Petal  Length   1.4
    6   1  setosa Petal   Width   0.2
    7   1  setosa Sepal  Length   4.9
    8   1  setosa Sepal   Width   3.0
    9   2  setosa Petal  Length   1.3
    10  2  setosa Petal   Width   0.2
    11  2  setosa Sepal  Length   4.7
    12  2  setosa Sepal   Width   3.2


### Cross-language work

As the record transform specifications, both in Python `data_algebra` and R `cata` are simple data structures (just the control table, and a few lists of key column names), they can be moved from one language to another by `YAML` (as we also demonstrated in the [logistic scoring example](https://github.com/WinVector/data_algebra/blob/master/Examples/LogisticExample/ScoringExample.ipynb).

`data_algebra` supplies a write method, so cross-language interoperation is just a matter of adding additional read/write methods.


```python
print(yaml.dump(mp_to_blocks.to_simple_obj()))
```

    type: data_algebra.cdata_impl.RecordMap
    blocks_out:
      type: data_algebra.cdata.RecordSpecification
      record_keys:
      - id
      - Species
      control_table_keys:
      - Part
      - Measure
      control_table:
        Part:
        - Petal
        - Petal
        - Sepal
        - Sepal
        Measure:
        - Length
        - Width
        - Length
        - Width
        Value:
        - Petal.Length
        - Petal.Width
        - Sepal.Length
        - Sepal.Width
    


## Conclusion

The [`cdata`](https://github.com/WinVector/cdata) and [`data_algebra`](https://github.com/WinVector/data_algebra) systems offer powerful implementations, and deep understanding of the nature of record transformations.  They allow one to reshape data quickly and conveniently either in R. Python/[Pandas](https://pandas.pydata.org), or even SQL.
