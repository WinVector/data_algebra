

```python
import pyspark
import pyspark.sql

import pandas

from data_algebra.data_ops import *
import data_algebra.SparkSQL
```


```python
d_local = pandas.DataFrame({
    'subjectID':[1, 1, 2, 2],
    'surveyCategory': [ "withdrawal behavior", "positive re-framing", "withdrawal behavior", "positive re-framing"],
    'assessmentTotal': [5, 2, 3, 4],
    'irrelevantCol1': ['irrel1']*4,
    'irrelevantCol2': ['irrel2']*4,
})
d_local
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
      <th>subjectID</th>
      <th>surveyCategory</th>
      <th>assessmentTotal</th>
      <th>irrelevantCol1</th>
      <th>irrelevantCol2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>withdrawal behavior</td>
      <td>5</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>positive re-framing</td>
      <td>2</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>withdrawal behavior</td>
      <td>3</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>positive re-framing</td>
      <td>4</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc = pyspark.SparkContext()
spark = pyspark.sql.SparkSession.builder.appName('pandasToSparkDF').getOrCreate()

d_spark = spark.createDataFrame(d_local)
d_spark
```




    DataFrame[subjectID: bigint, surveyCategory: string, assessmentTotal: bigint, irrelevantCol1: string, irrelevantCol2: string]




```python
d_spark.createOrReplaceTempView("d")
sql_df = spark.sql("SELECT * FROM d")
sql_df.show()
```

    +---------+-------------------+---------------+--------------+--------------+
    |subjectID|     surveyCategory|assessmentTotal|irrelevantCol1|irrelevantCol2|
    +---------+-------------------+---------------+--------------+--------------+
    |        1|withdrawal behavior|              5|        irrel1|        irrel2|
    |        1|positive re-framing|              2|        irrel1|        irrel2|
    |        2|withdrawal behavior|              3|        irrel1|        irrel2|
    |        2|positive re-framing|              4|        irrel1|        irrel2|
    +---------+-------------------+---------------+--------------+--------------+
    



```python
local_copy = pandas.DataFrame(sql_df.collect())
local_copy
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>withdrawal behavior</td>
      <td>5</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>positive re-framing</td>
      <td>2</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>withdrawal behavior</td>
      <td>3</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>positive re-framing</td>
      <td>4</td>
      <td>irrel1</td>
      <td>irrel2</td>
    </tr>
  </tbody>
</table>
</div>




```python
scale = 0.237

with data_algebra.env.Env(locals()) as env:
    ops = data_algebra.data_ops.describe_table(d_local, 'd'). \
        extend({'probability': '(assessmentTotal * scale).exp()'}). \
        extend({'total': 'probability.sum()'},
               partition_by='subjectID'). \
        extend({'probability': 'probability/total'}). \
        extend({'sort_key': '-probability'}). \
        extend({'row_number': '_row_number()'},
               partition_by=['subjectID'],
               order_by=['sort_key']). \
        select_rows('row_number == 1'). \
        select_columns(['subjectID', 'surveyCategory', 'probability']). \
        rename_columns({'diagnosis': 'surveyCategory'})
    
print(ops.to_python(pretty=True))
```

    TableDescription(
        table_name="d",
        column_names=[
            "subjectID",
            "surveyCategory",
            "assessmentTotal",
            "irrelevantCol1",
            "irrelevantCol2",
        ],
    ).extend({"probability": "(assessmentTotal * 0.237).exp()"}).extend(
        {"total": "probability.sum()"}, partition_by=["subjectID"]
    ).extend(
        {"probability": "probability / total"}
    ).extend(
        {"sort_key": "-probability"}
    ).extend(
        {"row_number": "_row_number()"}, partition_by=["subjectID"], order_by=["sort_key"]
    ).select_rows(
        "row_number == 1"
    ).select_columns(
        ["subjectID", "surveyCategory", "probability"]
    ).rename_columns(
        {"diagnosis": "surveyCategory"}
    )
    



```python
db_model = data_algebra.SparkSQL.SparkSQLModel()
sql = ops.to_sql(db_model, pretty=True)
print(sql)
```

    SELECT `probability`,
           `subjectid`,
           `surveycategory` AS `diagnosis`
    FROM (
    SELECT `surveycategory`,
           `probability`,
           `subjectid`
    FROM (
    SELECT `surveycategory`,
           `probability`,
           `subjectid`
    FROM (
    SELECT `surveycategory`,
           `probability`,
           `sort_key`,
           `subjectid`,
           ROW_NUMBER() OVER (PARTITION BY `subjectid`
                              ORDER BY `sort_key`) AS `row_number`
    FROM (
    SELECT `surveycategory`,
           `probability`,
           `subjectid`, ( -`probability` ) AS `sort_key` FROM ( SELECT `surveycategory`, `subjectid`, `probability` / `total` AS `probability` FROM ( SELECT `surveycategory`, `probability`, `subjectid`, SUM(`probability`) OVER ( PARTITION BY `subjectid`  )  AS `total` FROM ( SELECT `surveycategory`, `subjectid`, EXP((`assessmenttotal` * 0.237)) AS `probability` FROM ( SELECT `assessmenttotal`, `surveycategory`, `subjectid` FROM `d` ) `sq_0` ) `sq_1` ) `sq_2` ) `sq_3` ) `sq_4` ) `sq_5` WHERE `row_number` = 1 ) `sq_6` ) `sq_7`



```python
sql_q = spark.sql(sql)

```


```python
res = pandas.DataFrame(sql_q.collect())
res
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.670622</td>
      <td>1</td>
      <td>withdrawal behavior</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.558974</td>
      <td>2</td>
      <td>positive re-framing</td>
    </tr>
  </tbody>
</table>
</div>




```python
sc.stop()
```
