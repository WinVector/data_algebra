
This is a quick re-work of the [`Keras` record transform example](http://winvector.github.io/FluidData/FluidDataReshapingWithCdata.html) in `Python`. For an `R` version please see [here](https://github.com/WinVector/cdata/blob/master/Examples/Inverse/Inverse.md).


In the [original article](http://winvector.github.io/FluidData/FluidDataReshapingWithCdata.html) we had `Keras` model performance data, which looked like the following.


```python
import pandas
import data_algebra.cdata
import data_algebra.cdata_impl


df = pandas.DataFrame({
    'val_loss': [-0.377, -0.2997, -0.2964, -0.2779, -0.2843, -0.312],
    'val_acc': [0.8722, 0.8895, 0.8822, 0.8899, 0.8861, 0.8817],
    'loss': [-0.5067, -0.3002, -0.2166, -0.1739, -0.1411, -0.1136],
    'acc': [0.7852, 0.904, 0.9303, 0.9428, 0.9545, 0.9656],
    'epoch': [1, 2, 3, 4, 5, 6],
    })

df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>val_loss</th>
      <th>val_acc</th>
      <th>loss</th>
      <th>acc</th>
      <th>epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.3770</td>
      <td>0.8722</td>
      <td>-0.5067</td>
      <td>0.7852</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.2997</td>
      <td>0.8895</td>
      <td>-0.3002</td>
      <td>0.9040</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.2964</td>
      <td>0.8822</td>
      <td>-0.2166</td>
      <td>0.9303</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.2779</td>
      <td>0.8899</td>
      <td>-0.1739</td>
      <td>0.9428</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.2843</td>
      <td>0.8861</td>
      <td>-0.1411</td>
      <td>0.9545</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.3120</td>
      <td>0.8817</td>
      <td>-0.1136</td>
      <td>0.9656</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




But for plotting, it is more convenient to have the data in the following form:

| epoch | measure                    | training | validation |
| ----: | :------------------------- | -------: | ---------: |
|     1 | minus binary cross entropy | \-0.5067 |   \-0.3770 |
|     1 | accuracy                   |   0.7852 |     0.8722 |
| ...                                                        |

[The article](http://winvector.github.io/FluidData/FluidDataReshapingWithCdata.html) uses ideas similar to [these](https://winvector.github.io/cdata/articles/design.html) to visualize the desired record structure and then write down this visualization as a concrete data record example.

The principle is: if you have a visualization of the input and output, it is then trivial to marshal these into a graphical representation of the desired transform. And if you can't work out what the input and output look like, then you really are not quite ready to perform the transform.  Knowing what we want is the minimum requirement and with this methodology it is also all that is needed.



```python
shape = pandas.DataFrame({
    'measure': ['minus binary cross entropy', 'accuracy'],
    'training': ['loss', 'acc'],
    'validation': ['val_loss', 'val_acc'],
    })

shape
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>measure</th>
      <th>training</th>
      <th>validation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>minus binary cross entropy</td>
      <td>loss</td>
      <td>val_loss</td>
    </tr>
    <tr>
      <th>1</th>
      <td>accuracy</td>
      <td>acc</td>
      <td>val_acc</td>
    </tr>
  </tbody>
</table>
</div>



This description of the desired record shape is easily transformed into a data transformation specification.


```python
record_map = data_algebra.cdata_impl.RecordMap(
    blocks_out=data_algebra.cdata.RecordSpecification(
        control_table=shape,
        record_keys=['epoch']
    ),
)

record_map
```




    data_algebra.cdata_impl.RecordMap(
        blocks_in=None,
        blocks_out=data_algebra.cdata.RecordSpecification(
        record_keys=['epoch'],
        control_table=pandas.DataFrame({
        'measure': ['minus binary cross entropy', 'accuracy'],
        'training': ['loss', 'acc'],
        'validation': ['val_loss', 'val_acc'],
        }),
        control_table_keys=['measure']))




```python
print(str(record_map))
```

    Transform row records of the form:
      record_keys: ['epoch']
     ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    to block records of structure:
    RecordSpecification
       record_keys: ['epoch']
       control_table_keys: ['measure']
       control_table:
                             measure training validation
       0  minus binary cross entropy     loss   val_loss
       1                    accuracy      acc    val_acc
    


Just about any transfrom we want can be specified through `data_algebra.cdata_impl.RecordMap` by specifying the `blocks_in` and `blocks_out` shapes (leaving these as `None` specifies the corresponding shape is a row record or record that is entirely in a single row).

We can easily apply this transform to our data.


```python
res = record_map.transform(df)

res
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>measure</th>
      <th>training</th>
      <th>validation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>accuracy</td>
      <td>0.7852</td>
      <td>0.8722</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>minus binary cross entropy</td>
      <td>-0.5067</td>
      <td>-0.3770</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>accuracy</td>
      <td>0.9040</td>
      <td>0.8895</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>minus binary cross entropy</td>
      <td>-0.3002</td>
      <td>-0.2997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>accuracy</td>
      <td>0.9303</td>
      <td>0.8822</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>minus binary cross entropy</td>
      <td>-0.2166</td>
      <td>-0.2964</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>accuracy</td>
      <td>0.9428</td>
      <td>0.8899</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>minus binary cross entropy</td>
      <td>-0.1739</td>
      <td>-0.2779</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>accuracy</td>
      <td>0.9545</td>
      <td>0.8861</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5</td>
      <td>minus binary cross entropy</td>
      <td>-0.1411</td>
      <td>-0.2843</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6</td>
      <td>accuracy</td>
      <td>0.9656</td>
      <td>0.8817</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
      <td>minus binary cross entropy</td>
      <td>-0.1136</td>
      <td>-0.3120</td>
    </tr>
  </tbody>
</table>
</div>



And it is simple to build an inverse transform.


```python
inv = record_map.inverse()

print(str(inv))
```

    Transform block records of structure:
    RecordSpecification
       record_keys: ['epoch']
       control_table_keys: ['measure']
       control_table:
                             measure training validation
       0  minus binary cross entropy     loss   val_loss
       1                    accuracy      acc    val_acc
    to row records of the form:
      record_keys: ['epoch']
     ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    


And equally easy to apply this inverse transform to data.


```python
inv.transform(res)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>loss</th>
      <th>val_loss</th>
      <th>acc</th>
      <th>val_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.5067</td>
      <td>-0.3770</td>
      <td>0.7852</td>
      <td>0.8722</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.3002</td>
      <td>-0.2997</td>
      <td>0.9040</td>
      <td>0.8895</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.2166</td>
      <td>-0.2964</td>
      <td>0.9303</td>
      <td>0.8822</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.1739</td>
      <td>-0.2779</td>
      <td>0.9428</td>
      <td>0.8899</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-0.1411</td>
      <td>-0.2843</td>
      <td>0.9545</td>
      <td>0.8861</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>-0.1136</td>
      <td>-0.3120</td>
      <td>0.9656</td>
      <td>0.8817</td>
    </tr>
  </tbody>
</table>
</div>



Notice how each step can be inspected and checked as we worked. I would definitely recommend re-reading [the original article](http://winvector.github.io/FluidData/FluidDataReshapingWithCdata.html) with the new transform notation in mind. In any case, please check out the `cdata` [package](https://github.com/WinVector/cdata) and [documentation](https://winvector.github.io/cdata/).

