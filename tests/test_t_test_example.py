
import sqlite3
import numpy
import numpy.random


from data_algebra.data_ops import *
from data_algebra.cdata import *
import data_algebra.SQLite
import data_algebra.test_util


def test_t_test_example():
    # example from:
    # https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/solving_problems_using_data_algebra.ipynb

    # seed the pseudo-random generator for repeatability
    numpy.random.seed(1999)

    # choose our simulated number of observations
    n_obs = 1000

    d = data_algebra.default_data_model.pd.DataFrame({
        'group': numpy.random.choice(['a', 'b', 'c'], size=n_obs, replace=True),
        'value': numpy.random.normal(0, 1, size=n_obs),
        'sensor': numpy.random.choice(['s1', 's2'], size=n_obs, replace=True),
    })
    # make the b group have an actual difference in means of s1 versus s2
    group_b_sensor_s2_rows = (d['group'] == 'b') & (d['sensor'] == 's2')
    d.loc[group_b_sensor_s2_rows, 'value'] = d.loc[group_b_sensor_s2_rows, 'value'] + 0.5

    ops = (
        TableDescription(
         table_name='d',
         column_names=[
           'group', 'value', 'sensor']) .\
           extend({
            'group_mean': 'value.mean()',
            'group_size': '(1).sum()'},
           partition_by=['group']) .\
           extend({
            'sq_diff': '(value - group_mean) ** 2'}) .\
           extend({
            'group_var': 'sq_diff.mean()'},
           partition_by=['group']) .\
           extend({
            'group_var': '(group_var * group_size) / (group_size - 1)'}) .\
           drop_columns(['sq_diff']) .\
           project({
            'group_sensor_mean': 'value.mean()',
            'group_sensor_size': '(1).sum()',
            'group_size': 'group_size.mean()',
            'group_var': 'group_var.mean()'},
           group_by=['group', 'sensor']) .\
           extend({
            'group_sensor_est_var': 'group_var / group_sensor_size'}) .\
           drop_columns(['group_sensor_size', 'group_size', 'group_var']) .\
           convert_records(data_algebra.cdata.RecordMap(
               blocks_in=data_algebra.cdata.RecordSpecification(
               record_keys=['group'],
               control_table=data_algebra.default_data_model.pd.DataFrame({
               'sensor': ['s1', 's2'],
               'group_sensor_mean': ['group_sensor_mean_s1', 'group_sensor_mean_s2'],
               'group_sensor_est_var': ['group_sensor_est_var_s1', 'group_sensor_est_var_s2'],
               }),
               control_table_keys=['sensor']),
               blocks_out=None)) .\
           extend({
            'mean_diff': 'group_sensor_mean_s1 - group_sensor_mean_s2'}) .\
           extend({
            't': 'mean_diff / ((group_sensor_est_var_s1 + group_sensor_est_var_s2)).sqrt()'}) .\
           drop_columns(['group_sensor_mean_s1', 'group_sensor_est_var_s1', 'group_sensor_mean_s2', 'group_sensor_est_var_s2']) .\
           order_rows(['group'])
        )

    # res = ops.transform(d)
    db_handle = data_algebra.db_model.DBHandle(db_model=data_algebra.SQLite.SQLiteModel(), conn=None)

    expect = data_algebra.default_data_model.pd.DataFrame({
        'group': ['a', 'b', 'c'],
        'mean_diff': [-0.1227200059303106, -0.37230214817674523, 0.05174373615504563],
        't': [-1.1391763140653364, -3.3953523914502717, 0.4678435568035228],
        })

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)
