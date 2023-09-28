import numpy
import numpy.random

from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex
from data_algebra.cdata import pivot_blocks_to_rowrecs,  pivot_rowrecs_to_blocks, pivot_specification, unpivot_specification, RecordMap, RecordSpecification 
import data_algebra.test_util
import scipy.stats


# example from earlier draft of
# https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/solving_problems_using_data_algebra.ipynb
def test_t_test_example():
    # seed the pseudo-random generator for repeatability
    numpy.random.seed(1999)

    # choose our simulated number of observations
    n_obs = 1000

    d = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": numpy.random.choice(["a", "b", "c"], size=n_obs, replace=True),
            "value": numpy.random.normal(0, 1, size=n_obs),
            "sensor": numpy.random.choice(["s1", "s2"], size=n_obs, replace=True),
        }
    )
    # make the b group have an actual difference in means of s1 versus s2
    group_b_sensor_s2_rows = (d["group"] == "b") & (d["sensor"] == "s2")
    d.loc[group_b_sensor_s2_rows, "value"] = (
        d.loc[group_b_sensor_s2_rows, "value"] + 0.5
    )

    ops = (
        TableDescription(table_name="d", column_names=["group", "value", "sensor"])
        .extend(
            {"group_mean": "value.mean()", "group_size": "(1).sum()"},
            partition_by=["group"],
        )
        .extend({"sq_diff": "(value - group_mean) ** 2"})
        .extend({"group_var": "sq_diff.mean()"}, partition_by=["group"])
        .extend({"group_var": "(group_var * group_size) / (group_size - 1)"})
        .drop_columns(["sq_diff"])
        .project(
            {
                "group_sensor_mean": "value.mean()",
                "group_sensor_size": "(1).sum()",
                "group_size": "group_size.mean()",
                "group_var": "group_var.mean()",
            },
            group_by=["group", "sensor"],
        )
        .extend({"group_sensor_est_var": "group_var / group_sensor_size"})
        .drop_columns(["group_sensor_size", "group_size", "group_var"])
        .convert_records(
            data_algebra.cdata.RecordMap(
                blocks_in=data_algebra.cdata.RecordSpecification(
                    record_keys=["group"],
                    control_table=data_algebra.data_model.default_data_model().pd.DataFrame(
                        {
                            "sensor": ["s1", "s2"],
                            "group_sensor_mean": [
                                "group_sensor_mean_s1",
                                "group_sensor_mean_s2",
                            ],
                            "group_sensor_est_var": [
                                "group_sensor_est_var_s1",
                                "group_sensor_est_var_s2",
                            ],
                        }
                    ),
                    control_table_keys=["sensor"],
                )
            )
        )
        .extend({"mean_diff": "group_sensor_mean_s1 - group_sensor_mean_s2"})
        .extend(
            {
                "t": "mean_diff / ((group_sensor_est_var_s1 + group_sensor_est_var_s2)).sqrt()"
            }
        )
        .drop_columns(
            [
                "group_sensor_mean_s1",
                "group_sensor_est_var_s1",
                "group_sensor_mean_s2",
                "group_sensor_est_var_s2",
            ]
        )
        .order_rows(["group"])
    )

    expect = data_algebra.data_model.default_data_model().pd.DataFrame(
        {
            "group": ["a", "b", "c"],
            "mean_diff": [
                -0.1227200059303106,
                -0.37230214817674523,
                0.05174373615504563,
            ],
            "t": [-1.1391763140653364, -3.3953523914502717, 0.4678435568035228],
        }
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)

    ops_join = (
        describe_table(d, table_name="d")
        .extend({"group_mean": "value.mean()"}, partition_by=["group"])
        .extend({"sq_diff": "(value - group_mean)**2"})
        .project(
            {"group_var": "sq_diff.mean()", "group_size": "(1).sum()",},
            group_by=["group"],
        )
        .extend({"group_var": "group_var * group_size / (group_size - 1)"})
        .natural_join(
            b=describe_table(d, table_name="d").project(
                {
                    "group_sensor_mean": "value.mean()",
                    "group_sensor_size": "(1).sum()",
                },
                group_by=["group", "sensor"],
            ),
            by=["group"],
            jointype="inner",
        )
        .extend({"group_sensor_est_var": "group_var / group_sensor_size"})
        .convert_records(
            data_algebra.cdata.RecordMap(
                blocks_in=data_algebra.cdata.RecordSpecification(
                    record_keys=["group"],
                    control_table=data_algebra.data_model.default_data_model().pd.DataFrame(
                        {
                            "sensor": ["s1", "s2"],
                            "group_sensor_mean": [
                                "group_sensor_mean_s1",
                                "group_sensor_mean_s2",
                            ],
                            "group_sensor_est_var": [
                                "group_sensor_est_var_s1",
                                "group_sensor_est_var_s2",
                            ],
                        }
                    ),
                    control_table_keys=["sensor"],
                )
            )
        )
        .extend({"mean_diff": "group_sensor_mean_s1 - group_sensor_mean_s2"})
        .extend(
            {
                "t": "mean_diff / (group_sensor_est_var_s1 + group_sensor_est_var_s2).sqrt()"
            }
        )
        .drop_columns(
            [
                "group_sensor_mean_s1",
                "group_sensor_est_var_s1",
                "group_sensor_mean_s2",
                "group_sensor_est_var_s2",
            ]
        )
        .order_rows(["group"])
    )

    data_algebra.test_util.check_transform(ops=ops, data=d, expect=expect)


# example from earlier draft of
# https://github.com/WinVector/data_algebra/blob/main/Examples/GettingStarted/solving_problems_using_data_algebra.ipynb
def test_t_test_example_2():
    # build synthetic example data
    # seed the pseudo-random generator for repeatability
    numpy.random.seed(1999)
    # choose our simulated number of observations
    n_obs = 1000
    d = data_algebra.data_model.default_data_model().pd.DataFrame({
        'group': numpy.random.choice(['a', 'b', 'c'], size=n_obs, replace=True),
        'value': numpy.random.normal(0, 1, size=n_obs),
        'sensor': numpy.random.choice(['s1', 's2'], size=n_obs, replace=True),
    })
    # make the b group have an actual difference in means of s1 versus s2
    group_b_sensor_s2_rows = (d['group'] == 'b') & (d['sensor'] == 's2')
    d.loc[group_b_sensor_s2_rows, 'value'] = d.loc[group_b_sensor_s2_rows, 'value'] + 0.5
    # get standard result
    groups = list(set(d['group']))
    groups.sort()
    d_grouped = d.groupby(['group'])

    def f(g):
        d_sub = d_grouped.get_group(g)
        v_s1 = d_sub.loc[d_sub['sensor'] == 's1', 'value']
        v_s2 = d_sub.loc[d_sub['sensor'] == 's2', 'value']
        res_g = scipy.stats.ttest_ind(v_s1, v_s2, equal_var=False)
        return data_algebra.data_model.default_data_model().pd.DataFrame({
            'group': [g],
            't': [res_g.statistic],
            'significance': [res_g.pvalue],
        })

    group_stats = [f(g) for g in groups]
    group_stats = data_algebra.data_model.default_data_model().pd.concat(group_stats).reset_index(inplace=False, drop=True)
    # define our operators
    td = descr(d=d)
    ops_var = (
        td
            .project(
            {
                'group_sensor_var': 'value.var()',  # estimate variance of items
                'group_sensor_mean': 'value.mean()',  # estimate mean of items
                'group_sensor_n': '(1).sum()',  # sample sizes
            },
            group_by=['group', 'sensor'])
            .extend(  # get the variance of the mean estimate
            {'group_sensor_est_var': 'group_sensor_var / group_sensor_n'})
    )
    # cdata steps
    a = data_algebra.data_model.default_data_model().pd.DataFrame({
        'group': ['a', 'a'],
        'sensor': ['s1', 's2'],
        'group_sensor_mean': [-0.103881, 0.018839],
        'group_sensor_est_var': [0.007211, 0.004606],
    })
    b = data_algebra.data_model.default_data_model().pd.DataFrame({
        'group': ['a'],
        'group_sensor_mean_s1': [-0.103881],
        'group_sensor_mean_s2': [0.018839],
        'group_sensor_est_var_s1': [0.007211],
        'group_sensor_est_var_s2': [0.004606],
    })
    record_in = data_algebra.data_model.default_data_model().pd.DataFrame({
        'sensor': ['s1', 's2'],
        'group_sensor_mean': ['group_sensor_mean_s1', 'group_sensor_mean_s2'],
        'group_sensor_est_var': ['group_sensor_est_var_s1', 'group_sensor_est_var_s2'],
    })
    record_map = RecordMap(
        blocks_in=RecordSpecification(
            control_table=record_in,
            record_keys=['group'],
            control_table_keys=["sensor"],
        ),
    )
    a_transform = record_map.transform(a)
    assert data_algebra.test_util.equivalent_frames(a_transform, b)
    ops = (
        ops_var
            .convert_records(record_map)
    )
    ops = (
        ops
            .extend({'mean_diff': 'group_sensor_mean_s1 - group_sensor_mean_s2'})
            .extend({'t': 'mean_diff / (group_sensor_est_var_s1 + group_sensor_est_var_s2).sqrt()'})
            .drop_columns(['group_sensor_mean_s1', 'group_sensor_est_var_s1',
                           'group_sensor_mean_s2', 'group_sensor_est_var_s2'])
            .order_rows(['group'])
    )
    pandas_res = ops.transform(d)
    assert data_algebra.test_util.equivalent_frames(
        pandas_res.loc[:, ['group', 't']],
        group_stats.loc[:, ['group', 't']],
        float_tol=0.01)
    data_algebra.test_util.check_transform(ops=ops, data=d, expect=pandas_res)
