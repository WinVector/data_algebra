
import sqlite3

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import data, descr, describe_table, ex
import data_algebra.util
import data_algebra.SQLite


def test_join_where_merge():
    d_observations = data_algebra.data_model.default_data_model().pd.DataFrame({
        "target_date": [10, 10, 11, 11],
        "as_of_date": [8, 9, 9, 10],
        })
    d_actions = data_algebra.data_model.default_data_model().pd.DataFrame({
        "target_date": [11, 10, 11, 11],
        "action_date": [7, 9, 9, 10],
        "reservation_count": [3, 1, 2, 5],
        })
    # example query showing the effect we want
    sql_query = """
       SELECT
         o.target_date,
         o.as_of_date,
         SUM(COALESCE(d.reservation_count, 0)) AS reservation_count
       FROM
         d_observations o
       LEFT JOIN
         d_actions d
       ON
         o.target_date = d.target_date
         AND d.action_date <= o.as_of_date
       GROUP BY
         o.as_of_date,
         o.target_date
       ORDER BY
         o.as_of_date,
         o.target_date
    """
    with sqlite3.connect(":memory:") as conn:
        d_observations.to_sql("d_observations", conn, index=False)
        d_actions.to_sql("d_actions", conn, index=False)
        res = data_algebra.data_model.default_data_model().pd.read_sql_query(sql_query, conn)
    expect = data_algebra.data_model.default_data_model().pd.DataFrame({
        "target_date": [10, 10, 11, 11],
        "as_of_date": [8, 9, 9, 10],
        "reservation_count": [0, 1, 5, 10],
    })
    assert res.equals(expect)
    # build up a similar op chain as we can't merge so many operations or perform non-equi joins 
    # we have to use a concat/cumsum strategy
    ops = (
        #     1) concat observations points into as count-0 actions so every
        #        observation key is in the new actions table.
        descr(d_observations=d_observations)
            .project({}, group_by=["target_date", "as_of_date"])
            .extend({"reservation_count": 0})
            .map_columns({"as_of_date": "action_date"})
            .concat_rows(descr(d_actions=d_actions))
            # 2) aggregate actions to eliminate any duplicate keys we have have introduced
            .project({"reservation_count": "reservation_count.sum()"}, group_by=["target_date", "action_date"])
            # 3) use cumulative sum to get reservations known up to a given date instead of exactly on a given date
            .extend(
                {"reservation_count": "reservation_count.cumsum()"},
                partition_by=["target_date"],
                order_by=["action_date"]
                )
            # 4) equi-join to the observation specifications to pull out desired data
            .map_columns({"action_date": "as_of_date"})
            .natural_join(
                descr(d_observations=d_observations),
                on=["target_date", "as_of_date"],
                jointype="inner"
            )
            .order_rows(["target_date", "as_of_date"])
    )
    # try in Pandas
    res_da = ops.eval({ "d_actions": d_actions, "d_observations": d_observations})
    assert res_da.equals(expect)
    # try in sqlite3
    with sqlite3.connect(":memory:") as conn:
        d_observations.to_sql("d_observations", conn, index=False)
        d_actions.to_sql("d_actions", conn, index=False)
        res_sql2 = data_algebra.data_model.default_data_model().pd.read_sql_query(ops.to_sql(data_algebra.SQLite.SQLiteModel()), conn)
    assert res_sql2.equals(expect)
    # check throughout
    data_algebra.test_util.check_transform(ops=ops, data={"d_actions": d_actions, "d_observations": d_observations}, expect=expect)
