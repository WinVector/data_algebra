
import os
import polars as pl
from data_algebra.polars_model import PolarsModel
import data_algebra
import pytest


def test_polars_sql_1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "pokemon.csv")
    # from:
    #  https://pola-rs.github.io/polars-book/user-guide/sql.html
    pokemon = pl.read_csv(data_path).lazy()
    polars_model = PolarsModel()
    ops = (
        data_algebra.descr(pokemon=pokemon.collect())
            .project({"avg_attack_by_type": "mean(Attack)"}, group_by=["Type 1"])
    )
    sql_2 = polars_model.to_sql(ops)
    assert isinstance(sql_2, str)
    if False:
        # Polars SQLContext some sort of non-rentrant singleton, so don't trigger it
        polars_sql = """
            SELECT 
                "Type 1",
                COUNT(DISTINCT "Type 2") AS count_type_2,
                AVG(Attack) AS avg_attack_by_type,
                MAX(Speed) AS max_speed
            FROM pokemon
            GROUP BY "Type 1"
            """
        res_1 = polars_model.eval_as_sql(polars_sql, data_map={"pokemon": pokemon})
        assert isinstance(res_1, pl.dataframe.frame.DataFrame)
    if False:
        # Polars SQL very limited right now, so below does not work
        res_2 = polars_model.eval_as_sql(ops, data_map={"pokemon": pokemon})
        assert isinstance(res_2, pl.dataframe.frame.DataFrame)

