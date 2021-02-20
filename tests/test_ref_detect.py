import numpy
import pandas

import pytest

from data_algebra.data_ops import *


def test_ref_detect_1():
    d = pandas.DataFrame(
        {
            "x": numpy.asanyarray(["a", "b"]),
            "x_coded": numpy.asanyarray([1, 2]),
            "idx": range(2),
        }
    )
    d2 = pandas.DataFrame(
        {
            "x": numpy.asanyarray(["a", "b"]),
            "x_coded": numpy.asanyarray([1, 2]),
            "idx": range(2),
            "z": [1, 1],
        }
    )
    ops = (
        describe_table(d, table_name="d")
        .rename_columns({"x_coded_left": "x_coded", "idx_left": "idx"})
        .natural_join(b=describe_table(d, table_name="d"), jointype="full", by=["x"])
        .select_rows("x_coded_left < x_coded")
    )
    with pytest.raises(Exception):
        ops = (
            describe_table(d, table_name="d")
            .rename_columns({"x_coded_left": "x_coded", "idx_left": "idx"})
            .natural_join(
                b=describe_table(d2, table_name="d"), jointype="full", by=["x"]
            )
            .select_rows("x_coded_left < x_coded")
        )


def test_ref_detect_2():
    d = pandas.DataFrame(
        {
            "x": numpy.asanyarray(["a", "b"]),
            "x_coded": numpy.asanyarray([1, 2]),
            "idx": range(2),
        }
    )
    d2 = pandas.DataFrame(
        {
            "x": numpy.asanyarray(["a", "b"]),
            "x_coded": numpy.asanyarray([1, 2]),
            "idx": range(2),
            "z": [1, 1],
        }
    )
    ops = (
        describe_table(d, table_name="d")
        .rename_columns({"x_coded_left": "x_coded", "idx_left": "idx"})
        .natural_join(
            b=describe_table(d, table_name="d").rename_columns(
                {"x_coded_right": "x_coded", "idx_right": "idx"}
            ),
            jointype="full",
            by=["x"],
        )
        .select_rows("x_coded_left < x_coded_right")
    )
    with pytest.raises(Exception):
        ops = (
            describe_table(d, table_name="d")
            .rename_columns({"x_coded_left": "x_coded", "idx_left": "idx"})
            .natural_join(
                b=describe_table(d2, table_name="d").rename_columns(
                    {"x_coded_right": "x_coded", "idx_right": "idx"}
                ),
                jointype="full",
                by=["x"],
            )
            .select_rows("x_coded_left < x_coded_right")
        )
