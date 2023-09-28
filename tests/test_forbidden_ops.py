
import pytest

import data_algebra
from data_algebra.view_representations import TableDescription, ViewRepresentation
from data_algebra.data_ops import data, descr, describe_table, ex

import lark.exceptions


def test_forbidden_ops_raises():
    with pytest.raises(lark.exceptions.UnexpectedToken):
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
                {"z": "x && y"}
            )

    with pytest.raises(lark.exceptions.UnexpectedToken):
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
                {"z": "x || y"}
            )

    with pytest.raises(lark.exceptions.UnexpectedCharacters):  # not in grammar
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
            {"z": "! y"}
        )

    with pytest.raises(AttributeError):  # objects don't implement ~
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
            {"z": "~ y"}
        )

    with pytest.raises(lark.exceptions.UnexpectedToken):
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
            {"z": "x = y"}
        )


def test_forbidden_ops_inlines_left_alone():
    assert 'x ** y' in str(TableDescription(table_name="d", column_names=["x", "y"]).extend(
            {"z": "x ** y"}
        ))


def test_forbidden_ops_inline():
    with pytest.raises(ValueError):
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
                    {"z": "x & y"}
                )

    with pytest.raises(ValueError):
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
                    {"z": "x | y"}
                )

    with pytest.raises(ValueError):
        TableDescription(table_name="d", column_names=["x", "y"]).extend(
                    {"z": "x ^ y"}
                )
