"""
Base class for SQL adapters for data algebra.
"""

import math
import re
import warnings
from collections import OrderedDict
from typing import List, Optional
from typing import Dict, Set, Tuple

import data_algebra
import data_algebra.data_model
import data_algebra.near_sql
import data_algebra.expr_rep
import data_algebra.util
import data_algebra.data_ops_types
import data_algebra.data_ops
from data_algebra.OrderedSet import OrderedSet
import data_algebra.op_catalog
from data_algebra.sql_format_options import SQLFormatOptions

# The db_model can be a bit tricky as SQL is represented a few ways depending
# on how close to a final result we are.
# The end representation is a single string.
# The nearly there representation is a list of strings, which makes indenting much easier.
# The primary computational representation is a NearSQL structure, as it is a dag of objects.
# Note: near sql has a bound/unbound variation treating the top layer differently than
# subordinate nodes.


def _list_join_expecting_list(joiner: str, str_list: List[str]) -> List[str]:
    assert isinstance(joiner, str)
    assert isinstance(str_list, list)
    assert all([isinstance(vi, str) for vi in str_list])
    n = len(str_list)
    return [" " + str_list[i] + (joiner if i < (n - 1) else "") for i in range(n)]


def _clean_annotation(annotation: Optional[str]) -> Optional[str]:
    assert isinstance(annotation, (str, type(None)))
    if annotation is None:
        return annotation
    annotation = annotation.strip()
    annotation = re.sub(r"(\s|\r|\n)+", " ", annotation)
    annotation = annotation.replace(
        "%", "percent"
    )  # SQL alchemy doesn't like these in comments
    return annotation.strip()


# map from op-name to special SQL formatting code


def _db_lag_expr(dbmodel, expression):
    arg_0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    if len(expression.args) == 1:
        return f"LAG({arg_0})"
    elif len(expression.args) == 2:
        periods = expression.args[1].value
        if periods > 0:
            return f"LAG({arg_0}, {periods})"
        elif periods < 0:
            return f"LEAD({arg_0}, {-periods})"
        else:
            raise ValueError("shift by zero not supported")
    else:
        raise ValueError("too many arguments to SQL LAG/LEAD")


# Note, mod should not always equal remainder:
# https://rob.conery.io/2018/08/21/mod-and-remainder-are-not-the-same/
# But they do in numpy, which we will use as the
# reference implementation.
# import numpy as np
# np.mod([5, 5, -5, 5], [2, -2, 2, -2])
# # array([ 1, -1,  1, -1])
# np.remainder([5, 5, -5, 5], [2, -2, 2, -2])
# # array([ 1, -1,  1, -1])
# so we are just going to send it out and use destination semantics
def _db_mod_expr(dbmodel, expression):
    return (
        "MOD("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ")"
    )


# extend to floating point
# import numpy as np
# e0 = np.array([5, 5, -5, -5])
# e1 = np.array([2, -2, 2, -2])
# (e0 - np.floor(e0 / (1.0 * e1)) * e1)
# # array([ 1., -1.,  1., -1.])
def _db_remainder_expr(dbmodel, expression):
    e0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    e1 = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return f"({e0} - FLOOR({e0} / (1.0 * {e1})) * {e1})"


def _db_mean_expr(dbmodel, expression):
    return (
        "AVG(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


# noinspection PyUnusedLocal
def _db_size_expr(dbmodel, expression):
    return "SUM(1)"


def _db_is_null_expr(dbmodel, expression):
    return (
        "("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " IS NULL)"
    )


def _db_is_inf_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    plus_inf = f"CAST({dbmodel.value_to_sql('+infinity')} AS {dbmodel.float_type})"
    minus_inf = f"CAST({dbmodel.value_to_sql('-infinity')} AS {dbmodel.float_type})"
    return (
        "(CASE"
        + f" WHEN {subexpr} IS NULL THEN FALSE"
        + f" WHEN NOT (({subexpr} > {minus_inf}) AND ({subexpr} < {plus_inf})) THEN TRUE"
        + " ELSE FALSE"
        + " END)"
    )


def _db_is_nan_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return (
        "(CASE"
        + f" WHEN {subexpr} IS NULL THEN FALSE"
        + f" WHEN ({subexpr} > 0) AND (-{subexpr} > 0) THEN TRUE"
        + f" WHEN ({subexpr} != {subexpr}) THEN TRUE"
        + f" WHEN ({subexpr} != 0) AND ({subexpr} = -{subexpr}) THEN TRUE"
        + " ELSE FALSE"
        + " END)"
    )


def _db_is_bad_expr(dbmodel, expression):
    subexpr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    plus_inf = f"CAST({dbmodel.value_to_sql('+infinity')} AS {dbmodel.float_type})"
    minus_inf = f"CAST({dbmodel.value_to_sql('-infinity')} AS {dbmodel.float_type})"
    return (
        "(CASE"
        + f" WHEN {subexpr} IS NULL THEN TRUE"
        + f" WHEN ({subexpr} > 0) AND (-{subexpr} > 0) THEN TRUE"
        + f" WHEN ({subexpr} != {subexpr}) THEN TRUE"
        + f" WHEN ({subexpr} != 0) AND ({subexpr} = -{subexpr}) THEN TRUE"
        + f" WHEN NOT (({subexpr} > {minus_inf}) AND ({subexpr} < {plus_inf})) THEN TRUE"
        + " ELSE FALSE"
        + " END)"
    )


def _db_if_else_expr(dbmodel, expression):
    assert len(expression.args) == 3
    if_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    x_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[2], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + if_expr
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + if_expr
        + " THEN "
        + y_expr
        + " ELSE "
        + "NULL"
        + " END"
    )


def _db_where_expr(dbmodel, expression):
    assert len(expression.args) == 3
    if_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    x_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[2], want_inline_parens=True)
    return "CASE" + " WHEN " + if_expr + " THEN " + x_expr + " ELSE " + y_expr + " END"


def _db_mapv(dbmodel, expression):
    if_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    mapping_dict = expression.args[1]
    default_value_expr = dbmodel.expr_to_sql(
        expression.args[2], want_inline_parens=True
    )
    terms = [
        "WHEN " + dbmodel.value_to_sql(k) + " THEN " + dbmodel.value_to_sql(v)
        for k, v in mapping_dict.value.items()
    ]
    if len(terms) <= 0:
        return default_value_expr
    return (
        "CASE"
        + " "
        + if_expr
        + " "
        + " ".join(terms)
        + " ELSE "
        + default_value_expr
        + " END"
    )


def _db_maximum_expr(dbmodel, expression):
    x_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + "(("
        + y_expr
        + " IS NULL) OR "
        + "("
        + x_expr
        + ") >= ("
        + y_expr
        + "))"
        + " THEN "
        + x_expr
        + " WHEN "
        + "(("
        + x_expr
        + " IS NULL) OR "
        + "("
        + y_expr
        + ") >= ("
        + x_expr
        + "))"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_fmax_expr(dbmodel, expression):
    x_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + "("
        + x_expr
        + ") >= ("
        + y_expr
        + ")"
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + "("
        + x_expr
        + ") >= ("
        + y_expr
        + ")"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_minimum_expr(dbmodel, expression):
    x_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + "(("
        + y_expr
        + " IS NULL) OR "
        + "("
        + x_expr
        + ") <= ("
        + y_expr
        + "))"
        + " THEN "
        + x_expr
        + " WHEN "
        + "(("
        + x_expr
        + " IS NULL) OR "
        + "("
        + y_expr
        + ") <= ("
        + x_expr
        + "))"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_fmin_expr(dbmodel, expression):
    x_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    y_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return (
        "CASE"
        + " WHEN "
        + "("
        + x_expr
        + ") <= ("
        + y_expr
        + ")"
        + " THEN "
        + x_expr
        + " WHEN NOT "
        + "("
        + x_expr
        + ") <= ("
        + y_expr
        + ")"
        + " THEN "
        + y_expr
        + " ELSE NULL END"
    )


def _db_is_in_expr(dbmodel, expression):
    is_in_expr = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    x_expr = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return "(" + is_in_expr + " IN " + x_expr + ")"


def _db_count_expr(dbmodel, expression):
    """Count number of non-null entries (as in Pandas)"""
    if len(expression.args) != 1:
        return "SUM(1)"
    e0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return f"SUM(CASE WHEN {e0} IS NOT NULL THEN 1 ELSE 0 END)"


def _db_coalesce_expr(dbmodel, expression):
    return (
        "COALESCE("
        + ", ".join(
            [
                dbmodel.expr_to_sql(ai, want_inline_parens=False)
                for ai in expression.args
            ]
        )
        + ")"
    )


def _db_pow_expr(dbmodel, expression):
    if isinstance(expression.args[1], data_algebra.expr_rep.Value) and (
        expression.args[1].value == 1
    ):
        return dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    return (
        "POWER("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ")"
    )


def _db_round_expr(dbmodel, expression):
    return (
        "ROUND("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _db_around_expr(dbmodel, expression):
    if isinstance(expression.args[1], data_algebra.expr_rep.Value) and (
        expression.args[1].value == 0
    ):
        return dbmodel.expr_to_sql(expression.args[0].round(), want_inline_parens=False)
    mult = data_algebra.expr_rep.Value(10.0).__pow__(expression.args[1])
    derived = (expression.args[0] * mult).round() / mult
    return dbmodel.expr_to_sql(derived, want_inline_parens=True)


def _db_floor_expr(dbmodel, expression):
    return (
        "FLOOR("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _db_ceil_expr(dbmodel, expression):
    return (
        "CEILING("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _db_int_divide_expr(dbmodel, expression):
    # example of a derived expression
    ratio = (expression.args[0] / expression.args[1]).floor()
    return dbmodel.expr_to_sql(ratio, want_inline_parens=False)


def _db_float_divide_expr(dbmodel, expression):
    # don't issue an error
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/mathematical_functions#ieee_divide
    assert len(expression.args) == 2
    e0 = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=True)
    e1 = dbmodel.expr_to_sql(expression.args[1], want_inline_parens=True)
    return f"({e0} / (1.0 * {e1}))"


def _db_nunique_expr(dbmodel, expression):
    return (
        "COUNT(DISTINCT ("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + "))"
    )


def _as_int64(dbmodel, expression):
    return (
        "CAST("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " AS INT64)"
    )


def _as_str(dbmodel, expression):
    return (
        "CAST("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + " AS "
        + dbmodel.string_type
        + ")"
    )


def _db_concat_expr(dbmodel, expression):
    return (
        "("
        + " || ".join(
            [
                dbmodel.expr_to_sql(ai.as_str(), want_inline_parens=True)
                for ai in expression.args
            ]
        )
        + ")"
    )


def _trimstr(dbmodel, expression):
    return (
        "SUBSTR("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", 1 + "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[2], want_inline_parens=False)
        + ")"
    )


def _any_expr(dbmodel, expression):
    derived = expression.args[0].where(1, 0).max() >= data_algebra.expr_rep.Value(1)
    return dbmodel.expr_to_sql(derived, want_inline_parens=True)


def _all_expr(dbmodel, expression):
    derived = expression.args[0].where(1, 0).min() >= data_algebra.expr_rep.Value(1)
    return dbmodel.expr_to_sql(derived, want_inline_parens=True)


def _any_value_expr(dbmodel, expression):
    return (
        "MAX(" + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False) + ")"
    )


# date/time fns


def _datetime_to_date(dbmodel, expression):
    return (
        "DATE("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _parse_datetime(dbmodel, expression):
    return (
        "PARSE_DATETIME("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _parse_date(dbmodel, expression):
    return (
        "PARSE_DATE("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _format_datetime(dbmodel, expression):
    return (
        "FORMAT_DATETIME("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _format_date(dbmodel, expression):
    return (
        "FORMAT_DATE("
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _dayofweek(dbmodel, expression):
    return (
        "EXTRACT(DAYOFWEEK FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _dayofyear(dbmodel, expression):
    return (
        "EXTRACT(DAYOFYEAR FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _dayofmonth(dbmodel, expression):
    return (
        "EXTRACT(DAY FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _weekofyear(dbmodel, expression):
    return (
        "EXTRACT(WEEK FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _month(dbmodel, expression):
    return (
        "EXTRACT(MONTH FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _quarter(dbmodel, expression):
    return (
        "EXTRACT(QUARTER FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _year(dbmodel, expression):
    return (
        "EXTRACT(YEAR FROM "
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ")"
    )


def _timestamp_diff(dbmodel, expression):
    return (
        "TIMESTAMP_DIFF("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", SECOND)"
    )


def _date_diff(dbmodel, expression):
    return (
        "TIMESTAMP_DIFF("
        + dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
        + ", "
        + dbmodel.expr_to_sql(expression.args[1], want_inline_parens=False)
        + ", DAY)"
    )


def _base_Sunday(dbmodel, expression):
    subex = dbmodel.expr_to_sql(expression.args[0], want_inline_parens=False)
    return f"DATE_SUB({subex}, INTERVAL (EXTRACT(DAYOFWEEK FROM {subex}) - 1) DAY)"


db_expr_formatters = {
    "shift": _db_lag_expr,
    "is_inf": _db_is_inf_expr,
    "is_nan": _db_is_nan_expr,
    "is_null": _db_is_null_expr,
    "is_bad": _db_is_bad_expr,
    "mean": _db_mean_expr,
    "size": _db_size_expr,
    "if_else": _db_if_else_expr,
    "where": _db_where_expr,
    "is_in": _db_is_in_expr,
    "maximum": _db_maximum_expr,
    "fmax": _db_fmax_expr,
    "minimum": _db_minimum_expr,
    "fmin": _db_fmin_expr,
    "count": _db_count_expr,
    "concat": _db_concat_expr,
    "coalesce": _db_coalesce_expr,
    "round": _db_round_expr,
    "around": _db_around_expr,
    "floor": _db_floor_expr,
    "ceil": _db_ceil_expr,
    "//": _db_int_divide_expr,
    "%/%": _db_float_divide_expr,
    "**": _db_pow_expr,
    "nunique": _db_nunique_expr,
    "mapv": _db_mapv,
    "%": _db_mod_expr,
    "mod": _db_mod_expr,
    "remainder": _db_remainder_expr,
    # additional fns
    "as_int64": _as_int64,
    "as_str": _as_str,
    "trimstr": _trimstr,
    "any": _any_expr,
    "all": _all_expr,
    "any_value": _any_value_expr,
    "datetime_to_date": _datetime_to_date,
    "parse_datetime": _parse_datetime,
    "parse_date": _parse_date,
    "format_datetime": _format_datetime,
    "format_date": _format_date,
    "dayofweek": _dayofweek,
    "dayofyear": _dayofyear,
    "dayofmonth": _dayofmonth,
    "weekofyear": _weekofyear,
    "month": _month,
    "quarter": _quarter,
    "year": _year,
    "timestamp_diff": _timestamp_diff,
    "date_diff": _date_diff,
    "base_Sunday": _base_Sunday,
}


db_default_op_replacements = {
    "==": "=",
    "cumsum": "SUM",
    "cumcount": "COUNT",
    "cummax": "MAX",
    "cummin": "MIN",
    "cumprod": "PROD",
    "cummean": "AVG",
    "and": "AND",
    "or": "OR",
    "_count": "COUNT",
    "_ngroup": "NGROUP",
    "_row_number": "ROW_NUMBER",
    "_size": "SIZE",
    "_connected_components": "CONNECTED_COMPONENTS",
    "_uniform": "RAND",
}


def _annotated_method_catalogue(
    model_name: str,
) -> Tuple[
    Set[data_algebra.data_ops_types.MethodUse],
    Set[data_algebra.data_ops_types.MethodUse],
]:
    """
    Prepare method lookup tables

    :param: model name (used as column key)
    :return: (known_method_uses, recommended_method_uses)
    """
    known_method_uses: Set[data_algebra.data_ops_types.MethodUse] = set()
    recommended_method_uses: Set[data_algebra.data_ops_types.MethodUse] = set()
    for i in range(data_algebra.op_catalog.methods_table.shape[0]):
        row = data_algebra.op_catalog.methods_table.loc[i, :]
        matches = []
        if row["op_class"] in {"p", "up"}:
            matches.append(
                data_algebra.data_ops_types.MethodUse(
                    row["op"],
                    is_project=True,
                    is_windowed=False,
                    is_ordered=False,
                )
            )
        elif row["op_class"] == "g":
            matches.append(
                data_algebra.data_ops_types.MethodUse(
                    row["op"],
                    is_project=False,
                    is_windowed=True,
                    is_ordered=False,
                )
            )
        elif row["op_class"] == "w":
            matches.append(
                data_algebra.data_ops_types.MethodUse(
                    row["op"],
                    is_project=False,
                    is_windowed=True,
                    is_ordered=True,
                )
            )
        else:  # e and u case
            matches.append(
                data_algebra.data_ops_types.MethodUse(
                    row["op"],
                    is_project=False,
                    is_windowed=False,
                    is_ordered=False,
                )
            )
        if len(matches) > 0:
            good_use = (
                model_name in data_algebra.op_catalog.methods_table.columns
            ) and (data_algebra.op_catalog.methods_table[model_name].values[i] == "y")
            for m in matches:
                known_method_uses.add(m)
                if good_use:
                    recommended_method_uses.add(m)
    return known_method_uses, recommended_method_uses


class DBModel:
    """A model of how SQL should be generated for a given database."""

    identifier_quote: str
    string_quote: str
    on_start: str
    on_end: str
    on_joiner: str
    drop_text: str
    string_type: str
    supports_with: bool
    supports_cte_elim: bool
    allow_extend_merges: bool
    default_SQL_format_options: SQLFormatOptions
    union_all_term_start: str
    union_all_term_end: str
    known_methods: Optional[Set[data_algebra.data_ops_types.MethodUse]]
    recommended_methods: Optional[Set[data_algebra.data_ops_types.MethodUse]]

    def __init__(
        self,
        *,
        identifier_quote: str = '"',
        string_quote: str = "'",
        sql_formatters=None,
        op_replacements=None,
        on_start: str = "",
        on_end: str = "",
        on_joiner: str = "AND",
        drop_text: str = "DROP TABLE",
        string_type: str = "VARCHAR",
        float_type: str = "FLOAT64",
        supports_with: bool = True,
        supports_cte_elim: bool = True,
        allow_extend_merges: bool = True,
        default_SQL_format_options=None,
        union_all_term_start: str = "(",
        union_all_term_end: str = ")",
    ):
        self.local_data_model = data_algebra.data_model.default_data_model()
        if sql_formatters is None:
            sql_formatters = {}
        assert identifier_quote != string_quote
        self.identifier_quote = identifier_quote
        self.string_quote = string_quote
        self.sql_formatters = sql_formatters.copy()
        for k in db_expr_formatters.keys():
            if k not in self.sql_formatters.keys():
                self.sql_formatters[k] = db_expr_formatters[k]
        if op_replacements is None:
            op_replacements = db_default_op_replacements
        self.op_replacements = op_replacements.copy()
        self.on_start = on_start
        self.on_end = on_end
        self.on_joiner = on_joiner
        self.drop_text = drop_text
        self.string_type = string_type
        self.float_type = float_type
        if default_SQL_format_options is None:
            default_SQL_format_options = SQLFormatOptions()
        assert isinstance(default_SQL_format_options, SQLFormatOptions)
        self.default_SQL_format_options = default_SQL_format_options
        self.supports_with = supports_with
        self.supports_cte_elim = supports_cte_elim and supports_with
        self.allow_extend_merges = allow_extend_merges
        self.union_all_term_start = union_all_term_start
        self.union_all_term_end = union_all_term_end
        self.known_methods = None
        self.recommended_methods = None
        if str(self) in data_algebra.op_catalog.methods_table.columns:
            k_meth, r_meth = _annotated_method_catalogue(str(self))
            assert k_meth is not None  # type hint
            assert r_meth is not None  # type hint
            self.known_methods = k_meth
            self.recommended_methods = r_meth

    def db_handle(self, conn, *, db_engine=None):
        """

        :param conn: database connection
        :param db_engine: optional sqlalchemy style engine (for closing)
        """
        return DBHandle(db_model=self, conn=conn, db_engine=db_engine)

    def prepare_connection(self, conn):
        """
        Do any augmentation or preparation of a database connection. Example: adding stored procedures.
        """
        pass

    # database helpers

    # noinspection PyMethodMayBeStatic
    def execute(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        """
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        assert isinstance(q, str)
        data_algebra.data_model.default_data_model().pd.io.sql.execute(q, conn)

    def read_query(self, conn, q):
        """

        :param conn: database connection
        :param q: sql query
        :return: query results as table
        """
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self)
        else:
            q = str(q)
        assert isinstance(q, str)
        r = self.local_data_model.pd.io.sql.read_sql(q, conn)
        return r

    def table_exists(self, conn, table_name: str) -> bool:
        """
        Return true if table exists.
        """
        assert isinstance(table_name, str)
        q_table_name = self.quote_table_name(table_name)
        table_exists = True
        # noinspection PyBroadException
        try:
            self.read_query(conn, "SELECT * FROM " + q_table_name + " LIMIT 1")
        except Exception:
            table_exists = False
        return table_exists

    def drop_table(self, conn, table_name: str, *, check: bool = True) -> None:
        """
        Remove a table.
        """
        if (not check) or self.table_exists(conn, table_name):
            q_table_name = self.quote_table_name(table_name)
            self.execute(conn, self.drop_text + " " + q_table_name)

    # noinspection PyMethodMayBeStatic,SqlNoDataSourceInspection
    def insert_table(
        self, conn, d, table_name: str, *, qualifiers=None, allow_overwrite=False
    ) -> None:
        """
        Insert a table.

        :param conn: a database connection
        :param d: a Pandas table
        :param table_name: name to give write to
        :param qualifiers: schema and such
        :param allow_overwrite logical, if True drop previous table
        """

        if qualifiers is not None:
            raise ValueError("non-empty qualifiers not yet supported on insert")
        incoming_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
        d = incoming_data_model.to_pandas(d)
        if self.table_exists(conn, table_name):
            if not allow_overwrite:
                raise ValueError("table " + table_name + " already exists")
            else:
                self.drop_table(conn, table_name, check=False)
        # Note: the Pandas to_sql() method appears to have SQLite hard-wired into it
        # it refers to sqlite_master
        # this behavior seems to change if sqlalchemy is active
        # n.b. sqlalchemy tries to insert values on %
        d = data_algebra.data_model.lookup_data_model_for_dataframe(d).to_pandas(d)
        if d.shape[1] < 1:
            # deal with no columns case
            d = d.copy()
            d["_index"] = range(d.shape[0])
        d.to_sql(name=table_name, con=conn, index=False)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def read_table(self, conn, table_name: str, *, qualifiers=None, limit=None):
        """
        Return table contents as a Pandas data frame.
        """
        assert isinstance(table_name, str)
        q_table_name = self.quote_table_name(table_name)
        sql = "SELECT * FROM " + q_table_name
        if limit is not None:
            sql = sql + " LIMIT " + limit.__repr__()
        return self.read_query(conn, sql)

    def read(self, conn, table):
        """
        Return table as a pandas data frame for table description.
        """
        if table.node_name != "TableDescription":
            raise TypeError(
                "Expect table to be a data_algebra.data_ops.TableDescription"
            )
        return self.read_table(
            conn=conn, table_name=table.table_name, qualifiers=table.qualifiers
        )

    def quote_identifier(self, identifier: str) -> str:
        """
        Quote identifier.
        """
        assert isinstance(identifier, str)
        if self.identifier_quote in identifier:
            raise ValueError(
                "did not expect " + self.identifier_quote + " in identifier"
            )
        return self.identifier_quote + identifier + self.identifier_quote

    def quote_table_name(self, table_description) -> str:
        """
        Quote a table name.
        """
        if not isinstance(table_description, str):
            try:
                if table_description.node_name == "TableDescription":
                    table_description = table_description.table_name
                else:
                    raise TypeError(
                        "Expected table_description to be a string or data_algebra.data_ops.TableDescription)"
                    )
            except KeyError:
                raise TypeError(
                    "Expected table_description to be a string or data_algebra.data_ops.TableDescription)"
                )
        return self.quote_identifier(table_description)

    def quote_string(self, string: str) -> str:
        """
        Quote a string value.
        """
        assert isinstance(string, str)
        # replace all string quotes with doubled string quotes
        return (
            self.string_quote
            + re.sub(self.string_quote, self.string_quote + self.string_quote, string)
            + self.string_quote
        )

    def value_to_sql(self, v) -> str:
        """
        Convert a value to valid SQL.
        """
        if v is None:
            return "NULL"
        if isinstance(v, data_algebra.expr_rep.ListTerm):
            return "(" + ", ".join([self.value_to_sql(vi) for vi in v.value]) + ")"
        if isinstance(v, data_algebra.expr_rep.Value):
            return self.value_to_sql(v.value)
        if isinstance(v, str):
            return self.quote_string(v)
        if isinstance(v, bool):
            if v:
                return "TRUE"
            else:
                return "FALSE"
        if isinstance(v, float):
            if math.isnan(v):
                return "NULL"  # Pandas confuses NaN and None, so we use NULL
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, list) or isinstance(v, tuple):
            return "(" + ", ".join([self.value_to_sql(vi) for vi in v]) + ")"
        return str(v)

    def table_values_to_sql_str_list(
        self, v, *, result_name: str = "table_values"
    ) -> List[str]:
        """
        Convert a table of values to a SQL. Only for small tables.
        """
        assert v is not None
        m = v.shape[0]
        assert m > 0
        n = v.shape[1]
        assert n > 0
        qi = self.quote_identifier
        qv = self.value_to_sql

        def q_row(i):
            """quote row"""
            return (
                self.union_all_term_start
                + "SELECT "
                + ", ".join(
                    [
                        f"{qv(v[v.columns[j]][i])} AS {qi(v.columns[j])}"
                        for j in range(n)
                    ]
                )
                + self.union_all_term_end
            )

        sql = (
            ["SELECT", " *", "FROM ("]
            + ["    " + ("" if (i < 1) else "UNION ALL ") + q_row(i) for i in range(m)]
            + [f") {qi(result_name)}"]
        )
        return sql

    def expr_to_sql(self, expression, *, want_inline_parens: bool = False) -> str:
        """
        Convert an expression to SQL.
        """
        if isinstance(expression, str):
            return expression
        assert isinstance(expression, data_algebra.expr_rep.PreTerm)
        if isinstance(expression, data_algebra.expr_rep.Value):
            return self.value_to_sql(expression.value)
        if isinstance(expression, data_algebra.expr_rep.ColumnReference):
            return self.quote_identifier(expression.column_name)
        if isinstance(expression, data_algebra.expr_rep.Expression):
            op = expression.op
            if op in self.op_replacements.keys():
                op = self.op_replacements[op]
            elif op.lower() in self.op_replacements.keys():
                op = self.op_replacements[op.lower()]
            elif op.upper() in self.op_replacements.keys():
                op = self.op_replacements[op.upper()]
            if op in self.sql_formatters.keys():
                return self.sql_formatters[op](self, expression)
            if op.lower() in self.sql_formatters.keys():
                return self.sql_formatters[op.lower()](self, expression)
            if op.upper() in self.sql_formatters.keys():
                return self.sql_formatters[op.upper()](self, expression)
            if (len(expression.args) > 1) and expression.inline:
                subs = [
                    self.expr_to_sql(ai, want_inline_parens=True)
                    for ai in expression.args
                ]
                res = ""
                if want_inline_parens:
                    res = res + "("
                assert len(subs) > 0
                if len(subs) == 1:
                    res = op.upper() + subs[0]
                else:
                    res = res + (" " + op.upper() + " ").join(subs)
                if want_inline_parens:
                    res = res + ")"
                return res
            subs = [
                self.expr_to_sql(ai, want_inline_parens=False) for ai in expression.args
            ]
            return op.upper() + "(" + ", ".join(subs) + ")"
        if isinstance(expression, data_algebra.expr_rep.ListTerm):
            return self.value_to_sql(expression.value)
        raise TypeError("unexpected type: " + str(type(expression)))

    def _indent_and_sep_terms(
        self, terms, *, sep: str = ",", sql_format_options=None
    ) -> List[str]:
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        n = len(terms)
        assert n >= 1
        comma_spacer = " " * len(sep)
        if sql_format_options.initial_commas:
            return [
                sql_format_options.sql_indent
                + (comma_spacer if i == 0 else sep)
                + " "
                + terms[i]
                for i in range(n)
            ]
        return [
            sql_format_options.sql_indent
            + terms[i]
            + ((" " + sep) if i < (n - 1) else "")
            for i in range(n)
        ]

    def table_def_to_near_sql(
        self, table_def, *, using=None, temp_id_source=None, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert a table description to NearSQL.
        """
        if table_def.node_name != "TableDescription":
            raise TypeError(
                "Expected table_def to be a data_algebra.data_ops.TableDescription)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = set(table_def.column_names)
        missing = using - OrderedSet(table_def.column_names)
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        cols_using = [c for c in table_def.column_names if c in using]
        subsql = data_algebra.near_sql.NearSQLTable(
            terms={k: self.quote_identifier(k) for k in cols_using},
            table_name=self.quote_table_name(table_def),
            quoted_table_name=self.quote_table_name(table_def),
        )
        near_sql = subsql
        if (len(using) > 0) and (
            not (set([k for k in using]) == set([k for k in table_def.column_names]))
        ):
            # need a non-trivial select here
            terms = OrderedDict()
            for k in using:
                terms[k] = k  # these get quoted later
            view_name = "table_reference_" + str(temp_id_source[0])
            temp_id_source[0] = temp_id_source[0] + 1
            return data_algebra.near_sql.NearSQLUnaryStep(
                terms=terms,
                query_name=view_name,
                quoted_query_name=self.quote_identifier(view_name),
                sub_sql=subsql.to_bound_near_sql(columns=using),
                ops_key=f"table({table_def.table_name}, {terms.keys()})",
            )
        return near_sql

    def extend_to_near_sql(
        self, extend_node, *, using=None, temp_id_source=None, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert an extend step into NearSQL.
        """
        if extend_node.node_name != "ExtendNode":
            raise TypeError(
                "Expected extend_node to be a data_algebra.data_ops.ExtendNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(extend_node.column_names)
        using = using.union(
            extend_node.partition_by, extend_node.order_by, extend_node.reverse
        )
        subops = OrderedDict()
        for (k, op) in extend_node.ops.items():
            if k in using:
                subops[k] = op
        if len(subops) <= 0:
            # using was not None is this case as len(extend_node.ops)>0 and all keys are in extend_node.column_names
            return extend_node.sources[0].to_near_sql_implementation_(
                db_model=self, using=using, temp_id_source=temp_id_source
            )
        if len(using) < 1:
            raise ValueError("must produce at least one column")
        missing = using - set(extend_node.column_names)
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        # get set of columns we need from subquery
        subusing = extend_node.columns_used_from_sources(using=using)[0]
        subsql = extend_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        window_term = ""
        window_vars = set()
        if (
            extend_node.windowed_situation
            or (len(extend_node.partition_by) > 0)
            or (len(extend_node.order_by) > 0)
        ):
            window_term = " OVER ( "
            if len(extend_node.partition_by) > 0:
                pt = [self.quote_identifier(ci) for ci in extend_node.partition_by]
                window_term = window_term + "PARTITION BY " + ", ".join(pt) + " "
                window_vars.update(extend_node.partition_by)
            if len(extend_node.order_by) > 0:
                revs = set(extend_node.reverse)
                rt = [
                    self.quote_identifier(ci) + (" DESC" if ci in revs else "")
                    for ci in extend_node.order_by
                ]
                window_term = window_term + "ORDER BY " + ", ".join(rt) + " "
                window_vars.update(extend_node.order_by)
            window_term = window_term + " ) "
        terms: Dict[str, Optional[str]] = OrderedDict()
        declared_term_dependencies = OrderedDict()
        origcols = [k for k in using if k not in subops.keys()]
        for k in origcols:
            terms[k] = None
            declared_term_dependencies[k] = {k}
        for (ci, oi) in subops.items():
            terms[ci] = self.expr_to_sql(oi) + window_term
            cols_used_in_term: Set[str] = set()
            oi.get_column_names(cols_used_in_term)
            cols_used_in_term.update(window_vars)
            declared_term_dependencies[ci] = cols_used_in_term
        annotation = str(extend_node.to_python_src_(print_sources=False, indent=-1))
        # TODO: see if we can merge with subsql instead of building a new one
        if (
            self.allow_extend_merges
            and isinstance(subsql, data_algebra.near_sql.NearSQLUnaryStep)
            and subsql.mergeable
            and (subsql.declared_term_dependencies is not None)
            and ((subsql.suffix is None) or (len(subsql.suffix) == 0))
        ):
            # check detailed merge conditions
            def non_trivial_terms(*, dep_dict, term_dict):
                """pickout non-trivial terms"""
                return [
                    ki
                    for ki, vi in dep_dict.items()
                    if (len(vi - {ki}) > 0)
                    or (ki not in vi)
                    or ((term_dict[ki] is not None) and (term_dict[ki] != ki))
                ]

            our_non_trivial_terms = non_trivial_terms(
                dep_dict=declared_term_dependencies, term_dict=terms
            )
            our_needs = set().union(
                *[declared_term_dependencies[k] for k in our_non_trivial_terms]
            )
            sub_non_trivial_terms = non_trivial_terms(
                dep_dict=subsql.declared_term_dependencies, term_dict=subsql.terms
            )
            sub_needs = set().union(
                *[subsql.declared_term_dependencies[k] for k in sub_non_trivial_terms]
            )
            contention = set().union(
                set(our_non_trivial_terms).intersection(sub_non_trivial_terms),
                set(our_non_trivial_terms).intersection(sub_needs),
                set(sub_non_trivial_terms).intersection(our_needs),
            )
            if len(contention) == 0:
                # merge our stuff into subsql
                assert subsql.terms is not None  # type hint
                if subsql.annotation is None:
                    subsql.annotation = annotation
                else:
                    subsql.annotation = subsql.annotation + "." + annotation
                for k in our_non_trivial_terms:
                    subsql.terms[k] = terms[k]
                    subsql.declared_term_dependencies[k] = declared_term_dependencies[k]
                we_use = set.union(
                    set(terms.keys()), set(declared_term_dependencies.keys())
                )
                excess_sub_term_keys = set(subsql.terms.keys()) - we_use
                for k in excess_sub_term_keys:
                    del subsql.terms[k]
                excess_sub_declared_keys = (
                    set(subsql.declared_term_dependencies.keys()) - we_use
                )
                for k in excess_sub_declared_keys:
                    del subsql.declared_term_dependencies[k]
                return subsql
        view_name = "extend_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            annotation=annotation,
            mergeable=True,
            declared_term_dependencies=declared_term_dependencies,
            ops_key=f"extend({extend_node}, {terms.keys()})",
        )
        return near_sql

    def project_to_near_sql(
        self, project_node, *, using=None, temp_id_source=None, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert a project step to NearSQL
        """
        if project_node.node_name != "ProjectNode":
            raise TypeError(
                "Expected project_node to be a data_algebra.data_ops.ProjectNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(project_node.column_names)
        subops = {k: op for (k, op) in project_node.ops.items() if k in using}
        subusing = project_node.columns_used_from_sources(using=using)[0]
        terms = {ci: self.expr_to_sql(oi) for (ci, oi) in subops.items()}
        terms.update({g: None for g in project_node.group_by})
        subsql = project_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "project_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        suffix = []
        if len(project_node.group_by) > 0:
            group_terms = [self.quote_identifier(c) for c in project_node.group_by]
            suffix = ["GROUP BY"] + self._indent_and_sep_terms(
                group_terms, sql_format_options=sql_format_options
            )
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            annotation=str(project_node.to_python_src_(print_sources=False, indent=-1)),
            ops_key=f"project({project_node}, {terms.keys()})",
        )
        return near_sql

    def select_rows_to_near_sql(
        self,
        select_rows_node,
        *,
        using=None,
        temp_id_source=None,
        sql_format_options=None,
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert select rows into NearSQL
        """
        if select_rows_node.node_name != "SelectRowsNode":
            raise TypeError(
                "Expected select_rows_node to be a data_algebra.data_ops.SelectRowsNode)"
            )
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(select_rows_node.column_names)
        subusing = select_rows_node.columns_used_from_sources(using=using)[0]
        subsql = select_rows_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "select_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = {ci: None for ci in using}
        suffix = ["WHERE"] + [
            sql_format_options.sql_indent + self.expr_to_sql(select_rows_node.expr)
        ]
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            annotation=str(
                select_rows_node.to_python_src_(print_sources=False, indent=-1)
            ),
            ops_key=f"select({select_rows_node}, {terms.keys()})",
        )
        return near_sql

    def select_columns_to_near_sql(
        self,
        select_columns_node,
        *,
        using=None,
        temp_id_source=None,
        sql_format_options=None,
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert select columns to NearSQL.
        """
        if select_columns_node.node_name != "SelectColumnsNode":
            raise TypeError(
                "Expected select_columns_to_near_sql to be a data_algebra.data_ops.SelectColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(select_columns_node.column_names)
        subusing = select_columns_node.columns_used_from_sources(using=using)[0]
        subusing = [
            c for c in select_columns_node.column_selection if c in subusing
        ]  # fix order
        subsql = select_columns_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=set(subusing), temp_id_source=temp_id_source
        )
        # order/limit columns
        if subsql.terms is not None:
            subsql.terms = {
                k: subsql.terms[k]
                for k in select_columns_node.column_selection
                if k in subusing
            }
        else:
            subsql.terms = []
        return subsql

    def drop_columns_to_near_sql(
        self,
        drop_columns_node,
        *,
        using=None,
        temp_id_source=None,
        sql_format_options=None,
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert drop columns to NearSQL
        """
        if drop_columns_node.node_name != "DropColumnsNode":
            raise TypeError(
                "Expected drop_columns_node to be a data_algebra.data_ops.DropColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(drop_columns_node.column_names)
        subusing = drop_columns_node.columns_used_from_sources(using=using)[0]
        subsql = drop_columns_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        # /limit columns
        subsql.terms = {
            k: subsql.terms[k]
            for k in using
            if k not in drop_columns_node.column_deletions
        }
        return subsql

    def order_to_near_sql(
        self, order_node, *, using=None, temp_id_source=None, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert order rows to NearSQL.
        """
        if order_node.node_name != "OrderRowsNode":
            raise TypeError(
                "Expected order_node to be a data_algebra.data_ops.OrderRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        using_was_None = False
        if using is None:
            using = OrderedSet(order_node.column_names)
            using_was_None = True
        subusing = order_node.columns_used_from_sources(using=using)[0]
        subusing = [c for c in order_node.column_names if c in subusing]  # fix order
        subsql = order_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=set(subusing), temp_id_source=temp_id_source
        )
        view_name = "order_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        terms = None
        if not using_was_None:
            terms = {ci: None for ci in subusing}
        suffix: List[str] = []
        if len(order_node.order_columns) > 0:
            suffix = (
                suffix
                + ["ORDER BY"]
                + self._indent_and_sep_terms(
                    [
                        self.quote_identifier(ci)
                        + (" DESC" if ci in set(order_node.reverse) else "")
                        for ci in order_node.order_columns
                    ],
                    sql_format_options=sql_format_options,
                )
            )
        if order_node.limit is not None:
            suffix = suffix + ["LIMIT " + order_node.limit.__repr__()]
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            suffix=suffix,
            annotation=str(order_node.to_python_src_(print_sources=False, indent=-1)),
            ops_key=f"order({order_node})",  # no terms
        )
        return near_sql

    def map_columns_to_near_sql(
        self,
        map_columns_node,
        *,
        using=None,
        temp_id_source=None,
        sql_format_options=None,
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert map columns columns to NearSQL.
        """
        if map_columns_node.node_name != "MapColumnsNode":
            raise TypeError(
                "Expected map_columns_node to be a data_algebra.data_ops.MapColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(map_columns_node.column_names)
        subusing = map_columns_node.columns_used_from_sources(using=using)[0]
        subsql = map_columns_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "map_columns_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        unchanged_columns = subusing - (
            set(map_columns_node.column_remapping.values()).union(map_columns_node.column_remapping.keys()).union(map_columns_node.column_deletions))
        terms = {
            ki: self.quote_identifier(vi)
            for (vi, ki) in map_columns_node.column_remapping.items()
            if (vi is not None) and (ki is not None)
        }
        terms.update({vi: None for vi in unchanged_columns})
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            annotation=str(
                map_columns_node.to_python_src_(print_sources=False, indent=-1)
            ),
            ops_key=f"map_columns({map_columns_node}, {terms.keys()})",
        )
        return near_sql

    def rename_to_near_sql(
        self, rename_node, *, using=None, temp_id_source=None, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert rename columns to NearSQL.
        """
        if rename_node.node_name != "RenameColumnsNode":
            raise TypeError(
                "Expected rename_node to be a data_algebra.data_ops.RenameColumnsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(rename_node.column_names)
        subusing = rename_node.columns_used_from_sources(using=using)[0]
        subsql = rename_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=subusing, temp_id_source=temp_id_source
        )
        view_name = "rename_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        unchanged_columns = subusing - (
            set(rename_node.column_remapping.values()).union(rename_node.column_remapping.keys()))
        terms = {
            ki: self.quote_identifier(vi)
            for (ki, vi) in rename_node.column_remapping.items()
        }
        terms.update({vi: None for vi in unchanged_columns})
        near_sql = data_algebra.near_sql.NearSQLUnaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql=subsql.to_bound_near_sql(columns=subusing),
            annotation=str(rename_node.to_python_src_(print_sources=False, indent=-1)),
            ops_key=f"rename({rename_node}, {terms.keys()})",
        )
        return near_sql

    def _coalesce_terms(
        self, *, sub_view_name_first, sub_view_name_second, cols
    ) -> OrderedDict:
        coalesce_formatter = self.sql_formatters["coalesce"]

        class PseudoExpression:
            """
            Class to carry info into a method expecting an actual expression type.
            """

            def __init__(self, args):
                self.args = args.copy()

        terms = OrderedDict()
        for ci in cols:
            terms[ci] = coalesce_formatter(
                self,
                PseudoExpression(
                    [
                        sub_view_name_first + "." + self.quote_identifier(ci),
                        sub_view_name_second + "." + self.quote_identifier(ci),
                    ]
                ),
            )
        return terms

    def _natural_join_sub_queries(
        self, *, join_node, using, temp_id_source: List
    ) -> Tuple[
        OrderedSet,
        data_algebra.near_sql.NearSQL,
        OrderedSet,
        data_algebra.near_sql.NearSQL,
    ]:
        if join_node.node_name != "NaturalJoinNode":
            raise TypeError(
                "Expected join_node to be a data_algebra.data_ops.NaturalJoinNode)"
            )
        if using is None:
            using = OrderedSet(join_node.column_names)
        if len(using) < 1:
            raise ValueError("join must use or select at least one column")
        missing = using - set(join_node.column_names)
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        using_left, using_right = join_node.columns_used_from_sources(
            using=using.union(join_node.on_a).union(join_node.on_b)
        )
        sql_left = join_node.sources[0].to_near_sql_implementation_(
            db_model=self, using=using_left, temp_id_source=temp_id_source
        )
        sql_right = join_node.sources[1].to_near_sql_implementation_(
            db_model=self, using=using_right, temp_id_source=temp_id_source
        )
        return using_left, sql_left, using_right, sql_right

    def natural_join_to_near_sql(
        self,
        join_node,
        *,
        using=None,
        temp_id_source=None,
        sql_format_options=None,
        left_is_first=True,
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert natural join into NearSQL.
        """
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(join_node.column_names)
        view_name = f"natural_join_{temp_id_source[0]}"
        left_q = f"join_source_left_{temp_id_source[0]}"
        right_q = f"join_source_right_{temp_id_source[0]}"
        temp_id_source[0] = temp_id_source[0] + 1
        using_left, sql_left, using_right, sql_right = self._natural_join_sub_queries(
            join_node=join_node, using=using, temp_id_source=temp_id_source
        )
        left_qqn = self.quote_identifier(left_q)
        right_qqn = self.quote_identifier(right_q)
        common = using_left.intersection(using_right)
        if left_is_first:
            terms = self._coalesce_terms(
                sub_view_name_first=left_qqn,
                sub_view_name_second=right_qqn,
                cols=[ci for ci in common if ci in using],
            )
        else:
            terms = self._coalesce_terms(
                sub_view_name_first=right_qqn,
                sub_view_name_second=left_qqn,
                cols=[ci for ci in common if ci in using],
            )
        for ci in using_left:
            if ci not in common:
                terms[ci] = None
        for ci in using_right:
            if ci not in common:
                terms[ci] = None
        on_terms = []
        if len(join_node.on_a) > 0:
            on_terms = ["ON " + self.on_start] + self._indent_and_sep_terms(
                [
                    left_qqn
                    + "."
                    + self.quote_identifier(c_a)
                    + " = "
                    + right_qqn
                    + "."
                    + self.quote_identifier(c_b)
                    for c_a, c_b in zip(join_node.on_a, join_node.on_b)
                ],
                sep=self.on_joiner,
                sql_format_options=sql_format_options,
            )
            if (self.on_end is not None) and (len(self.on_end) > 0):
                on_terms = on_terms + [self.on_end]
        near_sql = data_algebra.near_sql.NearSQLBinaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_bound_near_sql(
                columns=using_left,
                force_sql=False,
                public_name=left_q,
                public_name_quoted=left_qqn,
            ),
            joiner=join_node.jointype + " JOIN",
            sub_sql2=sql_right.to_bound_near_sql(
                columns=using_right,
                force_sql=False,
                public_name=right_q,
                public_name_quoted=right_qqn,
            ),
            suffix=on_terms,
            annotation=str(join_node.to_python_src_(print_sources=False, indent=-1)),
            ops_key=f"join({join_node}, {terms.keys()})",
        )
        return near_sql

    def concat_rows_to_near_sql(
        self, concat_node, *, using=None, temp_id_source=None, sql_format_options=None
    ) -> data_algebra.near_sql.NearSQL:
        """
        Convert concat rows into NearSQL.
        """
        if concat_node.node_name != "ConcatRowsNode":
            raise TypeError(
                "Expected concat_node to be a data_algebra.data_ops.ConcatRowsNode)"
            )
        if temp_id_source is None:
            temp_id_source = [0]
        if using is None:
            using = OrderedSet(concat_node.column_names)
        if len(using) < 1:
            raise ValueError("must select at least one column")
        missing = using - set(concat_node.column_names)
        if len(missing) > 0:
            raise KeyError("referred to unknown columns: " + str(missing))
        subusing = concat_node.columns_used_from_sources(using=using)
        using_left = subusing[0]
        using_right = subusing[1]
        if set(using_left) != set(using_right):
            raise ValueError("left/right usings did not match")
        using_joint = using_left.copy()
        terms = {ci: None for ci in using_joint}
        expr_left = concat_node.sources[0]
        expr_right = concat_node.sources[1]
        if concat_node.id_column is not None:
            expr_left = expr_left.extend(
                {concat_node.id_column: f'"{concat_node.a_name}"'}
            )
            expr_right = expr_right.extend(
                {concat_node.id_column: f'"{concat_node.b_name}"'}
            )
            using_joint.add(concat_node.id_column)
            terms.update({concat_node.id_column: None})
        sql_left = expr_left.to_near_sql_implementation_(
            db_model=self, using=using_joint, temp_id_source=temp_id_source
        )
        sql_right = expr_right.to_near_sql_implementation_(
            db_model=self, using=using_joint, temp_id_source=temp_id_source
        )
        view_name = "concat_rows_" + str(temp_id_source[0])
        temp_id_source[0] = temp_id_source[0] + 1
        near_sql = data_algebra.near_sql.NearSQLBinaryStep(
            terms=terms,
            query_name=view_name,
            quoted_query_name=self.quote_identifier(view_name),
            sub_sql1=sql_left.to_bound_near_sql(
                columns=using_joint.copy(), force_sql=True
            ),
            joiner="UNION ALL",
            sub_sql2=sql_right.to_bound_near_sql(
                columns=using_joint.copy(), force_sql=True
            ),
            annotation=str(concat_node.to_python_src_(print_sources=False, indent=-1)),
            ops_key=f"concat({concat_node}, {terms.keys()})",
        )
        return near_sql

    def non_known_methods(
        self, ops: data_algebra.data_ops.ViewRepresentation
    ) -> List[data_algebra.data_ops_types.MethodUse]:
        """Return list of used non-recommended methods."""
        if self.known_methods is None:
            return []  # can't check, just pass
        method_uses = ops.methods_used()
        non_recommended = [op for op in method_uses if op not in self.known_methods]
        non_recommended.sort()
        return non_recommended

    def non_recommended_methods(
        self, ops: data_algebra.data_ops.ViewRepresentation
    ) -> List[data_algebra.data_ops_types.MethodUse]:
        """Return list of used non-recommended methods."""
        if (self.recommended_methods is None) or (self.known_methods is None):
            return []  # can't check, just pass
        method_uses = ops.methods_used()
        non_recommended = [
            op
            for op in method_uses
            if (op not in self.recommended_methods) and (op in self.known_methods)
        ]
        return non_recommended

    def to_sql(
        self,
        ops: data_algebra.data_ops.ViewRepresentation,
        *,
        sql_format_options: Optional[SQLFormatOptions] = None,
    ) -> str:
        """
        Convert ViewRepresentation into SQL string.

        :param ops: ViewRepresentation to convert
        :param sql_format_options: sql formatting options
        :return: sql string
        """
        assert isinstance(self, DBModel)
        assert isinstance(ops, data_algebra.data_ops.ViewRepresentation)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        ops.columns_used()  # for table consistency check/raise
        if sql_format_options.warn_on_novel_methods:
            non_known = self.non_known_methods(ops)
            if len(non_known) > 0:
                warnings.warn(
                    f"{self} translation using undocumented method context: {non_known}",
                    UserWarning,
                )
        if sql_format_options.warn_on_method_support:
            non_rec = self.non_recommended_methods(ops)
            if len(non_rec) > 0:
                warnings.warn(
                    f"{self} translation doesn't fully support method context: {non_rec}",
                    UserWarning,
                )
        # TODO: put common sub-expression control object here and pass into converters
        temp_id_source = [0]
        near_sql = ops.to_near_sql_implementation_(
            db_model=self, using=None, temp_id_source=temp_id_source
        )
        assert isinstance(near_sql, data_algebra.near_sql.NearSQL)
        sql_str_list = None
        if sql_format_options.use_with and self.supports_with:
            cte_cache: Optional[Dict] = None
            if (
                sql_format_options.use_cte_elim
                and self.supports_with
                and self.supports_cte_elim
            ):
                cte_cache = dict()
            sequence = near_sql.to_with_form(cte_cache=cte_cache)
            len_sequence = len(sequence.previous_steps)
            # can fall back to the non-with path
            if len(sequence.previous_steps) >= 1:
                sql_sequence: List[str] = []
                for i in range(len_sequence):
                    nmi = sequence.previous_steps[i][0]  # already quoted
                    sqli = sequence.previous_steps[i][1].convert_subsql(
                        db_model=self, sql_format_options=sql_format_options
                    )
                    sql_sequence = (
                        sql_sequence
                        + [f"{sql_format_options.sql_indent}{nmi} AS ("]
                        + [
                            sql_format_options.sql_indent
                            + sql_format_options.sql_indent
                            + s
                            for s in sqli
                        ]
                    )
                    if i < (len_sequence - 1):
                        sql_sequence = sql_sequence + [
                            sql_format_options.sql_indent + ") ,"
                        ]
                    else:
                        sql_sequence = sql_sequence + [
                            sql_format_options.sql_indent + ")"
                        ]
                sql_last = sequence.last_step.to_sql_str_list(
                    db_model=self, force_sql=True, sql_format_options=sql_format_options
                )
                sql_str_list = ["WITH"] + sql_sequence + sql_last
        if sql_str_list is None:
            # non-with path
            sql_str_list = near_sql.to_sql_str_list(
                db_model=self, force_sql=True, sql_format_options=sql_format_options
            )
        if sql_format_options.annotate:
            model_descr = re.sub(r"\s+", " ", str(self))
            sql_str_list = [
                f"-- data_algebra SQL https://github.com/WinVector/data_algebra",
                f"--  dialect: {model_descr} {data_algebra.__version__}",
                f"--       string quote: {self.string_quote}",
                f"--   identifier quote: {self.identifier_quote}",
            ] + sql_str_list
        sql_str_list = [v.rstrip() for v in sql_str_list]
        sql_str_list = [v for v in sql_str_list if len(v) > 0]
        return "\n".join(sql_str_list) + "\n"

    def row_recs_to_blocks_query_str_list_pair(
        self, record_spec
    ) -> Tuple[List[str], List[str]]:
        """
        Convert row recs to blocks transformation into structures to help with SQL conversion.
        """
        control_value_cols = [
            c
            for c in record_spec.control_table.columns
            if c not in record_spec.control_table_keys
        ]
        control_cols = [
            "a." + self.quote_identifier(c) for c in record_spec.record_keys
        ] + [
            "b." + self.quote_identifier(key_col)
            for key_col in record_spec.control_table_keys
        ]
        col_stmts = []
        for c in record_spec.record_keys:
            col_stmts.append(
                " a." + self.quote_identifier(c) + " AS " + self.quote_identifier(c)
            )
        for key_col in record_spec.control_table_keys:
            col_stmts.append(
                " b."
                + self.quote_identifier(key_col)
                + " AS "
                + self.quote_identifier(key_col)
            )
        seen = set()
        for result_col in control_value_cols:
            if result_col in seen:
                continue
            seen.add(result_col)
            cstmt = " CASE "
            col = record_spec.control_table[result_col]
            isnull = col.isnull()
            for i in range(len(col)):
                if not (isnull[i]):
                    source_col = col[i]
                    col_sql = (
                        "  WHEN CAST(b."
                        + self.quote_identifier(result_col)
                        + " AS "
                        + self.string_type
                        + ") = "
                        + self.quote_string(str(source_col))
                        + " THEN a."
                        + self.quote_identifier(source_col)
                        + " "
                    )
                    cstmt = cstmt + col_sql
            cstmt = cstmt + " ELSE NULL END AS " + self.quote_identifier(result_col)
            col_stmts.append(cstmt)
        ctab_sql = self.table_values_to_sql_str_list(record_spec.control_table)
        sql_prefix = _list_join_expecting_list(",", col_stmts) + [
            "FROM ( SELECT * FROM "
        ]
        sql_suffix = (
            [" ) a"]
            + ["CROSS JOIN ("]
            + _list_join_expecting_list("", ctab_sql)
            + [" ) b"]
            + [" ORDER BY"]  # order by not required, but nice to have
            + _list_join_expecting_list(", ", control_cols)
        )
        return sql_prefix, sql_suffix

    # noinspection PyUnusedLocal
    def blocks_to_row_recs_query_str_list_pair(
        self, record_spec
    ) -> Tuple[List[str], List[str]]:
        """
        Convert blocks to row recs transform into structures to help with SQL translation.
        """
        assert record_spec.control_table.shape[0] >= 1
        col_stmts = []
        for c in record_spec.record_keys:
            col_stmts.append(
                " " + self.quote_identifier(c) + " AS " + self.quote_identifier(c)
            )
        if record_spec.control_table.shape[0] == 1:
            # special case with one row control table (rowrec)
            for cc in record_spec.control_table.columns:
                if cc not in record_spec.control_table_keys:
                    cc0 = record_spec.control_table[cc][0]
                    col_stmts.append(
                        " "
                        + self.quote_identifier(cc)
                        + " AS "
                        + self.quote_identifier(cc0)
                    )
            sql_prefix = _list_join_expecting_list(",", col_stmts) + [
                "FROM ( SELECT * FROM "
            ]
            sql_suffix = [" ) a"]
            return sql_prefix, sql_suffix
        control_value_cols = [
            c
            for c in record_spec.control_table.columns
            if c not in record_spec.control_table_keys
        ]
        control_cols = [self.quote_identifier(c) for c in record_spec.record_keys]
        seen = set()
        for i in range(record_spec.control_table.shape[0]):
            for vc in control_value_cols:
                col = record_spec.control_table[vc]
                isnull = col.isnull()
                if col[i] not in seen and not isnull[i]:
                    seen.add(col[i])
                    clauses = []
                    for cc in record_spec.control_table_keys:
                        clauses.append(
                            " ( CAST("
                            + self.quote_identifier(cc)
                            + " AS "
                            + self.string_type
                            + ") = "
                            + self.quote_string(str(record_spec.control_table[cc][i]))
                            + " ) "
                        )
                    cstmt = (
                        " MAX(CASE WHEN "
                        + " AND ".join(clauses)
                        + " THEN "
                        + self.quote_identifier(vc)
                        + " ELSE NULL END) AS "
                        + self.quote_identifier(col[i])
                    )
                    col_stmts.append(cstmt)
        sql_prefix = _list_join_expecting_list(",", col_stmts) + [
            "FROM ( SELECT * FROM "
        ]
        sql_suffix = (
            [" ) a"]
            + ["GROUP BY"]
            + _list_join_expecting_list(",", control_cols)
            + ["ORDER BY "]  # order by not required, but nice to have
            + _list_join_expecting_list(",", control_cols)
        )
        return sql_prefix, sql_suffix

    # encode and name a term for use in a SQL expression
    def enc_term_(self, k, *, terms) -> str:
        """
        encode and name a term for use in a SQL expression
        """
        v = None
        try:
            v = terms[k]
        except KeyError:
            pass
        if (v is None) or (v == k):
            return self.quote_identifier(k)
        return v + " AS " + self.quote_identifier(k)

    def nearsqlcte_to_sql_str_list_(
        self,
        near_sql,
        *,
        columns=None,
        force_sql=False,
        sql_format_options=None,
    ) -> List[str]:
        """
        Convert SQL common table expression to list of SQL string lines.
        """
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLCommonTableExpression)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if force_sql:
            return [
                "SELECT",
                sql_format_options.sql_indent + "*",
                "FROM ",
                sql_format_options.sql_indent + near_sql.quoted_query_name,
            ]
        return [near_sql.quoted_query_name]

    def nearsqltable_to_sql_str_list_(
        self,
        near_sql,
        *,
        columns=None,
        force_sql=False,
        sql_format_options=None,
    ) -> List[str]:
        """
        Convert SQL table description to list of SQL string lines.
        """
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLTable)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if columns is None:
            if near_sql.terms is not None:
                columns = [k for k in near_sql.terms.keys()]
            else:
                columns = []
        if len(columns) <= 0:
            columns = []
        if force_sql:
            terms_strs = [self.quote_identifier(k) for k in columns]
            if len(terms_strs) < 1:
                terms_strs = ["*"]
            return (
                ["SELECT"]
                + self._indent_and_sep_terms(
                    terms_strs, sql_format_options=sql_format_options
                )
                + ["FROM"]
                + [sql_format_options.sql_indent + near_sql.quoted_table_name]
            )
        return [near_sql.quoted_table_name]

    def nearsqlunary_to_sql_str_list_(
        self,
        near_sql,
        *,
        columns=None,
        force_sql=False,
        sql_format_options=None,
    ) -> List[str]:
        """
        Convert SQL unary operation to list of SQL string lines.
        """
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLUnaryStep)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        terms_strs = ["*"]  # allow * notation if nothing is specified
        terms = near_sql.terms
        if terms is not None:
            if columns is None:
                columns = [k for k in terms.keys()]
            terms_strs = [self.enc_term_(k, terms=terms) for k in columns]
            if len(terms_strs) < 1:
                terms_strs = ["*"]
        sql_start = "SELECT"
        if (
            sql_format_options.annotate
            and (near_sql.annotation is not None)
            and (len(near_sql.annotation) > 0)
        ):
            clean_anno = _clean_annotation(near_sql.annotation)
            if clean_anno is not None:
                sql_start = "SELECT  -- " + clean_anno
        sql = (
            [sql_start]
            + self._indent_and_sep_terms(
                terms_strs, sql_format_options=sql_format_options
            )
            + ["FROM"]
            + [
                sql_format_options.sql_indent + si
                for si in near_sql.sub_sql.convert_subsql(
                    db_model=self,
                    sql_format_options=sql_format_options,
                    quoted_query_name_annotation=near_sql.sub_sql.near_sql.quoted_query_name,
                )
            ]
        )
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + near_sql.suffix
        return sql

    def nearsqlrawq_to_sql_str_list_(
        self,
        near_sql,
        *,
        sql_format_options=None,
        add_select=True,
    ) -> List[str]:
        """
        Convert user SQL query to list of SQL string lines.
        """
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLRawQStep)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        sql: List[str] = []
        if near_sql.annotation is not None:
            clean_anno = _clean_annotation(near_sql.annotation)
            if clean_anno is not None:
                sql = sql + ["-- " + clean_anno]
        if add_select:
            sql = sql + ["SELECT"]
        sql = sql + [" " + v for v in near_sql.prefix]
        if near_sql.sub_sql is not None:
            sql = sql + [
                sql_format_options.sql_indent + si
                for si in near_sql.sub_sql.convert_subsql(
                    db_model=self,
                    sql_format_options=sql_format_options,
                    quoted_query_name_annotation=near_sql.sub_sql.near_sql.quoted_query_name,
                )
            ]
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + [" " + v for v in near_sql.suffix]
        return sql

    def nearsqlbinary_to_sql_str_list_(
        self,
        near_sql,
        *,
        columns=None,
        force_sql=False,
        sql_format_options=None,
        quoted_query_name=None,
    ) -> List[str]:
        """
        Convert SQL binary operation to list of SQL string lines.
        """
        assert isinstance(near_sql, data_algebra.near_sql.NearSQLBinaryStep)
        if sql_format_options is None:
            sql_format_options = self.default_SQL_format_options
        assert isinstance(sql_format_options, SQLFormatOptions)
        if near_sql.terms is None:
            terms: Dict[str, Optional[str]] = OrderedDict()
        else:
            terms = near_sql.terms
        if columns is None:
            columns = [k for k in terms.keys()]
        terms_strs = [self.enc_term_(k, terms=terms) for k in columns]
        if len(terms_strs) < 1:
            terms_strs = ["*"]
        is_union = "union" in near_sql.joiner.lower()
        sql_start = "SELECT"
        if (
            sql_format_options.annotate
            and (near_sql.annotation is not None)
            and (len(near_sql.annotation) > 0)
        ):
            clean_anno = _clean_annotation(near_sql.annotation)
            if clean_anno is not None:
                sql_start = "SELECT  -- " + clean_anno
        subsql_add_query_name = not is_union
        substr_1 = near_sql.sub_sql1.convert_subsql(
            db_model=self,
            sql_format_options=sql_format_options,
            quoted_query_name_annotation=near_sql.sub_sql1.public_name_quoted
            if subsql_add_query_name
            else None,
        )
        substr_2 = near_sql.sub_sql2.convert_subsql(
            db_model=self,
            sql_format_options=sql_format_options,
            quoted_query_name_annotation=near_sql.sub_sql2.public_name_quoted
            if subsql_add_query_name
            else None,
        )
        sql = (
            [sql_start]
            + self._indent_and_sep_terms(
                terms_strs, sql_format_options=sql_format_options
            )
            + ["FROM"]
            + ["("]
            + [sql_format_options.sql_indent + si for si in substr_1]
            + [near_sql.joiner]
            + [sql_format_options.sql_indent + si for si in substr_2]
        )
        if (near_sql.suffix is not None) and (len(near_sql.suffix) > 0):
            sql = sql + near_sql.suffix
        if (
            is_union
            and (quoted_query_name is not None)
            and (len(quoted_query_name) > 0)
        ):
            sql = sql + [
                ") " + quoted_query_name
            ]  # duplicate name, but not of whole query
        else:
            sql = sql + [")"]
        return sql

    def __str__(self):
        return str(type(self).__name__)

    def __repr__(self):
        return self.__str__()


class DBHandle:
    """
    Container for database connection handles.
    """

    def __init__(self, *, db_model: DBModel, conn, db_engine=None):
        """

        :param db_model: associated database model
        :param conn: database connection
        :param db_engine: optional sqlalchemy style engine (for closing)
        """
        assert isinstance(db_model, DBModel)
        self.db_model = db_model
        self.db_engine = db_engine
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def read_query(self, q):
        """
        Return results of query as a Pandas data frame.
        """
        return self.db_model.read_query(conn=self.conn, q=q)
    
    def read_table(self, table_name:str):
        """
        Return table as a Pandas data frame.

        :param table_name: table to read
        """
        tn = self.db_model.quote_table_name(table_name)
        return self.read_query(f"SELECT * FROM {tn}")
    
    def create_table(self, *, table_name:str, q):
        """
        Create table from query.

        :param table_name: table to create
        :param q: query
        :return: table description
        """
        tn = self.db_model.quote_table_name(table_name)
        if isinstance(q, data_algebra.data_ops.ViewRepresentation):
            q = q.to_sql(db_model=self.db_model)
        else:
            q = str(q)
        self.execute(f"CREATE TABLE {tn} AS {q}")
        return self.describe_table(table_name)

    def describe_table(
        self, table_name: str, *, qualifiers=None, row_limit: Optional[int] = 7
    ):
        """
        Return a description of a database table.
        """
        head = self.read_query(
            q="SELECT * FROM "
            + self.db_model.quote_table_name(table_name)
            + " LIMIT "
            + str(row_limit),
        )
        return data_algebra.data_ops.describe_table(
            head, table_name=table_name, qualifiers=qualifiers, row_limit=row_limit
        )

    def execute(self, q) -> None:
        """
        Execute a SQL query or operator dag.
        """
        self.db_model.execute(conn=self.conn, q=q)

    def drop_table(self, table_name: str) -> None:
        """
        Remove a table.
        """
        self.db_model.drop_table(self.conn, table_name)

    def insert_table(self, d, *, table_name: str, allow_overwrite: bool = False) -> data_algebra.data_ops.TableDescription:
        """
        Insert a table into the database.
        """
        incoming_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
        d = incoming_data_model.to_pandas(d)
        self.db_model.insert_table(
            conn=self.conn, d=d, table_name=table_name, allow_overwrite=allow_overwrite
        )
        res = self.describe_table(table_name)
        return res
    
    def to_sql(
        self,
        ops: data_algebra.data_ops.ViewRepresentation,
        *,
        sql_format_options: Optional[SQLFormatOptions] = None,
    ) -> str:
        """
        Convert operations into SQL
        """
        return self.db_model.to_sql(ops=ops, sql_format_options=sql_format_options)

    def query_to_csv(self, q, *, res_name: str) -> None:
        """
        Execute a query and save the results as a CSV file.
        """
        d = self.read_query(q)
        d.to_csv(res_name, index=False)

    def table_values_to_sql_str_list(
        self, v, *, result_name: str = "table_values"
    ) -> List[str]:
        """
        Convert a table of values to a SQL. Only for small tables.
        """
        return self.db_model.table_values_to_sql_str_list(v, result_name=result_name)

    def __str__(self):
        return (
            str(type(self).__name__)
            + "("
            + "db_model="
            + str(self.db_model)
            + ", conn="
            + str(self.conn)
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def close(self) -> None:
        """
        Dispose of engine, or close connection.
        """
        if self.conn is not None:
            caught = None
            if self.db_engine is not None:
                # sqlalchemy style handle
                # noinspection PyBroadException
                try:
                    self.db_engine.dispose()
                except Exception as ex:
                    caught = ex
            else:
                # noinspection PyBroadException
                try:
                    self.conn.close()
                except Exception as ex:
                    caught = ex
            self.db_engine = None
            self.conn = None
            if caught is not None:
                raise ValueError("close caught: " + str(caught))
