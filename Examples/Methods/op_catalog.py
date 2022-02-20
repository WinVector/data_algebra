

import data_algebra

pd = data_algebra.default_data_model.pd

    
methods_table = pd.DataFrame(
    {
        "op": [
            "!=",
            "%",
            "%/%",
            "*",
            "**",
            "+",
            "-",
            "-",
            "/",
            "//",
            "<",
            "<=",
            "==",
            "==",
            ">",
            ">=",
            "abs",
            "and",
            "arccos",
            "arccosh",
            "arcsin",
            "arcsinh",
            "arctan",
            "arctan2",
            "arctanh",
            "around",
            "as_int64",
            "as_str",
            "base_Sunday",
            "ceil",
            "ceil",
            "coalesce",
            "coalesce",
            "coalesce",
            "concat",
            "concat",
            "cos",
            "cosh",
            "date_diff",
            "datetime_to_date",
            "dayofmonth",
            "dayofweek",
            "dayofyear",
            "exp",
            "expm1",
            "floor",
            "floor",
            "fmax",
            "fmin",
            "format_date",
            "format_datetime",
            "if_else",
            "is_bad",
            "is_in",
            "is_inf",
            "is_nan",
            "is_null",
            "log",
            "log10",
            "log1p",
            "mapv",
            "maximum",
            "minimum",
            "mod",
            "month",
            "or",
            "parse_date",
            "parse_datetime",
            "quarter",
            "remainder",
            "round",
            "sign",
            "sin",
            "sinh",
            "sqrt",
            "sum",
            "tanh",
            "timestamp_diff",
            "trimstr",
            "weekofyear",
            "where",
            "year",
            "_count",
            "_ngroup",
            "_size",
            "count",
            "max",
            "mean",
            "median",
            "min",
            "nunique",
            "size",
            "std",
            "sum",
            "sum",
            "var",
            "_size",
            "all",
            "any",
            "count",
            "max",
            "mean",
            "median",
            "min",
            "nunique",
            "size",
            "std",
            "sum",
            "sum",
            "var",
            "_uniform",
            "any_value",
            "_row_number",
            "bfill",
            "cumcount",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "ffill",
            "first",
            "last",
            "rank",
            "shift",
        ],
        "expression": [
            "x != y",
            "row_id % q",
            "x %/% y",
            "x * y",
            "x ** y",
            "x + y",
            "-x",
            "x - y",
            "x / y",
            "row_id // q",
            "x < y",
            "x <= y",
            "not a",
            "x == y",
            "x > y",
            "x >= y",
            "z.abs()",
            "a and b",
            "x.arccos()",
            "x.arccosh()",
            "x.arcsin()",
            "x.arcsinh()",
            "x.arctan()",
            "x.arctan2(y)",
            "x.arctanh()",
            "y.around(2)",
            "y.as_int64()",
            "y.as_str()",
            "date_col_1.base_Sunday()",
            "y.ceil()",
            "z.ceil()",
            "z %?% 2",
            "z.coalesce(2)",
            "z.coalesce_0()",
            'g %+% "_" %+% s2',
            "g.concat(s2)",
            "x.cos()",
            "x.cosh()",
            "date_col_0.date_diff(date_col_1)",
            "datetime_col_0.datetime_to_date()",
            "date_col_0.dayofmonth()",
            "date_col_0.dayofweek()",
            "date_col_0.dayofyear()",
            "x.exp()",
            "y.expm1()",
            "y.floor()",
            "z.floor()",
            "row_id.fmax(x)",
            "row_id.fmin(x)",
            "date_col_0.format_date()",
            "datetime_col_0.format_datetime()",
            "a.if_else(x, y)",
            "z.is_bad()",
            "row_id.is_in({1, 3})",
            "y.is_inf()",
            "y.is_nan()",
            "z.is_null()",
            "x.log()",
            "x.log10()",
            "x.log1p()",
            'g.mapv({"a": 1, "b": 2, "z": 26}, 0)',
            "row_id.maximum(x)",
            "row_id.minimum(x)",
            "row_id.mod(2)",
            "date_col_0.month()",
            "a or b",
            "str_date_col.parse_date()",
            "str_datetime_col.parse_datetime()",
            "date_col_0.quarter()",
            "row_id.remainder(2)",
            "y.round()",
            "z.sign()",
            "x.sin()",
            "x.sinh()",
            "x.sqrt()",
            "x.sum()",
            "x.tanh()",
            "datetime_col_0.timestamp_diff(datetime_col_1)",
            "g.trimstr(0, 2)",
            "date_col_0.weekofyear()",
            "a.where(x, y)",
            "date_col_0.year()",
            "_count()",
            "_ngroup()",
            "_size()",
            "z.count()",
            "x.max()",
            "x.mean()",
            "x.median()",
            "x.min()",
            "x.nunique()",
            "x.size()",
            "x.std()",
            "(1).sum()",
            "x.sum()",
            "x.var()",
            "_size()",
            "a.all()",
            "a.any()",
            "z.count()",
            "x.max()",
            "x.mean()",
            "x.median()",
            "x.min()",
            "x.nunique()",
            "x.size()",
            "x.std()",
            "(1).sum()",
            "x.sum()",
            "x.var()",
            "_uniform()",
            "x.any_value()",
            "_row_number()",
            "z.bfill()",
            "z.cumcount()",
            "x.cummax()",
            "x.cummin()",
            "x.cumprod()",
            "x.cumsum()",
            "z.ffill()",
            "x.first()",
            "x.last()",
            "x.rank()",
            "x.shift()",
        ],
        "op_class": [
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "e",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "g",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "p",
            "u",
            "up",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
            "w",
        ],
        "Pandas": [
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
        ],
        "SQLiteModel": [
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "w",
            "n",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "n",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "n",
            "y",
            "n",
            "w",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "n",
            "y",
            "n",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "w",
            "y",
            "y",
            "n",
            "y",
            "n",
            "n",
            "n",
            "n",
            "y",
        ],
        "BigQueryModel": [
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "w",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "w",
            "y",
            "y",
            "n",
            "y",
            "n",
            "n",
            "n",
            "n",
            "y",
        ],
        "PostgreSQLModel": [
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "n",
            "n",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "w",
            "y",
            "y",
            "w",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "w",
            "y",
            "y",
            "n",
            "y",
            "n",
            "n",
            "n",
            "n",
            "y",
        ],
        "SparkSQLModel": [
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "y",
            "n",
            "y",
            "n",
            "y",
            "w",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "w",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "w",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "w",
            "y",
            "y",
            "w",
            "n",
            "y",
            "w",
            "y",
            "y",
            "n",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "w",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "w",
            "y",
            "y",
            "n",
            "y",
            "n",
            "y",
            "w",
            "y",
            "y",
        ],
        "MySQLModel": [
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "n",
            "y",
            "n",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "n",
            "n",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "n",
            "n",
            "y",
            "y",
            "y",
            "y",
            "w",
            "n",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "y",
            "n",
            "w",
            "y",
            "y",
            "n",
            "y",
            "n",
            "n",
            "n",
            "n",
            "y",
        ],
        "version": [
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
            "1.4.1",
        ],
    }
)

