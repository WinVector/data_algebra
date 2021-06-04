import numpy

import data_algebra
import data_algebra.util
import data_algebra.connected_components


class CustomFunction:
    def __init__(self, *, name, pandas_formatter, implementation):
        self.name = name
        self.pandas_formatter = pandas_formatter
        self.implementation = implementation

    def format_for_pandas(self, expr):
        return self.pandas_formatter(expr)


def make_custom_function_map(data_model):
    if data_model is None:
        raise ValueError("Expect data_model to not be None")

    custom_functions = [
        CustomFunction(
            name="is_bad",
            pandas_formatter=lambda expr: "@is_bad(" + expr.args[0].to_pandas() + ")",
            implementation=lambda x: data_model.bad_column_positions(x),
        ),
        CustomFunction(
            name="is_null",
            pandas_formatter=lambda expr: "@is_null(" + expr.args[0].to_pandas() + ")",
            implementation=lambda x: data_model.isnull(x),
        ),
        CustomFunction(
            name="if_else",
            pandas_formatter=lambda expr: (
                "@if_else("
                + expr.args[0].to_pandas()
                + ", "
                + expr.args[1].to_pandas()
                + ", "
                + expr.args[2].to_pandas()
                + ")"
            ),
            implementation=lambda c, x, y: numpy.where(c, x, y),
        ),
        CustomFunction(
            name="maximum",
            pandas_formatter=lambda expr: (
                    "@maximum("
                    + expr.args[0].to_pandas()
                    + ", "
                    + expr.args[1].to_pandas()
                    + ")"
            ),
            implementation=lambda x, y: numpy.maximum(x, y),
        ),
        CustomFunction(
            name="minimum",
            pandas_formatter=lambda expr: (
                    "@minimum("
                    + expr.args[0].to_pandas()
                    + ", "
                    + expr.args[1].to_pandas()
                    + ")"
            ),
            implementation=lambda x, y: numpy.minimum(x, y),
        ),
        CustomFunction(
            name="is_in",
            pandas_formatter=lambda expr: (
                    "@is_in("
                    + expr.args[0].to_pandas()
                    + ", "
                    + expr.args[1].to_pandas()
                    + ")"
            ),
            implementation=lambda c, v: numpy.isin(c, v),
        ),
        CustomFunction(
            name="neg",
            pandas_formatter=lambda expr: "-" + expr.args[0].to_pandas(want_inline_parens=True),
            implementation=lambda x: numpy.negative(x),  #
        ),
        CustomFunction(
            name="round",
            pandas_formatter=lambda expr: "@round(" + expr.args[0].to_pandas(want_inline_parens=False) + ")",
            implementation=lambda x: numpy.round(x),
        ),
        CustomFunction(
            name="co_equalizer",
            pandas_formatter=lambda expr: (
                "@co_equalizer("
                + expr.args[0].to_pandas()
                + ", "
                + expr.args[1].to_pandas()
                + ")"
            ),
            implementation=lambda f, g: data_algebra.connected_components.connected_components(
                f, g
            ),
        ),
        CustomFunction(
            name="connected_components",
            pandas_formatter=lambda expr: (
                "@connected_components("
                + expr.args[0].to_pandas()
                + ", "
                + expr.args[1].to_pandas()
                + ")"
            ),
            implementation=lambda f, g: data_algebra.connected_components.connected_components(
                f, g
            ),
        ),
        CustomFunction(
            name="partitioned_eval",
            pandas_formatter=lambda expr: (
                "@partitioned_eval("
                # expr.args[0] is a FnValue
                + "@"
                + expr.args[0].to_pandas()
                + ", "
                # expr.args[1] is a ListTerm
                + "["
                + ", ".join([ei.to_pandas() for ei in expr.args[1].value.value])
                + "]"
                + ", "
                # expr.args[2] is a ListTerm
                + "["
                + ", ".join([ei.to_pandas() for ei in expr.args[2].value.value])
                + "]"
                + ")"
            ),
            implementation=lambda fn, arg_columns, partition_columns: (
                data_algebra.connected_components.partitioned_eval(
                    fn, arg_columns, partition_columns
                )
            ),
        ),
        CustomFunction(
            name="max",
            pandas_formatter=lambda expr: ("@max(" + expr.args[0].to_pandas() + ")"),
            implementation=lambda x: numpy.asarray([numpy.max(x)] * len(x)),
        ),
        CustomFunction(
            name="min",
            pandas_formatter=lambda expr: ("@min(" + expr.args[0].to_pandas() + ")"),
            implementation=lambda x: numpy.asarray([numpy.min(x)] * len(x)),
        ),
        CustomFunction(
            name="concat",
            pandas_formatter=lambda expr: ("@concat(" + expr.args[0].to_pandas() + ", " + expr.args[1].to_pandas() + ")"),
            implementation=lambda x, y: numpy.char.add(numpy.asarray(x, dtype=numpy.str), numpy.asarray(y, dtype=numpy.str))
        ),
        CustomFunction(
            name="coalesce",
            pandas_formatter=lambda expr: ("@coalesce(" + expr.args[0].to_pandas() + ", " + expr.args[1].to_pandas() + ")"),
            implementation=lambda x, y: x.combine_first(y),  # assuming Pandas series
        ),
    ]

    mp = {cf.name: cf for cf in custom_functions}
    return mp
