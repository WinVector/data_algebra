import numpy

import data_algebra
import data_algebra.util
import data_algebra.connected_components


# TODO: obsolete these folding all into the impl_map and a similar SQL structure

class CustomFunction:
    def __init__(self, *, name, implementation):
        self.name = name
        self.implementation = implementation


def make_custom_function_map(data_model):
    if data_model is None:
        raise ValueError("Expect data_model to not be None")

    custom_functions = [
        CustomFunction(
            name="is_bad",
            implementation=lambda x: data_model.bad_column_positions(x),
        ),
        CustomFunction(
            name="is_null",
            implementation=lambda x: data_model.isnull(x),
        ),
        CustomFunction(
            name="if_else",
            implementation=lambda c, x, y: numpy.where(c, x, y),
        ),
        CustomFunction(
            name="maximum",
            implementation=lambda x, y: numpy.maximum(x, y),
        ),
        CustomFunction(
            name="minimum",
            implementation=lambda x, y: numpy.minimum(x, y),
        ),
        CustomFunction(
            name="is_in",
            implementation=lambda c, v: numpy.isin(c, v),  # TODO: check this!!!
        ),
        CustomFunction(
            name="neg",
            implementation=lambda x: numpy.negative(x),  #
        ),
        CustomFunction(
            name="round",
            implementation=lambda x: numpy.round(x),
        ),
        CustomFunction(
            name="co_equalizer",
            implementation=lambda f, g: data_algebra.connected_components(
                f, g
            ),
        ),
        CustomFunction(
            name="connected_components",
            implementation=lambda f, g: data_algebra.connected_components.connected_components(
                f, g
            ),
        ),
        CustomFunction(
            name="max",
            implementation=lambda x: numpy.asarray([numpy.max(x)] * len(x)),
        ),
        CustomFunction(
            name="min",
            implementation=lambda x: numpy.asarray([numpy.min(x)] * len(x)),
        ),
        CustomFunction(
            name="concat",
            implementation=lambda x, y: numpy.char.add(numpy.asarray(x, dtype=numpy.str), numpy.asarray(y, dtype=numpy.str))
        ),
        CustomFunction(
            name="coalesce",
            implementation=lambda x, y: x.combine_first(y),  # assuming Pandas series
        ),
    ]

    mp = {cf.name: cf for cf in custom_functions}
    return mp
