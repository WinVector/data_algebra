"""
Parse expresion to data ops in a medium environment. Uses eval() (don't run on untrusted expressions).
Used for confirming repr() is reversible.
"""


import data_algebra
import data_algebra.data_model
import numpy as np  # for globals() in eval_da_ops()
from data_algebra.data_ops import (
    describe_table,
    table,
    descr,
    data,
)  # for globals() in eval_da_ops()
from data_algebra.view_representations import (
    ViewRepresentation,
    TableDescription,
)  # for globals() in eval_da_ops()

pd = data_algebra.data_model.default_data_model().pd  # for globals() in eval_da_ops()
g_env = {k: v for k, v in globals().items()}

from typing import Any, Dict, Optional


def eval_da_ops(
    ops_str: str, *, data_model_map: Optional[Dict[str, Any]]
) -> ViewRepresentation:
    """
    Parse ops_str into a ViewRepresentation. Uses eval() (don't run on untrusted expressions).
    Used for confirming repr() is reversible.

    :param ops_str: text representation of a data algebra pipeline or expression.
    :param data_model_map: tables
    :return: data algebra ops
    """
    assert isinstance(ops_str, str)
    if data_model_map is None:
        local_data_model = data_algebra.data_model.default_data_model()
        data_model_map = {
            local_data_model.presentation_model_name: local_data_model.module
        }
    assert isinstance(data_model_map, Dict)
    ops = eval(
        ops_str,
        g_env,
        data_model_map,  # make our definition of data module available
        # cdata uses this
    )
    assert isinstance(ops, ViewRepresentation)
    return ops
