

import abc
from typing import Any, Callable, Dict, Iterable, List, Optional


class ExpressionWalker(abc.ABC):
    """Abstract class will expression walking callbacks."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def act_on_literal(self, *, value):
        """
        Action for a literal/constant in an expression.

        :param value: literal value being supplied
        :return: converted result
        """

    @abc.abstractmethod
    def act_on_column_name(self, *, arg, value):
        """
        Action for a column name.

        :param arg: None
        :param value: column name
        :return: arg acted on
        """
    
    @abc.abstractmethod
    def act_on_expression(self, *, arg, values: List, op):
        """
        Action for a column name.

        :param arg: None
        :param values: list of values to work on
        :param op: data_algebra.expr_rep.Expression operator to apply
        :return: arg acted on
        """
