import pytest

import data_algebra.parse_by_lark

def test_lark_1():
    # raises an exception if no lark parser
    tree = data_algebra.parse_by_lark._parse_by_lark('1 + 1')

