import ast

import lark
import lark.tree
import lark.lexer

import data_algebra.util

import data_algebra.expr_rep
import data_algebra.python3_lark

# TODO: documentation pages (esp logical symbol issues and assignment).


# set up parser
parser = None
# noinspection PyBroadException
try:
    parser = lark.Lark(
        data_algebra.python3_lark.grammar,
        parser="lalr",
        # start='single_input',
        start="test",  # In the lark Python grammar test falls through to expression, making it a good entry point
        # propagate_positions=True,
    )
except:
    parser = None


# set up tree walker, including re-mapped names

op_remap = {
    "==": "__eq__",
    "!=": "__ne__",
    "<>": "__ne__",
    "<": "__lt__",
    "<=": "__le__",
    ">": "__gt__",
    ">=": "__ge__",
    "+": "__add__",
    "-": "__sub__",
    "*": "__mul__",
    "/": "__truediv__",
    "//": "__floordiv__",
    "%": "__mod__",
    "**": "__pow__",
    "&": "__and__",
    "&&": "__and__",
    "^": "__xor__",
    "|": "__or__",
    "||": "__or__",
    "%+%": "concat",
    "%?%": "coalesce",
}


factor_remap = {
    "-": "__neg__",  # unary!
    "+": "__pos__",  # unary!
    "not": "not",  # unary! # TODO: implement
}


def _walk_lark_tree(op, *, data_def=None):
    """
    Walk a lark parse tree and return our own representation.

    :param op: lark parse tree
    :param data_def: dictionary of data_algebra.expr_rep.ColumnReference
    :return: PreTerm tree.
    """
    if data_def is None:
        data_def = dict()

    def lookup_symbol(key):
        try:
            return data_algebra.expr_rep.enc_value(data_def[key])
        except KeyError:
            raise NameError(f"unknown symbol: {key}")

    # noinspection SpellCheckingInspection
    def _r_walk_lark_tree(r_op):
        if isinstance(r_op, lark.lexer.Token):
            if r_op.type == "DEC_NUMBER":
                return data_algebra.expr_rep.Value(int(r_op))
            if r_op.type == "FLOAT_NUMBER":
                return data_algebra.expr_rep.Value(float(r_op))
            if r_op.type == "STRING":
                return data_algebra.expr_rep.Value(
                    ast.literal_eval(str(r_op))
                )  # strip excess quotes
            if r_op.type == "NAME":
                return lookup_symbol(str(r_op))
            raise ValueError("unexpected Token type: " + r_op.type)
        if isinstance(r_op, lark.tree.Tree):
            if r_op.data == "const_true":
                return data_algebra.expr_rep.Value(True)
            if r_op.data == "const_false":
                return data_algebra.expr_rep.Value(False)
            if r_op.data == "const_none":
                return data_algebra.expr_rep.Value(None)
            if r_op.data in ["single_input", "number", "string", "var"]:
                return _r_walk_lark_tree(r_op.children[0])
            if r_op.data in ["arith_expr", "term", "comparison"]:
                # expect a v (r_op v)+ pattern
                nc = len(r_op.children)
                # check we have 3 or more pieces (and an odd number of such)
                if (nc < 3) or ((nc % 2) != 1):
                    raise ValueError("unexpected " + r_op.data + " length")
                # check ops are all the same
                ops_seen = [str(r_op.children[i]) for i in range(nc) if (i % 2) == 1]
                if (len(set(ops_seen)) == 1) and (r_op.data in ["arith_expr", "term"]):
                    op_name = ops_seen[0]
                    if op_name in {"+", "*"}:
                        children = [
                            _r_walk_lark_tree(r_op.children[i])
                            for i in range(nc)
                            if (i % 2) == 0
                        ]
                        res = data_algebra.expr_rep.kop_expr(
                            op_name, children, inline=True, method=False
                        )
                        return res
                # just linear chain ops
                res = _r_walk_lark_tree(r_op.children[0])
                for i in range((nc - 1) // 2):
                    op_name = str(r_op.children[2 * i + 1])
                    try:
                        op_name = op_remap[op_name]
                    except KeyError:
                        pass
                    res = getattr(res, op_name)(
                        _r_walk_lark_tree(r_op.children[2 * i + 2])
                    )
                return res
            if r_op.data == "power":
                if len(r_op.children) != 2:
                    raise ValueError("unexpected " + r_op.data + " length")
                left = _r_walk_lark_tree(r_op.children[0])
                op_name = "__pow__"
                right = _r_walk_lark_tree(r_op.children[1])
                return getattr(left, op_name)(right)
            if r_op.data == "factor":
                if len(r_op.children) != 2:
                    raise ValueError("unexpected arith_expr length")
                op_name = str(r_op.children[0])
                try:
                    op_name = factor_remap[op_name]
                except KeyError:
                    pass
                right = _r_walk_lark_tree(r_op.children[1])
                return getattr(right, op_name)()
            if r_op.data == "funccall":
                if len(r_op.children) > 2:
                    raise ValueError("unexpected funccall length")
                var = None
                op_name = None
                method_carrier = r_op.children[0]
                if isinstance(method_carrier, lark.tree.Tree):
                    if method_carrier.data == "getattr":
                        # method invoke
                        var = _r_walk_lark_tree(method_carrier.children[0])
                        op_name = str(method_carrier.children[1])
                    else:
                        # function invoke
                        var = None
                        op_name = str(method_carrier.children[0])
                else:
                    if isinstance(method_carrier, str):
                        op_name = method_carrier
                if op_name is None:
                    raise ValueError("couldn't work out method name")
                args = []
                if len(r_op.children) > 1:
                    raw_args = r_op.children[1].children
                    args = [_r_walk_lark_tree(ai) for ai in raw_args]
                if var is not None:
                    method = getattr(var, op_name)
                    return method(*args)
                else:
                    if op_name.startswith("_"):  # TODO: get rid of underbar aliases
                        op_name = op_name[1 : len(op_name)]
                    return data_algebra.expr_rep.Expression(op=op_name, args=args)
            if r_op.data in {"or_test", "or_test_sym", "and_test", "and_test_sym"}:
                if len(r_op.children) < 2:
                    raise ValueError("unexpected " + r_op.data + " length")
                if r_op.data in {"or_test", "or_test_sym"}:
                    op_name = "or"
                elif r_op.data in {"and_test", "and_test_sym"}:
                    op_name = "and"
                else:
                    raise ValueError(f"unexpected test: {r_op.data}")
                children = [_r_walk_lark_tree(ci) for ci in r_op.children]
                res = data_algebra.expr_rep.kop_expr(
                    op_name, children, inline=True, method=False
                )
                return res
            if r_op.data == "not":
                if len(r_op.children) != 1:
                    raise ValueError("unexpected " + r_op.data + " length")
                left = _r_walk_lark_tree(r_op.children[0])
                op_name = "__eq__"
                return getattr(left, op_name)(data_algebra.expr_rep.Value(False))
            if r_op.data in ["list", "tuple", "set"]:
                op_values = [_r_walk_lark_tree(vi) for vi in r_op.children[0].children]
                # check all args are values, not None, same type
                assert all(
                    [isinstance(vi, data_algebra.expr_rep.Value) for vi in op_values]
                )
                assert all([vi.value is not None for vi in op_values])
                observed_types = {type(vi.value) for vi in op_values}
                observed_types = [
                    data_algebra.util.map_type_to_canonical(v) for v in observed_types
                ]
                assert len(observed_types) == 1
                return data_algebra.expr_rep.ListTerm(op_values)
            if r_op.data == "expr_stmt":
                raise ValueError("Error must use == for comparison, not =")
            raise ValueError("unexpected/not-allowed lark Tree kind: " + str(r_op.data))
        raise ValueError("unexpected lark parse type: " + str(type(r_op)))

    return _r_walk_lark_tree(op)


def parse_by_lark(source_str, *, data_def=None):
    """
    Parse an expression in terms of data views and values.

    :param source_str: string to parse
    :param data_def: dictionary of data_algebra.expr_rep.ColumnReference
    :return:
    """
    assert parser is not None
    if not isinstance(source_str, str):
        source_str = str(source_str)
    tree = parser.parse(source_str)
    # convert parse tree to our data structures for isolation
    v = _walk_lark_tree(tree, data_def=data_def)
    v.source_string = source_str
    return v
