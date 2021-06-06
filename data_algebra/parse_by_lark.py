
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
        parser='lalr',
        #start='single_input',
        start='test',  # In the lark Python grammar test falls through to expression, making it a good entry point
        # propagate_positions=True,
    )
except:
    parser = None


# set up tree walker, including re-mapped names

op_remap = {
    '==': '__eq__',
    '!=': '__ne__',
    '<>': '__ne__',
    '<': '__lt__',
    '<=': '__le__',
    '>': '__gt__',
    '>=': '__ge__',
    '+': '__add__',
    '-': '__sub__',
    '*': '__mul__',
    '/': '__truediv__',
    '//': '__floordiv__',
    '%': '__mod__',
    '**': '__pow__',
    '&': '__and__',
    '&&': '__and__',
    '^': '__xor__',
    '|': '__or__',
    '||': '__or__',
    '%+%': 'concat',
    '%?%': 'coalesce',
}


factor_remap = {
    '-': '__neg__',  # unary!
    '+': '__pos__',  # unary!
    'not': 'not',  # unary! # TODO: implement
}

logical_remap = {
    'expr': '__or__',
    'and_expr': '__and__',
    'xor_expr': '__xor__',
}


def _walk_lark_tree(op, *, data_def=None, outer_environment=None):
    """
    Walk a lark parse tree and return our own reperesentation.

    :param op: lark parse tree
    :param data_def: dictionary of data_algebra.expr_rep.ColumnReference
    :param outer_environment: dictionary of system functions and values
    :return: PreTerm tree.
    """
    if data_def is None:
        data_def = {}
    if outer_environment is None:
        outer_environment = {}
    else:
        outer_environment = {
            k: v for (k, v) in outer_environment.items() if not k.startswith("_")
        }
    # don't have to completely kill this environment, as the code is something
    # the user intends to run (and may have even typed in).
    # But let's cut down the builtins anyway.
    outer_environment["__builtins__"] = {
        k: v for (k, v) in outer_environment.items() if isinstance(v, Exception)
    }

    def lookup_symbol(key):
        try:
            return data_algebra.expr_rep._enc_value(data_def[key])
        except KeyError:
            try:
                return data_algebra.expr_rep._enc_value(outer_environment[key])
            except KeyError:
                raise NameError("unknown symbol: " + key)

    def _r_walk_lark_tree(op):
        if isinstance(op, lark.lexer.Token):
            if op.type == 'DEC_NUMBER':
                return data_algebra.expr_rep.Value(int(op))
            if op.type == 'FLOAT_NUMBER':
                return data_algebra.expr_rep.Value(float(op))
            if op.type == 'STRING':
                return data_algebra.expr_rep.Value(ast.literal_eval(str(op)))   # strip excess quotes
            if op.type == 'NAME':
                return lookup_symbol(str(op))
            raise ValueError("unexpected Token type: " + op.type)
        if isinstance(op, lark.tree.Tree):
            if op.data == 'const_true':
                return data_algebra.expr_rep.Value(True)
            if op.data == 'const_false':
                return data_algebra.expr_rep.Value(False)
            if op.data == 'const_none':
                return data_algebra.expr_rep.Value(None)
            if op.data in ['single_input', 'number', 'string', 'var']:
                return _r_walk_lark_tree(op.children[0])
            if op.data in ['arith_expr', 'term', 'comparison']:
                # expect a v (op v)+ pattern
                nc = len(op.children)
                if (nc < 1) or ((nc % 2) != 1):
                    raise ValueError("unexpected " + op.data + " length")
                # just linear chain them
                res = _r_walk_lark_tree(op.children[0])
                for i in range((nc-1)//2):
                    op_name = str(op.children[2*i + 1])
                    try:
                        op_name = op_remap[op_name]
                    except KeyError:
                        pass
                    res = getattr(res, op_name)(_r_walk_lark_tree(op.children[2*i + 2]))
                return res
            if op.data == 'power':
                if len(op.children) != 2:
                    raise ValueError("unexpected " + op.data + " length")
                left = _r_walk_lark_tree(op.children[0])
                op_name = '__pow__'
                right = _r_walk_lark_tree(op.children[1])
                return getattr(left, op_name)(right)
            if op.data == 'factor':
                if len(op.children) != 2:
                    raise ValueError("unexpected arith_expr length")
                op_name = str(op.children[0])
                try:
                    op_name = factor_remap[op_name]
                except KeyError:
                    pass
                right = _r_walk_lark_tree(op.children[1])
                return getattr(right, op_name)()
            if op.data in logical_remap.keys():
                if len(op.children) < 2:
                    raise ValueError("unexpected ' + op.data + ' length")
                op_name = logical_remap[op.data]
                children = [_r_walk_lark_tree(ci) for ci in op.children]
                # just linear chain them
                res = children[0]
                for i in range(1, len(children)):
                    res = getattr(res, op_name)(children[i])
                return res
            if op.data == 'funccall':
                if len(op.children) > 2:
                    raise ValueError("unexpected funccall length")
                method_carrier = op.children[0]
                if isinstance(method_carrier, lark.tree.Tree) and (method_carrier.data == 'getattr'):
                    # method invoke
                    var = _r_walk_lark_tree(method_carrier.children[0])
                    op_name = str(method_carrier.children[1])
                else:
                    # function invoke
                    var = None
                    op_name = str(method_carrier.children[0])
                args = []
                if len(op.children) > 1:
                    raw_args = op.children[1].children
                    args = [_r_walk_lark_tree(ai) for ai in raw_args]
                if var is not None:
                    method = getattr(var, op_name)
                    return method(*args)
                else:
                    if op_name.startswith('_'):  # TODO: research why we are adding and removing underbar
                        op_name = op_name[1:len(op_name)]
                    return data_algebra.expr_rep.Expression(
                        op = op_name,
                        args = args
                    )
            if (op.data == 'or_test') or (op.data == 'and_test'):
                if len(op.children) != 2:
                    raise ValueError("unexpected " + op.data + " length")
                left = _r_walk_lark_tree(op.children[0])
                if op.data == 'or_test':
                    op_name = '__or__'
                else:
                    op_name = '__and__'
                right = _r_walk_lark_tree(op.children[1])
                return getattr(left, op_name)(right)
            if op.data == 'not':
                if len(op.children) != 1:
                    raise ValueError("unexpected " + op.data + " length")
                left = _r_walk_lark_tree(op.children[0])
                op_name = '__eq__'
                return getattr(left, op_name)(data_algebra.expr_rep.Value(False))
            if op.data in ['list', 'tuple']:
                vals = [_r_walk_lark_tree(vi) for vi in op.children[0].children]
                return data_algebra.expr_rep.ListTerm(vals)
            if op.data == 'expr_stmt':
                raise ValueError("Error must use == for comparison, not =")
            raise ValueError("unexpected/not-allowed lark Tree kind: " + str(op.data))
        raise ValueError("unexpected lark parse type: " + str(type(op)))

    return _r_walk_lark_tree(op)


def parse_by_lark(source_str, *, data_def=None, outer_environment=None):
    """
    Parse an expression in terms of data views and values.

    :param source_str: string to parse
    :param data_def: dictionary of data_algebra.expr_rep.ColumnReference
    :param outer_environment: dictionary of system functions and values
    :return:
    """
    assert parser is not None
    if not isinstance(source_str, str):
        source_str = str(source_str)
    tree = parser.parse(source_str)
    # convert parse tree to our data structures for isolation
    v = _walk_lark_tree(tree, data_def=data_def, outer_environment=outer_environment)
    v.source_string = source_str
    return v

