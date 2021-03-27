
import pkgutil

import lark
import lark.tree
import lark.lexer

import data_algebra.util
import data_algebra.env
import data_algebra.custom_functions

import data_algebra.expr_rep
import data_algebra.python3_lark

# TODO: switch to lark parsing and use


# set up parser
parser = None
# noinspection PyBroadException
try:
    kwargs = {
        'start': 'single_input',
        # 'propagate_positions': True,
        }
    parser = lark.Lark(
        data_algebra.python3_lark.grammar,
        parser='lalr',
        **kwargs)
except:
    parser = None


# set up tree walker, including re-mapped names

op_remap = {
    '==': '__eq__',
    '!=': '__ne__',
    '<': '__lt__',
    '<=': '__le__',
    '>': '__gt__',
    '>=': '__ge__',
    '+': '__add__',
    '-': '__sub__',
    '*': '__mul__',
    '/': '__truediv__',
    '//': '__floordiv__',
    '**': '__pow__',
    '&': '__and__',
    '^': '__xor__',
    '|': '__or__',
}


factor_remap = {
    '-': '__neg__',  # unary!
    '+': '__pos__',  # unary!
    '~': '__not__',  # unary! # TODO: implement
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

    def _r_walk_lark_tree(op):
        if isinstance(op, lark.lexer.Token):
            if op.data == 'number':
                return data_algebra.expr_rep.Value(op.children[0])
            raise ValueError("unexpected lark Token kind: " + str(op.data))
        if isinstance(op, lark.tree.Tree):
            if op.data == 'single_input':
                return _r_walk_lark_tree(op.children[0])
            if op.data == 'number':
                tok = op.children[0]
                return data_algebra.expr_rep.Value(float(tok))
            if op.data == 'string':
                tok = op.children[0]
                return data_algebra.expr_rep.Value(str(tok))
            if op.data == 'var':
                key = str(op.children[0])
                try:
                    return data_def[key]
                except KeyError:
                    try:
                        return outer_environment[key]
                    except KeyError:
                        raise ValueError("unknown symbol: " + key)
            if op.data in ['arith_expr', 'term']:
                if len(op.children) != 3:
                    raise ValueError("unexpected " + op.data + " length")
                left = _r_walk_lark_tree(op.children[0])
                op_name = str(op.children[1])
                try:
                    op_name = op_remap[op_name]
                except KeyError:
                    pass
                right = _r_walk_lark_tree(op.children[2])
                return getattr(left, op_name)(right)
            if op.data == 'comparison':
                if len(op.children) != 3:
                    raise ValueError("unexpected comparison length")
                left = _r_walk_lark_tree(op.children[0])
                op_name = str(op.children[1])
                try:
                    op_name = op_remap[op_name]
                except KeyError:
                    pass
                right = _r_walk_lark_tree(op.children[2])
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
                if len(op.children) != 2:
                    raise ValueError("unexpected ' + op.data + ' length")
                op_name = logical_remap[op.data]
                left = _r_walk_lark_tree(op.children[0])
                right = _r_walk_lark_tree(op.children[1])
                return getattr(left, op_name)(right)
            if op.data == 'funccall':
                if len(op.children) > 2:
                    raise ValueError("unexpected funccall length")
                fn = op.children[0]
                var = _r_walk_lark_tree(fn.children[0])
                op_name = str(fn.children[1])
                args = []
                if len(op.children) > 1:
                    raw_args = op.children[1].children
                    args = [_r_walk_lark_tree(ai) for ai in raw_args]
                method = getattr(var, op_name)
                return method(*args)
            if (op.data == 'or_test') or (op.data == 'and_test'):
                raise ValueError("and/or and &&/|| can not be used in vector data context, please use &/|.")
            if op.data == 'not':
                if len(op.children) != 1:
                    raise ValueError("unexpected not length")
                left = _r_walk_lark_tree(op.children[0])
                return left.__not__()  # TODO: implement
            raise ValueError("unexpected lark Tree kind: " + str(op.data))
        raise ValueError("unexpected lark parse type: " + str(type(op)))

    return _r_walk_lark_tree(op)


def _parse_by_lark(source_str, *, data_def=None, outer_environment=None):
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
    tree = parser.parse(source_str + '\n')
    # convert parse tree to our data structures for isolation
    v = _walk_lark_tree(tree, data_def=data_def, outer_environment=outer_environment)
    v.source_string = source_str
    return v

