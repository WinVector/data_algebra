
from data_algebra.expr_rep import PreTerm, ColumnReference, UnQuotedStr


# a term that isn't parsed and both the SQL and Pandas realizations are functions
class FnTerm(PreTerm):
    # represent a function of columns
    def __init__(self, *,
                 pandas_fn,
                 sql_fn,
                 args=None,
                 display_form,
                 name):
        if not callable(pandas_fn):
            raise TypeError("pandas_fn type must be callable")
        if not callable(sql_fn):
            raise TypeError("sql_fn type must be callable")
        if not isinstance(display_form, str):
            raise TypeError("display_form type must be a string")
        self.pandas_fn = pandas_fn
        self.sql_fn = sql_fn
        self.display_form = display_form
        self.name = name
        if args is None:
            args = []
        if isinstance(args, str):
            args = [args]
        for v in args:
            if not isinstance(v, str):
                raise TypeError("Expected args to be None, [], or all strings")
        args = [ColumnReference(view=None, column_name=ai) for ai in args]
        self.args = args
        PreTerm.__init__(self)

    def is_equal(self, other):
        # can't use == as that builds a larger expression
        if not isinstance(other, FnTerm):
            return False
        if self.display_form != other.display_form:
            return False
        if len(self.args) != len(other.args):
            return False
        for li, ri in zip(self.args, other.args):
            if not li.is_equal(ri):
                return False
        return True

    def get_column_names(self, columns_seen):
        for vi in self.args:
            columns_seen.add(str(vi))

    def get_views(self):
        views = list()
        for ai in self.args:
            vi = ai.get_views()
            for vii in vi:
                if vi not in views:  # expectation is views is size zero or 1
                    views.append(vii)
        return views

    def replace_view(self, view):
        self.args = [ai.replace_view(view) for ai in self.args]
        return self

    def to_python(self, *, want_inline_parens=False):
        return UnQuotedStr(self.display_form)

    def to_sql(self, *, db_model):
        subs = [  # essentially just quoting identifiers
            db_model.expr_to_sql(ai, want_inline_parens=True) for ai in self.args
        ]
        return self.sql_fn(subs, db_model)

    def evaluate(self, data_frame):
        args = [ai.evaluate(data_frame) for ai in self.args]
        res = self.pandas_fn(*args)
        return res


# old adapter, phase out
# wrap a function as a user callable function in pipeline
# used for custom aggregators
def user_fn(fn, args=None, *,
            name=None,
            sql_name=None, sql_prefix=None, sql_suffix=None):
    if isinstance(fn, str):
        if name is None:
            name = fn
        fn = eval(fn)  # TODO: replace with an explicit lookup in globals()
    if not callable(fn):
        raise TypeError("expected fn to be callable")
    if name is None:
        name = fn.__name__
    if args is None:
        args = []
    if isinstance(args, str):
        args = [args]
    else:
        for v in args:
            if not isinstance(v, str):
                raise TypeError("Expect all vars names to be strings")
    if sql_prefix is None:
        sql_prefix = ''
    if sql_suffix is None:
        sql_suffix = ''
    assert isinstance(sql_prefix, str)
    assert isinstance(sql_suffix, str)
    if name is None:
        name = "UNKNOWN"
    if sql_name is None:
        sql_name = name
    return FnTerm(
        pandas_fn=fn,
        sql_fn=lambda subs, db_model: f'{sql_name}({sql_prefix} {", ".join(subs)} {sql_suffix})',
        args=args,
        display_form=name,
        name=name,
    )

