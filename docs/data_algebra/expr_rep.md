Module data_algebra.expr_rep
============================
Represent data processing expressions.

Functions
---------

    
`col(nm: str)`
:   represent a column reference

    
`connected_components(f, g)`
:   Compute connected components.

    
`enc_value(value)`
:   Encode a value as a PreTerm or derived class.

    
`get_columns_used(parsed_exprs) ‑> Set[str]`
:   Return set of columns used in an expression.

    
`implies_windowed(parsed_exprs: dict) ‑> bool`
:   Return true if expression implies a windowed calculation is needed.

    
`kop_expr(op, args, inline=False, method=False)`
:   three argument expression

    
`lit(x)`
:   Represent a value

    
`standardize_join_type(join_str)`
:   Replace join name with standard name.

Classes
-------

`ColumnNamer()`
:   class to generate column names using dot/attribute notation

`ColumnReference(column_name)`
:   class to represent referring to a column

    ### Ancestors (in MRO)

    * data_algebra.expr_rep.Term
    * data_algebra.expr_rep.PreTerm
    * abc.ABC

    ### Class variables

    `column_name: str`
    :

    ### Methods

    `get_column_names(self, columns_seen: Set[str]) ‑> None`
    :   Add column names to columns_seen
        :param columns_seen: set of strings
        :return:

`DictTerm(value)`
:   Class for carrying a dictionary or map.

    ### Ancestors (in MRO)

    * data_algebra.expr_rep.PreTerm
    * abc.ABC

`Expression(op: str, args, *, params=None, inline: bool = False, method: bool = False)`
:   Class for carrying an expression.

    ### Ancestors (in MRO)

    * data_algebra.expr_rep.Term
    * data_algebra.expr_rep.PreTerm
    * abc.ABC

    ### Methods

    `get_column_names(self, columns_seen: Set[str]) ‑> None`
    :   Add column names to columns_seen
        :param columns_seen: set of strings
        :return:

    `get_method_names(self, methods_seen: Set[str]) ‑> None`
    :   Add names of methods used to methods_seen.
        
        :param methods_seen: set to collect results
        :return: None

    `to_python(self, *, want_inline_parens: bool = False) ‑> data_algebra.expr_rep.PythonText`
    :   Convert parsed expression into a string
        
        :param want_inline_parens: bool,
                if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText

`ListTerm(value)`
:   Class to hold a collection.

    ### Ancestors (in MRO)

    * data_algebra.expr_rep.PreTerm
    * abc.ABC

    ### Methods

    `get_column_names(self, columns_seen: Set[str]) ‑> None`
    :   Add column names to columns_seen
        :param columns_seen: set of strings
        :return:

`PreTerm()`
:   abstract base class, without combination ability

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.expr_rep.DictTerm
    * data_algebra.expr_rep.ListTerm
    * data_algebra.expr_rep.Term

    ### Class variables

    `source_string: Optional[str]`
    :

    ### Methods

    `act_on(self, arg, *, expr_walker: data_algebra.expression_walker.ExpressionWalker)`
    :   Apply expression to argument.

    `get_column_names(self, columns_seen: Set[str]) ‑> None`
    :   Add column names to columns_seen
        :param columns_seen: set of strings
        :return: None

    `get_method_names(self, methods_seen: Set[str]) ‑> None`
    :   Add method names to methods_seen
        :param methods_seen: set of strings
        :return: None

    `is_equal(self, other)`
    :   Check if this expression code is the same as another expression.

    `to_python(self, *, want_inline_parens: bool = False) ‑> data_algebra.expr_rep.PythonText`
    :   Convert parsed expression into a string
        
        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :return: PythonText

    `to_source(self, *, want_inline_parens=False, dialect='Python') ‑> data_algebra.expr_rep.PythonText`
    :   Convert to source code.
        
        :param want_inline_parens: bool, if True put parens around complex expressions that don't already have a grouper.
        :param dialect: dialect to emit (not currently used)
        :return: PythonText

`PythonText(s: str, *, is_in_parens: bool = False)`
:   Class for holding text representation of Python, with possible additional annotations.
    str() method returns only the text for interoperability.

    ### Class variables

    `is_in_parens: bool`
    :

    `s: str`
    :

`Term()`
:   Abstract intermediate class with combination ability

    ### Ancestors (in MRO)

    * data_algebra.expr_rep.PreTerm
    * abc.ABC

    ### Descendants

    * data_algebra.expr_rep.ColumnReference
    * data_algebra.expr_rep.Expression
    * data_algebra.expr_rep.Value

    ### Methods

    `abs(self)`
    :   Return absolute value of items (vectorized).

    `all(self)`
    :   Return True if all items True (vectorized).

    `any(self)`
    :   Return True if any items True (vectorized).

    `any_value(self)`
    :   Return any_value (vectorized).

    `arccos(self)`
    :   Return trigometric arccos() (in radians) of item (vectorized).

    `arccosh(self)`
    :   Return hyperbolic arccosh() of item (vectorized).

    `arcsin(self)`
    :   Return trigometric arcsin() (in radians) of item (vectorized).

    `arcsinh(self)`
    :   Return hyperbolic arcsinh() of item (vectorized).

    `arctan(self)`
    :   Return trigometric arctan() (in radians) of item (vectorized).

    `arctan2(self, other)`
    :   Return trigometric arctan2() (in radians) of item (vectorized).

    `arctanh(self)`
    :   Return hyperbolic arctanh() of item (vectorized).

    `around(self, other)`
    :   Return rounded values (given numer of decimals) as real (vectorized).

    `as_int64(self)`
    :   Cast as int (vectorized).

    `as_str(self)`
    :   Cast as string (vectorized).

    `base_Sunday(self)`
    :   Compute prior Sunday date from date (self for Sundays) (vectorized).

    `bfill(self)`
    :   Return vector with missing vallues filled (vectorized).

    `ceil(self)`
    :   Return ceil() (smallest int no smaller than, as real type) of item (vectorized).

    `co_equalizer(self, x)`
    :   Compute the connected components (co-equalizer).

    `coalesce(self, x)`
    :   Replace missing values with alternative (vectorized).

    `coalesce_0(self)`
    :   Replace missing values with zero (vectorized).

    `concat(self, x)`
    :   Concatinate strings (vectorized).

    `cos(self)`
    :   Return trigometric cos() (in radians) of item (vectorized).

    `cosh(self)`
    :   Return hyperbolic cosh() of item (vectorized).

    `count(self)`
    :   Return number of non-NA cells (vectorized).

    `cumcount(self)`
    :   Return cumulative number of non-NA cells (vectorized).

    `cummax(self)`
    :   Return cumulative maximum (vectorized).

    `cummin(self)`
    :   Return cumulative minimum (vectorized).

    `cumprod(self)`
    :   Return cumprod() of items (vectorized).

    `cumsum(self)`
    :   Return cumsum() of items (vectorized).

    `date_diff(self, other)`
    :   Compute difference in dates in days (vectorized).

    `datetime_to_date(self)`
    :   Convert date time to date (vectorized).

    `dayofmonth(self)`
    :   Convert date to day of month (vectorized).

    `dayofweek(self)`
    :   Convert date to date of week (vectorized).

    `dayofyear(self)`
    :   Convert date to date of year (vectorized).

    `exp(self)`
    :   Return exp() of items (vectorized).

    `expm1(self)`
    :   Return exp() - 1 of items (vectorized).

    `ffill(self)`
    :   Return vector with missing vallues filled (vectorized).

    `first(self)`
    :   Return first (vectorized).

    `float_divide(self, other)`
    :

    `floor(self)`
    :   Return floor() (largest int no larger than, as real type) of item (vectorized).

    `fmax(self, other)`
    :   Return per row fmax of items and other (ignore missing, vectorized).

    `fmin(self, other)`
    :   Return per row fmin of items and other (ignore missing, vectorized).

    `format_date(self, format=None)`
    :   Format string as a date (vectorized).

    `format_datetime(self, format=None)`
    :   Format string as a date time (vectorized).

    `if_else(self, x, y)`
    :   Vectorized selection between two argument vectors.
        if_else(True, 1, 2) > 1, if_else(False, 1, 2) -> 2.
        None propagating behavior if_else(None, 1, 2) -> None.

    `is_bad(self)`
    :   Return which items in a numeric column are bad (null, None, nan, or infinite) (vectorized).

    `is_in(self, x)`
    :   Set membership (vectorized).

    `is_inf(self)`
    :   Return which items are inf (vectorized).

    `is_monotonic_decreasing(self)`
    :   Return vector True if monotonic decreasing (vectorized).

    `is_monotonic_increasing(self)`
    :   Return vector True if monotonic increasing (vectorized).

    `is_nan(self)`
    :   Return which items are nan (vectorized).

    `is_null(self)`
    :   Return which items are null (vectorized).

    `last(self)`
    :   Return last (vectorized).

    `log(self)`
    :   Return base e logarithm of items (vectorized).

    `log10(self)`
    :   Return base 10 logarithm of items (vectorized).

    `log1p(self)`
    :   Return base e logarithm of 1 + items (vectorized).

    `mapv(self, value_map, default_value=None)`
    :   Map values to values (vectorized).

    `max(self)`
    :   Return max (vectorized).

    `maximum(self, other)`
    :   Return per row maximum of items and other (propogate missing, vectorized).

    `mean(self)`
    :   Return mean (vectorized).

    `median(self)`
    :   Return median (vectorized).

    `min(self)`
    :   Return min (vectorized).

    `minimum(self, other)`
    :   Return per row minimum of items and other (propogate missing, vectorized).

    `mod(self, other)`
    :   Return modulo of items (vectorized).

    `month(self)`
    :   Convert date to month (vectorized).

    `nunique(self)`
    :   Return number of unique items (vectorized).

    `parse_date(self, format=None)`
    :   Parse string as a date (vectorized).

    `parse_datetime(self, format=None)`
    :   Parse string as a date time (vectorized).

    `quarter(self)`
    :   Convert date to quarter (vectorized).

    `rank(self)`
    :   Return item rangings (vectorized).

    `remainder(self, other)`
    :   Return remainder of items (vectorized).

    `round(self)`
    :   Return rounded values (nearest integer, subject to some rules) as real (vectorized).

    `shift(self, periods=None)`
    :   Return shifted items (vectorized).

    `sign(self)`
    :   Return -1, 0, 1 as sign of item (vectorized).

    `sin(self)`
    :   Return trigometric sin() (in radians) of item (vectorized).

    `sinh(self)`
    :   Return hyperbolic sinh() of item (vectorized).

    `size(self)`
    :   Return number of items (vectorized).

    `sqrt(self)`
    :   Return sqrt of items (vectorized).

    `std(self)`
    :   Return sample standard devaition (vectorized).

    `sum(self)`
    :   Return sum() of items (vectorized).

    `tanh(self)`
    :   Return hyperbolic tanh() of item (vectorized).

    `timestamp_diff(self, other)`
    :   Compute difference in timestamps in seconds (vectorized).

    `trimstr(self, start, stop)`
    :   Trim string start (inclusive) to stop (exclusive) (vectorized).

    `var(self)`
    :   Return sample variance (vectorized).

    `weekofyear(self)`
    :   Convert date to week of year (vectorized).

    `where(self, x, y)`
    :   Vectorized selection between two argument vectors.
        if_else(True, 1, 2) > 1, if_else(False, 1, 2) -> 2.
        numpy.where behavior: where(None, 1, 2) -> 2

    `year(self)`
    :   Convert date to year (vectorized).

`Value(value)`
:   Class for holding constants.

    ### Ancestors (in MRO)

    * data_algebra.expr_rep.Term
    * data_algebra.expr_rep.PreTerm
    * abc.ABC