_have_black = False
try:
    # noinspection PyUnresolvedReferences
    import black

    _have_black = True
except ImportError:
    pass


# noinspection PyBroadException
def pretty_format_python(python_str: str, *, black_mode=None) -> str:
    """
    Format Python code, using black.

    :param python_str: Python code
    :param black_mode: options for black
    :return: formatted Python code
    """
    assert isinstance(python_str, str)
    formatted_python = python_str
    if _have_black:
        try:
            if black_mode is None:
                black_mode = black.FileMode()
            formatted_python = black.format_str(python_str, mode=black_mode)
        except Exception:
            pass
    return formatted_python
