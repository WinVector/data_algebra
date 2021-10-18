"""
Flow text around a margin for presentation.
"""


def flow_text(tokens, *, align_right=70, sep_width=1):
    """

    :param tokens: list or tuple of strings
    :param align_right: integer, right alignment margin
    :param sep_width: integer, size of inline separator
    :return: list of lists of strings flowing the text to the margin
    """

    flowed = []
    length = 0
    working = []
    for i in range(len(tokens)):
        li = len(tokens[i])
        if (len(working) > 0) and ((length + sep_width + li) > align_right):
            flowed.append(working)
            length = 0
            working = []
        if len(working) > 0:
            length = length + sep_width
        length = length + li
        working.append(tokens[i])
    if len(working) > 0:
        flowed.append(working)

    return flowed
