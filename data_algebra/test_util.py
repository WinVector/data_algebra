
# needed for the eval
# noinspection PyUnresolvedReferences
from data_algebra.data_ops import *

def formats_to_self(ops):
    str1 = str(ops)
    ops2 = eval(str1)
    str2 = str(ops2)
    return str1 == str2
