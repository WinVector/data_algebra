
import types

class SimpleNamespaceDict(types.SimpleNamespace):
    """Allow square-bracket lookup on SimpleNamespace

       Example:
           import data_algebra.SimpleNamespaceDict
           d = data_algebra.SimpleNamespaceDict.SimpleNamespaceDict(**{'x':1, 'y':2})
           d.x
           var_name = 'y'
           d[var_name]

    """

    def __init__(self, **kwargs):
        types.SimpleNamespace.__init__(self, **kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]