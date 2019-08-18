
import data_algebra.table_rep
import data_algebra.pipe




class ExtendNode(data_algebra.table_rep.ViewRepresentation):

    def __init__(self, source, ops):
        if not isinstance(source, data_algebra.table_rep.ViewRepresentation):
            raise Exception("source must be a ViewRepresentation")
        column_names = source.column_names.copy()
        known_cols = set(column_names)
        for ci in ops.keys():
            if ci not in known_cols:
                column_names = column_names + [ci]
        data_algebra.table_rep.ViewRepresentation.__init__(
            self,
            table_name=None,
            column_names=column_names
        )
        self._source = source
        self._ops = ops
        # make last result referable by names _ and _0
        if data_algebra._ref_to_global_namespace is not None:
            data_algebra._ref_to_global_namespace['_'] = self.column_map
            data_algebra._ref_to_global_namespace['_0'] = self.column_map

    def __repr__(self):
        return str(self._source) + " >> " + "Extend(" + str(self._ops) + ")"

    def __str__(self):
        return str(self._source) + " >> " + "Extend(" + str(self._ops) + ")"

class Extend(data_algebra.pipe.PipeStep):
    """Class to specify adding or altering columns.

       If data_algebra._ref_to_global_namespace = globals() then
       _ and _0 are set to column name maps as a side-effect.

       Example:
           from data_algebra import *
           from data_algebra.data_ops import *
           data_algebra._ref_to_global_namespace = globals() # needed to define _
           ops = (
              mk_td('d', ['x', 'y']) >>
                 Extend({'z':_.x + _.y})
            )
            print(ops)


    """

    def __init__(self, ops):
        data_algebra.pipe.PipeStep.__init__(
            self,
            name="Extend")
        self._ops = ops

    def apply(self, other):
        return ExtendNode(other, self._ops)

