Module data_algebra.shift_pipe_action
=====================================

Classes
-------

`ShiftPipeAction()`
:   Class representing mapping a >> b to b.act_on(a).
    This is read as "sending a to b".

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * data_algebra.arrow.Arrow
    * data_algebra.cdata.RecordMap
    * data_algebra.data_ops_types.OperatorPlatform
    * data_algebra.db_model.DBHandle
    * data_algebra.db_model.DBModel

    ### Methods

    `act_on(self, b, *, correct_ordered_first_call: bool = False)`
    :   Apply self onto b.
        This is read as "self acting on b."
        
        :param b: item to act on, or item that has been sent to self.
        :param correct_ordered_first_call: if True indicates this is from not the fallback paths.