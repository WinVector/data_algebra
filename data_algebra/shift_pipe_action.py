
import abc


class ShiftPipeAction(abc.ABC):
    """
    Class representing mapping a >> b to b.act_on(a). 
    This is read as "sending a to b".
    """
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def act_on(self, b):
        """
        Apply self onto b.
        """

    def __rshift__(self, b):  # override self >> b
        """
        Delegate self >> b to b.act_on(self) b is a ShiftPipeAction instance, else call self.act_on(b)
        This is read as "sending self to b".
        """
        if isinstance(b, ShiftPipeAction):
            # this is the expected path
            return b.act_on(self)
        # fall back to our action
        return self.act_on(b)

    def __rrshift__(self, b):  # override b >> self
        """
        Delegate b >> self to self.act_on(b).
        This is read as sending b to self.
        """
        return self.act_on(b)
