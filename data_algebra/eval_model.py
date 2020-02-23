from abc import ABC


class EvalModel(ABC):
    def __init__(self):
        pass

    def to_pandas(self, handle, *, data_map=None):
        raise NotImplementedError("base class called")

    def eval(self, ops, *, data_map=None, result_name=None, eval_env=None, narrow=True):
        """
        apply ops to data frames in data_map

        :param ops OperatorPlatform, operation to apply OperatorPlatform
        :param data_map map from data frame names to data frame representations, altered by eval.
        :parap result_name Name for result.
        :param eval_env environment to look for symbols in
        :param narrow logical, if True don't copy unexpected columns
        :return: result data frame representation
        """
        raise NotImplementedError("base class called")
