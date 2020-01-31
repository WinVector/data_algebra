from abc import ABC


class EvalModel(ABC):
    def __init__(self):
        pass

    def eval(self, ops, data_map, *, eval_env=None, narrow=True):
        """
        apply ops to data frames in data_map

        :param ops OperatorPlatform, operation to apply OperatorPlatform
        :param data_map map from data frame names to data frames
        :param eval_env environment to look for symbols in
        :param narrow logical, if True don't copy unexpected columns
        :return: result DataFrame
        """
        raise NotImplementedError("base class called")

    # noinspection PyPep8Naming
    def transform(self, ops, X, *, eval_env=None, narrow=True):
        """
        apply ops to data frame X, may or may not commute with composition

        :param ops OperatorPlatform, operation to apply
        :param X input data frame
        :param eval_env environment to look for symbols in
        :param narrow logical, if True don't copy unexpected columns
        :return: transformed DataFrame
        """
        tabs = ops.get_tables()
        if len(tabs) is not 1:
            raise ValueError("ops must use exaclty one table")
        tname = [k for k in tabs.keys()][0]
        return self.eval(ops, {tname: X}, eval_env=eval_env, narrow=narrow)
