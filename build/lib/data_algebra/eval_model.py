from abc import ABC


class EvalModel(ABC):
    def __init__(self):
        self.temp_id = 0

    def mk_tmp_name(self, data_map):
        new_name = None
        seen = set()
        if data_map is not None:
            seen = set(data_map.keys())
        while (new_name is None) or (new_name in seen):
            new_id = self.temp_id
            self.temp_id = new_id + 1
            new_name = "TMP_" + str(new_id).zfill(7) + "_T"
        return new_name

    def managed_eval(self, ops, *, data_map=None, result_name=None, narrow=True):
        """
        apply ops to data frames in data_map

        :param ops OperatorPlatform, operation to apply OperatorPlatform
        :param data_map map from data frame names to data frame representations, altered by eval.
        :param result_name Name for result.
        :param narrow logical, if True don't copy unexpected columns
        :return: result name
        """
        raise NotImplementedError("base class called")
