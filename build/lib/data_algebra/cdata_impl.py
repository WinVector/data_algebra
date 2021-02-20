import data_algebra.data_ops
import data_algebra.cdata


def record_map_from_simple_obj(obj):
    blocks_in = None
    blocks_out = None
    if "blocks_in" in obj.keys():
        blocks_in = data_algebra.cdata.record_spec_from_simple_obj(obj["blocks_in"])
    if "blocks_out" in obj.keys():
        blocks_out = data_algebra.cdata.record_spec_from_simple_obj(obj["blocks_out"])
    return data_algebra.cdata.RecordMap(blocks_in=blocks_in, blocks_out=blocks_out)
