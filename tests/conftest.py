import os
import gzip
import pickle

import pytest

import data_algebra.eval_cache
import data_algebra.test_util

# to see:
# py.test -s -v


# cache results to speed up re-testing in some situations
@pytest.fixture(scope="session", autouse=True)
def user_pytest_start(request):
    data_algebra.test_util.global_test_result_cache = data_algebra.eval_cache.ResultCache()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    global_test_result_cache_fname = os.path.join(
        dir_path, "data_algebra_test_cache.pkl.gz"
    )
    try:
        with gzip.open(global_test_result_cache_fname, "rb") as in_f:
            res = data_algebra.test_util.global_test_result_cache = pickle.load(in_f)
        assert isinstance(res, data_algebra.eval_cache.ResultCache)
        data_algebra.test_util.global_test_result_cache = res
    except FileNotFoundError:
        pass

    def user_pytest_end():
        """write dirty cache on test system teardown"""
        if ((data_algebra.test_util.global_test_result_cache is not None)
                and data_algebra.test_util.global_test_result_cache.dirty):
            with gzip.open(global_test_result_cache_fname, "wb") as out_f:
                pickle.dump(
                    data_algebra.test_util.global_test_result_cache, out_f
                )

    request.addfinalizer(user_pytest_end)
