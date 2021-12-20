import os
import gzip
import pickle

import pytest

import data_algebra.test_util

# to see:
# py.test -s -v


def load_global_test_cache(fname):
    with gzip.open(fname, "rb") as in_f:
        data_algebra.test_util.global_test_result_cache = pickle.load(in_f)
        n = len(data_algebra.test_util.global_test_result_cache)
        print(f"len(data_algebra.test_util.global_test_result_cache) == {n}")


# cache results to speed up re-testing in some situations
@pytest.fixture(scope="session", autouse=True)
def user_pytest_start(request):
    data_algebra.test_util.global_test_result_cache = dict()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    global_test_result_cache_fname = os.path.join(
        dir_path, "data_algebra_test_cache.pkl.gz"
    )
    try:
        load_global_test_cache(global_test_result_cache_fname)
    except FileNotFoundError:
        pass

    def user_pytest_end():
        n = len(data_algebra.test_util.global_test_result_cache)
        print(f"len(data_algebra.test_util.global_test_result_cache) == {n}")
        with gzip.open(global_test_result_cache_fname, "wb") as out_f:
            data_algebra.test_util.global_test_result_cache = pickle.dump(
                data_algebra.test_util.global_test_result_cache, out_f
            )

    request.addfinalizer(user_pytest_end)
