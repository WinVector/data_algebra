
import os
import datetime

from google.cloud import bigquery

import data_algebra
import data_algebra.test_util
from data_algebra.data_ops import *
import data_algebra.BigQuery

import pytest

bq_client = None
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/johnmount/big_query/big_query_jm.json"
try:
    gac = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    bq_client = bigquery.Client()
except KeyError:
    pass
bq_handle = data_algebra.BigQuery.BigQueryModel().db_handle(bq_client)
data_catalog = 'data-algebra-test'
data_schema = 'test_1'
tables_to_delete = set()

table_name = f'{data_catalog}.{data_schema}.pytest_temp_d'
tables_to_delete.add(table_name)

