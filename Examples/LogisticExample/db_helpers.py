import io
from pprint import pprint
import psycopg2    # http://initd.org/psycopg/
import pandas      # https://pandas.pydata.org
import yaml        # https://pyyaml.org


def is_numeric(col):
    try:
        0.0 + col
        return True
    except Exception as ex:
        return False


def insert_table(conn, d, table_name):
    cr = [
        d.columns[i].lower()
        + " "
        + ("double precision" if is_numeric(d[d.columns[i]]) else "VARCHAR")
        for i in range(d.shape[1])
    ]
    create_stmt = "CREATE TABLE " + table_name + " ( " + ", ".join(cr) + " )"
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS " + table_name)
    conn.commit()
    cur.execute(create_stmt)
    conn.commit()
    buf = io.StringIO(d.to_csv(index=False, header=False, sep="\t"))
    cur.copy_from(buf, "d", columns=[c for c in d.columns])
    conn.commit()


def read_query(conn, q):
    cur = conn.cursor()
    cur.execute(q)
    r = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    return pandas.DataFrame(columns=colnames, data=r)


def read_table(conn, table_name):
    return read_query(conn, "SELECT * FROM " + table_name)
