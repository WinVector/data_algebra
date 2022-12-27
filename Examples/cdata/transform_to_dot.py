import pandas
import graphviz
from data_algebra.cdata import *
from typing import Iterable
import pandas as pd


def pandas_to_dot_lines(d, *, nm: str):
    assert isinstance(d, pandas.DataFrame)
    value_map = dict()
    lines = []
    # define nodes
    # grid idea from https://graphviz.org/Gallery/undirected/grid.html
    for j in range(d.shape[1]):
        lines.append(f'{nm}_c_{j} [label="{d.columns[j]}"]')
        for i in range(d.shape[0]):
            nd = f'{nm}_{i}_{j}'
            label = f'{d.iloc[i, j]}'
            lines.append(f'{nd} [label="{label}"]')
            value_map[label] = nd
    # lay out grid
    for j in range(d.shape[1]):
        lines.append(
            " -> ".join([f'{nm}_c_{j}'] + [f'{nm}_{i}_{j}' for i in range(d.shape[0])])
        )
    lines.append("rank=same { "
        + " -> ".join([f'{nm}_c_{j}' for j in range(d.shape[1])])
        + " }")
    for i in range(d.shape[0]):
        lines.append("rank=same { "
            + " -> ".join([f'{nm}_{i}_{j}' for j in range(d.shape[1])])
            + " }")
    return lines, value_map


def transform_to_dot(mp: RecordMap):
    assert isinstance(mp, RecordMap)
    d_input = mp.example_input()
    d_output = mp.transform(d_input)
    d_input = data_algebra.data_model.lookup_data_model_for_dataframe(d_input).to_pandas(d_input)
    d_output = data_algebra.data_model.lookup_data_model_for_dataframe(d_output).to_pandas(d_output)
    lines_input, v_input =  pandas_to_dot_lines(d_input, nm="in")
    lines_output, v_output =  pandas_to_dot_lines(d_output, nm="out")
    value_cells = [v for v in d_input.to_numpy().flatten() if v.endswith(" value")]
    lines_connect = [f"{v_input[v]} -> {v_output[v]} [constraint=false]" for v in value_cells]
    src = """
        fontname="Helvetica,Arial,sans-serif"
        node [fontname="Helvetica,Arial,sans-serif"]
        edge [fontname="Helvetica,Arial,sans-serif"]
        layout=dot
        labelloc = "t"
        node [shape=plaintext]
    """.split("\n")
    src2 = """
        node [fontname="Helvetica,Arial,sans-serif", fontcolor="blue"]
        edge [weight=1000 style=dashed, color="blue", arrowhead="none"]
    """.split("\n")
    src3 = """
        edge [weight=1000 style=dashed, color="orange", arrowhead="none"]
        node [fontname="Helvetica,Arial,sans-serif", fontcolor="orange"]
    """.split("\n")
    src4 = """
        edge [weight=0, style=solid, arrowhead="vee", color="black"]
    """.split("\n")
    dot = graphviz.Digraph(body=[l + "\n" for l in src + src2 + lines_input + src3 + lines_output + src4 + lines_connect  ])
    return dot



def format_table(d, *, record_id_cols: Iterable[str], control_id_cols: Iterable[str]):
    local_data_model = data_algebra.data_model.lookup_data_model_for_dataframe(d)
    d = local_data_model.to_pandas(d)
    record_id_cols = list(record_id_cols)
    control_id_cols = list(control_id_cols)
    record_id_col_pairs = [("record id", c) for c in d.columns if c in set(record_id_cols)]
    control_id_col_pairs = [("record structure", c) for c in d.columns if c in set(control_id_cols)]
    value_id_col_pairs = [("value", c) for c in d.columns if (c not in set(record_id_cols)) and (c not in set(control_id_cols))]
    d = pd.DataFrame({
        (cc, cn): d[cn] for (cc, cn) in record_id_col_pairs + control_id_col_pairs + value_id_col_pairs
    })
    d =  (
        d.style
            .set_properties(**{'background-color': '#FFE4C4'}, subset=record_id_col_pairs)
            .set_properties(**{'background-color': '#7FFFD4'}, subset=control_id_col_pairs)
    )
    return d
