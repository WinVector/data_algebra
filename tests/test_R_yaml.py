
import data_algebra.yaml
import data_algebra.expr


def test_R_yaml():
    have_yaml = False
    try:
        # noinspection PyUnresolvedReferences
        import yaml  # supplied by PyYAML

        have_yaml = True
    except ImportError:
        pass

    if not have_yaml:
        return

    yaml_str = "- op:\n  - TableDescription\n  table_name:\n  - d\n  column_names:\n  - subjectID\n  - surveyCategory\n  - assessmentTotal\n  - irrelevantCol1\n  - irrelevantCol2\n- op:\n  - Extend\n  ops:\n    probability: exp ( assessmentTotal * 0.237 )\n  partition_by: ~\n  order_by: ~\n  reverse: ~\n"

    obj = yaml.safe_load(yaml_str)
    ops = data_algebra.yaml.to_pipeline(obj, parse_env=data_algebra.expr.r_parse_env())

    yaml_str_2 = "- op: TableDescription\n  table_name: d\n  column_names:\n  - subjectID\n  - surveyCategory\n  - assessmentTotal\n  - irrelevantCol1\n  - irrelevantCol2\n- op: Extend\n  ops:\n    probability: exp ( assessmentTotal * 0.237 )\n  partition_by: ~\n  order_by: ~\n  reverse: ~\n- op: Extend\n  ops:\n    total: sum ( probability )\n  partition_by: subjectID\n  order_by: ~\n  reverse: ~\n- op: Extend\n  ops:\n    probability: probability / total\n  partition_by: ~\n  order_by: ~\n  reverse: ~\n- op: Extend\n  ops:\n    sort_key: '- ( ( probability ) )'\n  partition_by: ~\n  order_by: ~\n  reverse: ~\n- op: Extend\n  ops:\n    row_number: row_number ( )\n  partition_by: subjectID\n  order_by: sort_key\n  reverse: ~\n- op: SelectRows\n  expr:\n    '': row_number = 1\n- op: SelectColumns\n  columns:\n  - subjectID\n  - surveyCategory\n  - probability\n- op: Rename\n  column_remapping:\n    diagnosis: surveyCategory\n- op: Order\n  column_remapping: ~\n  order_columns: subjectID\n  reverse: ~\n"
    obj_2 = yaml.safe_load(yaml_str_2)
    ops = data_algebra.yaml.to_pipeline(obj_2, parse_env=data_algebra.expr.r_parse_env())
