import data_algebra.yaml

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
    r_parse_env = {'exp': lambda x: x.exp()}
    ops = data_algebra.yaml.to_pipeline(obj, parse_env=r_parse_env)