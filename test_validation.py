import json

from data_source import get_history_and_new_apps
from validate import validate

with open("artifacts/input_schema.json") as f:
    schema = json.load(f)
_, apps = get_history_and_new_apps()


def run(name, df):
    errors, warnings = validate(df, schema)
    print(f"\n[{'FAIL' if errors else 'PASS'}] {name}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARN:  {w}")


run("Clean batch", apps)

run("Missing column", apps.drop(columns="duration"))

bad = apps.copy()
bad.loc[bad.index[0], "age"] = -5
run("Impossible age", bad)

bad = apps.copy()
bad.loc[bad.index[0], "purpose"] = "boat"
run("Unknown category", bad)

bad = apps.copy()
bad.loc[bad.index[:3], "credit_amount"] = 50000
run("Loan amounts beyond training range", bad)

bad = apps.copy()
bad["duration"] = bad["duration"].astype(object)
bad.loc[bad.index[0], "duration"] = "sixty"
run("Text in a numeric column", bad)