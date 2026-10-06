import pandas as pd
from sklearn.datasets import fetch_openml

CAR_PURPOSES = ["new car", "used car"]


def load_auto_loans():
    """Pull German Credit from OpenML and keep only auto loans."""
    raw = fetch_openml("credit-g", version=1, as_frame=True).frame

    df = raw[raw["purpose"].isin(CAR_PURPOSES)].copy()

    # Convert pandas 'category' columns to plain text (simpler downstream)
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)

    df["default_flag"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns="class").reset_index(drop=True)
    df.insert(0, "loan_id", range(1, len(df) + 1))
    return df


def get_history_and_new_apps(new_share=0.3, seed=42):
    """
    Simulate the real-world split:
      - history: past loans WITH known outcomes (used for training)
      - new_apps: incoming applications WITHOUT outcomes (used for scoring)
    """
    df = load_auto_loans()
    new_apps = df.sample(frac=new_share, random_state=seed)
    history = df.drop(new_apps.index)
    return history, new_apps.drop(columns="default_flag")


if __name__ == "__main__":
    history, new_apps = get_history_and_new_apps()
    print(f"History: {len(history)} loans, "
          f"default rate = {history['default_flag'].mean():.1%}")
    print(f"New applications: {len(new_apps)}")
    print(history["purpose"].value_counts())
    print(history.head())