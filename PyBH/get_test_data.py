from pathlib import Path

import pandas as pd
from lifelines.datasets import load_lung


def get_medical_dataset():
    # 1. Load dataset
    print("Downloading NCCTG Lung Cancer data...")
    df = load_lung()

    # 2. Dynamic Status Analysis
    raw_status = df["status"].unique()
    print(f"Raw values found in 'status': {raw_status}")

    df_clean = pd.DataFrame()
    df_clean["patient_id"] = range(1, len(df) + 1)
    df_clean["time"] = df["time"]

    # 3. SMART MAPPING
    # Target: 0 = Censored (Alive), 1 = Death (Event)

    # Case A: Historic SAS format (1=Censored, 2=Death)
    if set(raw_status) == {1, 2}:
        print("Detected format: 1/2 -> Converting to 0/1")
        df_clean["event_death"] = df["status"].map({1: 0, 2: 1})

    # Case B: Modern format (0=Censored, 1=Death)
    elif set(raw_status) == {0, 1}:
        print("Detected format: 0/1 -> Keeping as is")
        df_clean["event_death"] = df["status"]

    # Case C: Unknown format (Fallback)
    else:
        print(f"⚠️ Unknown format: {raw_status}. Attempting raw conversion.")
        # Assume max value is the event (death)
        max_val = max(raw_status)
        df_clean["event_death"] = (df["status"] == max_val).astype(int)

    # 4. Features
    df_clean["age"] = df["age"]
    df_clean["sex"] = df["sex"].map({1: "Male", 2: "Female"})
    df_clean["ph_ecog"] = df["ph.ecog"]
    df_clean["weight_loss"] = df["wt.loss"]
    df_clean["calories"] = df["meal.cal"]

    # 5. Save
    output_path = Path("PyBH/data/lung_cancer.csv")
    if not output_path.parent.exists():
        output_path = Path("data/lung_cancer.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_path, index=False)

    print(f"\n Health Dataset generated: {output_path}")
    print("Final verification (Must be 0 and 1):")
    print(df_clean["event_death"].value_counts())


if __name__ == "__main__":
    get_medical_dataset()
