from pathlib import Path

from lifelines.datasets import load_recur


def get_recur_dataset():
    # 1. Load dataset
    df = load_recur()

    # 2. Select and Rename (Optional but recommended)
    # Keeping ID, times, and event
    # Renaming for clarity in the CSV
    df_final = df.copy()

    # 3. Duplicate check (Confirming need for GroupSplit)
    print(f"Row count: {len(df_final)}")
    print(f"Unique patients count: {df_final['ID'].nunique()}")

    has_duplicates = df_final["ID"].duplicated().any()
    status_msg = "YES -> Testing Group Split" if has_duplicates else "NO"
    print(f"ID duplicates detected? {status_msg}")

    # 4. Save
    output_dir = Path("PyBH/data")
    if not output_dir.exists():
        output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "bladder_cancer.csv"
    df_final.to_csv(output_path, index=False)

    print(f"\n File successfully created: {output_path}")
    print("Columns:", df_final.columns.tolist())


if __name__ == "__main__":
    get_recur_dataset()
