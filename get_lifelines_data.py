from pathlib import Path

from lifelines.datasets import load_gbsg2


def get_gbsg2_dataset():
    # 1. Load dataset from lifelines
    print("Downloading GBSG2 dataset...")
    df = load_gbsg2()

    # 2. Prepare output path
    output_dir = Path("PyBH/data")
    # Adjust if running from root directory
    if not output_dir.exists():
        output_dir = Path("data")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "breast_cancer.csv"

    # 3. Save to CSV
    df.to_csv(output_file, index=False)

    print(f" File successfully created: {output_file}")
    print("\nColumn preview:")
    print(df.columns.tolist())


if __name__ == "__main__":
    get_gbsg2_dataset()
