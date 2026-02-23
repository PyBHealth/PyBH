import sys
from pathlib import Path

import lifelines.datasets

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
# Change this name to load a different dataset.
# Valid examples: "lung", "rossi", "recur", "gbsg2", "dd", "canadian_senators"
TARGET_DATASET = "lung"
# ==========================================


def get_lifelines_dataset(dataset_name):
    """
    Dynamically loads a dataset from the lifelines library.
    """
    # 1. Construct the function name (e.g., 'load_lung')
    function_name = f"load_{dataset_name}"

    # 2. Check if this function exists in lifelines
    if not hasattr(lifelines.datasets, function_name):
        print(f"❌ Error: Dataset '{dataset_name}' not found.")
        print(f"   (Function '{function_name}' does not exist in lifelines.datasets)")
        print("   Try: lung, rossi, gbsg2, recur, dd, waltons...")
        sys.exit(1)

    # Get the function (e.g., load_lung)
    loader_func = getattr(lifelines.datasets, function_name)

    print(f"⬇️  Downloading dataset '{dataset_name}'...")

    # 3. Execute loading
    try:
        df = loader_func()
    except Exception as e:
        print(f"❌ Error loading: {e}")
        sys.exit(1)

    # 4. Prepare output directory
    output_dir = Path("PyBH/data")
    # Adjustment if run from root
    if not output_dir.exists():
        output_dir = Path("data")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save as CSV
    output_file = output_dir / f"{dataset_name}.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ File successfully created: {output_file}")
    print(f"   Dimensions: {df.shape}")
    print("\nColumn preview:")
    print(df.columns.tolist())


if __name__ == "__main__":
    # Bonus: Allow passing name as command line argument
    # Usage: python get_lifelines_data.py rossi
    if len(sys.argv) > 1:
        TARGET_DATASET = sys.argv[1]

    get_lifelines_dataset(TARGET_DATASET)
