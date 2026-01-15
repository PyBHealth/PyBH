"""
Data splitting module for Medical Survival Analysis.
Handles standard stratified splits and grouped splits (to prevent data leakage).
Now handles Missing Values (NaN) in targets before splitting.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from utils.auto_schema import SchemaDetector
from utils.config import (
    DATA_PATH,
    RANDOM_SEED,
    SOURCE_FILE,
    TEST_FILE,
    TEST_SIZE,
    TRAIN_FILE,
)
from utils.logger import get_logger


def find_grouping_column(df):
    """
    Searches for a column defining a group (Patient, Hospital, Family).
    If duplicates exist in this column, the split must be grouped.
    """
    group_keywords = [
        "id",
        "patient",
        "ref",
        "index",
        "subject",
        "matricule",
        "nip",
        "hospital",
        "center",
        "centre",
        "site",
        "clinic",
        "family",
        "famille",
        "cluster",
        "country",
        "pays",
        "ctryname",
        "nation",
    ]

    for col in df.columns:
        if any(kw in col.lower() for kw in group_keywords):
            return col
    return None


def split_dataset():
    logger = get_logger()
    logger.substep("Splitting Data (Auto-detecting Leakage Risks)")

    source_path = DATA_PATH / SOURCE_FILE

    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file '{SOURCE_FILE}' missing.")

    # 1. Load
    df = pd.read_csv(source_path)
    logger.info(f"Source loaded: {len(df)} rows")

    # 2. Schema Analysis
    detector = SchemaDetector(df)
    schema = detector.infer()

    # --- 3. PREVENTIVE CLEANING ---
    # Cannot stratify on NaN. A patient without Event or Time is unusable.
    subset_check = []
    if schema.event_col:
        subset_check.append(schema.event_col)
    if schema.duration_col:
        subset_check.append(schema.duration_col)

    # If targets found, drop rows where they are empty
    if subset_check:
        initial_len = len(df)
        df = df.dropna(subset=subset_check)
        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            logger.warning(
                f"Dropped {dropped_count} rows with missing Target (Time/Event)."
            )
            logger.info(f"Remaining rows: {len(df)}")

            # If empty after cleaning, stop
            if len(df) == 0:
                raise ValueError("Dataset is empty after dropping missing targets!")

    # 4. Split Logic
    group_col = find_grouping_column(df)
    train_df = None
    test_df = None

    # CASE A: Group detected with duplicates
    if group_col and df[group_col].duplicated().any():
        logger.warning(
            f"Potential leakage detected! Group column '{group_col}' has duplicates."
        )
        logger.info("➔ Switching to GroupShuffleSplit (keeping groups together).")

        gss = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        for train_idx, test_idx in gss.split(df, groups=df[group_col]):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

    # CASE B: Independent Data
    else:
        if group_col:
            logger.info(
                f"Group column '{group_col}' found but entries are unique. "
                "Using Standard Split."
            )
        else:
            logger.info("No grouping column found. Assuming independent rows.")

        stratify_col = None
        if schema.event_col:
            logger.info(f"Stratifying split by event column: '{schema.event_col}'")
            stratify_col = df[schema.event_col]

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=stratify_col,
        )

    # 5. Save
    train_path = DATA_PATH / TRAIN_FILE
    test_path = DATA_PATH / TEST_FILE

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with logger.indent():
        logger.success(f"Train set saved: {train_path.name} ({len(train_df)} rows)")
        logger.success(f"Test set saved:  {test_path.name} ({len(test_df)} rows)")

        if schema.event_col:
            train_cens = 1 - train_df[schema.event_col].mean()
            test_cens = 1 - test_df[schema.event_col].mean()
            msg = f"Censoring Rate - Train: {train_cens:.1%} | Test: {test_cens:.1%}"
            if abs(train_cens - test_cens) > 0.05:
                logger.warning(f"{msg} (Significant imbalance!)")
            else:
                logger.info(msg)


if __name__ == "__main__":
    split_dataset()
