"""
Data preprocessing module for Medical Survival Analysis Pipeline.
Full Dataset Version (No Train/Test Split).

This module handles:
1. Loading the full dataset directly.
2. Cleaning rows with missing targets (Time/Event).
3. Automatic schema detection (including ID columns).
4. Automatic censorship handling.
5. Preprocessing (Imputation + Scaling) on features only (preserving IDs).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.auto_schema import SchemaDetector
from utils.config import DATA_PATH, OVERRIDE_SCHEMA, SOURCE_FILE
from utils.logger import get_logger


class SurvivalDataProcessor:
    """
    Data processor for medical survival datasets.
    Uses the entire dataset (no train/test split).
    """

    def __init__(self):
        self.data = None  # Holds the full dataframe
        self.preprocessor = None
        self.logger = get_logger()
        self.data_processed = None
        self.schema = None

    def load_data(self):
        """
        Loads the source file, detects schema, cleans missing targets,
        and handles censorship logic.
        """
        self.logger.substep("Loading Full Dataset")

        source_path = DATA_PATH / SOURCE_FILE
        if not source_path.exists():
            msg = f"Source file '{SOURCE_FILE}' not found at {source_path}."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # 1. Load
        self.data = pd.read_csv(source_path)
        self.logger.info(f"Loaded {len(self.data)} rows from {SOURCE_FILE}")

        # 2. Detect Schema
        detector = SchemaDetector(self.data)
        self.schema = detector.infer()
        self._apply_overrides()

        # --- SCHEMA LOGGING BLOCK ---
        self.logger.substep("Schema Classification Details")
        with self.logger.indent():
            self.logger.info(f"• ID Col:          {self.schema.id_col}")
            self.logger.info(f"• Start Col:       {self.schema.start_col}")
            self.logger.info(f"• End or Dur Col:  {self.schema.duration_col}")
            self.logger.info(f"• Event Col:       {self.schema.event_col}")
            self.logger.info(f"• Numerical Cols:  {self.schema.numerical_cols}")
            self.logger.info(f"• Categorical Cols:{self.schema.categorical_cols}")
            self.logger.info(f"• Dropped Cols:    {self.schema.dropped_cols}")

        # 3. CLEANING MISSING TARGETS (Essential!)
        subset_check = []
        if self.schema.event_col:
            subset_check.append(self.schema.event_col)

        # Check for duration OR start column
        dur_col = self.schema.duration_col
        start_col = self.schema.start_col

        if dur_col and dur_col in self.data.columns:
            subset_check.append(dur_col)
        elif start_col and start_col in self.data.columns:
            subset_check.append(start_col)

        if subset_check:
            initial_len = len(self.data)
            self.data = self.data.dropna(subset=subset_check)
            dropped = initial_len - len(self.data)
            if dropped > 0:
                self.logger.warning(
                    f"Dropped {dropped} rows with missing Target (Time/Event)."
                )

        # 4. CENSORSHIP INVERSION
        if any(x in self.schema.event_col.lower() for x in ["cens", "censor"]):
            msg = (
                f"Censorship column detected ('{self.schema.event_col}'). "
                "Inverting values..."
            )
            self.logger.info(msg)
            self.data[self.schema.event_col] = 1 - self.data[self.schema.event_col]

        # 5. BINARY EVENT CHECK
        unique_events = self.data[self.schema.event_col].unique()
        if not set(unique_events).issubset({0, 1}):
            self.logger.warning(
                f"Non-binary event values detected: {unique_events}. Normalizing..."
            )
            self.data[self.schema.event_col] = (
                self.data[self.schema.event_col] > 0
            ).astype(int)

        # 6. DURATION CALCULATION
        if self.schema.start_col and self.schema.duration_col:
            self.logger.info("Computing real duration (Stop - Start)...")
            new_dur_col = "calculated_duration"
            self.data[new_dur_col] = (
                self.data[self.schema.duration_col] - self.data[self.schema.start_col]
            )

            self.schema.dropped_cols.extend(
                [self.schema.duration_col, self.schema.start_col]
            )
            self.schema.duration_col = new_dur_col

        # 7. LOG DATA SUMMARY
        self.logger.substep("Data Summary")
        with self.logger.indent():
            self.logger.dataframe_info(self.data, "Full Dataset")

        return self.data.copy()

    def _apply_overrides(self):
        """Apply manual overrides from config."""
        if not OVERRIDE_SCHEMA:
            return

        def _remove_from_features(col):
            if not col:
                return
            if col in self.schema.numerical_cols:
                self.schema.numerical_cols.remove(col)
            if col in self.schema.categorical_cols:
                self.schema.categorical_cols.remove(col)

        if OVERRIDE_SCHEMA.get("duration_col"):
            self.schema.duration_col = OVERRIDE_SCHEMA["duration_col"]
            _remove_from_features(self.schema.duration_col)

        if OVERRIDE_SCHEMA.get("event_col"):
            self.schema.event_col = OVERRIDE_SCHEMA["event_col"]
            _remove_from_features(self.schema.event_col)

        if OVERRIDE_SCHEMA.get("numerical_cols"):
            self.schema.numerical_cols = OVERRIDE_SCHEMA["numerical_cols"]
        if OVERRIDE_SCHEMA.get("categorical_cols"):
            self.schema.categorical_cols = OVERRIDE_SCHEMA["categorical_cols"]
        if OVERRIDE_SCHEMA.get("dropped_cols"):
            self.schema.dropped_cols = OVERRIDE_SCHEMA["dropped_cols"]
            for col in self.schema.dropped_cols or []:
                _remove_from_features(col)

    def _build_pipeline(self):
        """Builds the Scikit-Learn preprocessing pipeline."""
        if not self.schema:
            raise ValueError("Schema not detected. Call load_data() first.")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first", handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.schema.numerical_cols),
                ("cat", categorical_transformer, self.schema.categorical_cols),
            ],
            verbose_feature_names_out=False,
        )

    def preprocess_data(self):
        """
        Apply cleaning, imputation, encoding and scaling on the full dataset.
        Preserves the ID column without transforming it.
        """
        self.logger.substep("Preprocessing Data")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Selection : Nums + Cats + Time + Event + ID (if present)
        useful_cols = (
            self.schema.numerical_cols
            + self.schema.categorical_cols
            + [self.schema.duration_col, self.schema.event_col]
        )
        if self.schema.id_col:
            useful_cols.append(self.schema.id_col)

        # Work on copy
        df_subset = self.data[useful_cols].copy()

        # 2. Build Pipeline (Only for Nums and Cats)
        self._build_pipeline()

        X = df_subset[self.schema.numerical_cols + self.schema.categorical_cols]
        X_processed_arr = self.preprocessor.fit_transform(X)

        # Retrieve feature names
        try:
            feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            cat_feat = (
                self.preprocessor.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(self.schema.categorical_cols)
            )
            feature_names = list(self.schema.numerical_cols) + list(cat_feat)

        # 3. Reconstruct DataFrame
        self.data_processed = pd.DataFrame(
            X_processed_arr, columns=feature_names, index=df_subset.index
        )

        # 4. Re-attach Targets
        self.data_processed[self.schema.duration_col] = df_subset[
            self.schema.duration_col
        ].astype(float)
        self.data_processed[self.schema.event_col] = df_subset[
            self.schema.event_col
        ].astype(int)

        # 5. Re-attach ID (Untouched)
        if self.schema.id_col:
            self.data_processed[self.schema.id_col] = df_subset[
                self.schema.id_col
            ].values

        save_path = DATA_PATH / "processed_dataset.csv"
        self.data_processed.to_csv(save_path, index=False)
        self.logger.info(f"Processed data saved to: {save_path}")

        self.logger.success("Preprocessing Pipeline Completed")
        return self.data_processed

    def get_pymc_data(self):
        """
        Returns data formatted for PyMC.
        Includes: X matrix, time, event, features, and group_ids (if any).
        """
        if self.data_processed is None or self.data_processed.empty:
            raise ValueError("Data is not processed or empty.")

        df = self.data_processed

        # Exclude Time, Event AND ID from the feature matrix X
        exclude_cols = [self.schema.duration_col, self.schema.event_col]
        if self.schema.id_col:
            exclude_cols.append(self.schema.id_col)

        X_cols = [c for c in df.columns if c not in exclude_cols]

        result = {
            "X": df[X_cols].values,
            "time": df[self.schema.duration_col].values,
            "event": df[self.schema.event_col].values,
            "feature_names": X_cols,
        }

        # Add groups for PyMC hierarchical models
        if self.schema.id_col:
            result["group_ids"] = df[self.schema.id_col].values

        return result

    def inspect_censoring(self):
        """Log statistics about censoring rates."""
        if not self.schema or self.data is None:
            return

        total = len(self.data)
        events = self.data[self.schema.event_col].sum()
        censored = total - events
        pct = (censored / total) * 100

        with self.logger.indent():
            self.logger.info(f"Censoring: {pct:.2f}% ({censored}/{total})")


if __name__ == "__main__":
    # Test flow
    processor = SurvivalDataProcessor()
    processor.load_data()
    processor.inspect_censoring()
    processor.preprocess_data()
    out = processor.get_pymc_data()
    print(f"Ready. X shape: {out['X'].shape}")
    if "group_ids" in out:
        print(f"Groups present: {len(out['group_ids'])} items")
