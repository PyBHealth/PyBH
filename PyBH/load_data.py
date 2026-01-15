"""
Data preprocessing module for Medical Survival Analysis Pipeline.

This module handles:
1. Automatic data splitting (lazy loading via split_data.py).
2. Automatic schema detection (Time, Event, Numerical, Categorical).
3. Automatic handling of 'Censor' columns (inverting 1=Censored to 0=Event).
4. Automatic duration calculation (Stop - Start).
5. Data cleaning (Imputation) and formatting (Scaling/Encoding).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- 1. Import split module (Lazy Loading) ---
from split_data import split_dataset
from utils.auto_schema import SchemaDetector
from utils.config import DATA_PATH, OVERRIDE_SCHEMA, SOURCE_FILE, TEST_FILE, TRAIN_FILE
from utils.logger import get_logger


class SurvivalDataProcessor:
    """
    Data processor for medical survival datasets.
    Prepares data for PyMC (Bayesian) and Lifelines (Frequentist).
    """

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.preprocessor = None  # Sklearn Pipeline
        self.logger = get_logger()
        self.train_data_processed = None
        self.test_data_processed = None
        self.schema = None  # Will hold DatasetSchema

    def load_data(self):
        """
        Load raw training and test datasets.
        If they don't exist, triggers split_data.py automatically.
        Detects schema, handles censorship logic, and computes duration.
        """
        self.logger.substep("Checking Data Availability")

        train_path = DATA_PATH / TRAIN_FILE

        # --- 1. Automatic Split Logic ---
        if not train_path.exists():
            self.logger.warning(f"Train file '{TRAIN_FILE}' not found.")

            # Check if source file exists
            source_path = DATA_PATH / SOURCE_FILE
            if source_path.exists():
                msg = f"Found source file '{SOURCE_FILE}'. Auto-splitting now..."
                self.logger.info(msg)
                split_dataset()  # Automatic call to split script
            else:
                msg = f"Critical: Neither '{TRAIN_FILE}' nor Source '{SOURCE_FILE}' found."  # noqa: E501
                self.logger.error(msg)
                raise FileNotFoundError(msg)
        else:
            self.logger.info("Train/Test files found. Loading directly.")

        # --- 2. Standard Load ---
        self.logger.substep("Loading Medical Data & Detecting Schema")

        try:
            self.train_data = pd.read_csv(DATA_PATH / TRAIN_FILE)
            if TEST_FILE:
                self.test_data = pd.read_csv(DATA_PATH / TEST_FILE)
            else:
                self.test_data = self.train_data.iloc[:0].copy()

            # --- 3. Auto-Detection Schema ---
            detector = SchemaDetector(self.train_data)
            self.schema = detector.infer()
            self._apply_overrides()

            # --- 4. CENSORSHIP INVERSION HANDLING ---
            # If column name contains 'censor', 'cens', etc.
            # Common convention: 1 = Censored (Alive), 0 = Event (Dead)
            # PyMC/Lifelines convention: 1 = Event (Dead), 0 = Censored
            if any(x in self.schema.event_col.lower() for x in ["cens", "censor"]):
                self.logger.info(
                    f"Censorship column detected ('{self.schema.event_col}'). "
                    "Inverting values..."
                )
                self.logger.info("Logic applied: New_Event = 1 - Old_Censor")

                # Mathematical inversion: 1->0 and 0->1
                self.train_data[self.schema.event_col] = (
                    1 - self.train_data[self.schema.event_col]
                )
                self.test_data[self.schema.event_col] = (
                    1 - self.test_data[self.schema.event_col]
                )

            # --- 5. BINARY EVENT SAFETY CHECK ---
            # Verify we only have 0 and 1 (no 2, 3, etc.)
            unique_events = self.train_data[self.schema.event_col].unique()
            if not set(unique_events).issubset({0, 1}):
                self.logger.warning(
                    f"Non-binary event values detected: {unique_events}"
                )
                self.logger.info(
                    "Normalizing Event column to 0/1 (Values > 0 become 1)..."
                )

                self.train_data[self.schema.event_col] = (
                    self.train_data[self.schema.event_col] > 0
                ).astype(int)
                self.test_data[self.schema.event_col] = (
                    self.test_data[self.schema.event_col] > 0
                ).astype(int)

            # --- 6. Duration Calculation (Start / Stop) ---
            # If detector found both Start AND End columns
            if self.schema.start_col and self.schema.duration_col:
                self.logger.info(
                    f"Interval detected: '{self.schema.start_col}' -> "
                    f"'{self.schema.duration_col}'"
                )
                self.logger.info("Computing real duration (Stop - Start)...")

                new_dur_col = "calculated_duration"

                # Safe duration calculation
                self.train_data[new_dur_col] = (
                    self.train_data[self.schema.duration_col]
                    - self.train_data[self.schema.start_col]
                )
                self.test_data[new_dur_col] = (
                    self.test_data[self.schema.duration_col]
                    - self.test_data[self.schema.start_col]
                )

                # Schema update:
                # Add old start/stop columns to "dropped" list
                self.schema.dropped_cols.extend(
                    [self.schema.duration_col, self.schema.start_col]
                )
                # New target is the calculated duration
                self.schema.duration_col = new_dur_col

            # Informative logging
            with self.logger.indent():
                self.logger.dataframe_info(self.train_data, "Raw Training Data")
                self.logger.info(f"Target Time:  '{self.schema.duration_col}'")
                self.logger.info(f"Target Event: '{self.schema.event_col}'")
                self.logger.info(
                    f"Features: {len(self.schema.numerical_cols)} numerical, "
                    f"{len(self.schema.categorical_cols)} categorical."
                )

            self.logger.success("Data loaded and schema detected")
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise

        return self.train_data.copy(), self.test_data.copy()

    def _apply_overrides(self):
        """Apply manual overrides from config if auto-detection fails."""
        if not OVERRIDE_SCHEMA:
            return

        # Helper to remove columns from feature lists if they become targets
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
            # If we drop a column, remove it from features
            for col in self.schema.dropped_cols or []:
                _remove_from_features(col)

    def _build_pipeline(self):
        """Builds the Scikit-Learn preprocessing pipeline."""
        if not self.schema:
            raise ValueError("Schema not detected. Call load_data() first.")

        # Pipeline for numerical features (Imputation + Scaling)
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Pipeline for categorical features (Imputation + OneHot)
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
        Apply cleaning, imputation, encoding and scaling.
        Fits on Train, Transforms both Train and Test.
        """
        self.logger.substep("Preprocessing Data")

        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Select useful columns (exclude IDs and Dropped cols)
        useful_cols = (
            self.schema.numerical_cols
            + self.schema.categorical_cols
            + [self.schema.duration_col, self.schema.event_col]
        )

        # Work on copies
        train_subset = self.train_data[useful_cols].copy()
        test_subset = (
            self.test_data[useful_cols].copy()
            if not self.test_data.empty
            else pd.DataFrame(columns=useful_cols)
        )

        # 2. Build and Fit Pipeline
        self._build_pipeline()

        # Fit on TRAIN only (features only)
        X_train = train_subset[
            self.schema.numerical_cols + self.schema.categorical_cols
        ]
        X_train_processed_arr = self.preprocessor.fit_transform(X_train)

        if not test_subset.empty:
            X_test = test_subset[
                self.schema.numerical_cols + self.schema.categorical_cols
            ]
            X_test_processed_arr = self.preprocessor.transform(X_test)
        else:
            X_test_processed_arr = []

        # 3. Retrieve feature names
        feature_names = self._get_feature_names()

        # 4. Reconstruct DataFrames
        self.train_data_processed = pd.DataFrame(
            X_train_processed_arr, columns=feature_names, index=train_subset.index
        )

        if not test_subset.empty:
            self.test_data_processed = pd.DataFrame(
                X_test_processed_arr, columns=feature_names, index=test_subset.index
            )
        else:
            self.test_data_processed = pd.DataFrame(columns=feature_names)

        # 5. Re-attach Target Columns (Time & Event)
        for df_target, df_source in [
            (self.train_data_processed, train_subset),
            (self.test_data_processed, test_subset),
        ]:
            if not df_source.empty:
                df_target[self.schema.duration_col] = df_source[
                    self.schema.duration_col
                ].astype(float)
                df_target[self.schema.event_col] = df_source[
                    self.schema.event_col
                ].astype(int)

        self.logger.success("Preprocessing Pipeline Completed")
        return self.train_data_processed, self.test_data_processed

    def get_pymc_data(self, dataset="train"):
        """Returns data formatted for PyMC (X matrix, time vector, event vector)."""
        df = (
            self.train_data_processed
            if dataset == "train"
            else self.test_data_processed
        )

        if df is None or df.empty:
            raise ValueError(f"Data for '{dataset}' is not processed or empty.")

        # Extract Covariates matrix (Everything except Time and Event)
        X_cols = [
            c
            for c in df.columns
            if c not in [self.schema.duration_col, self.schema.event_col]
        ]

        return {
            "X": df[X_cols].values,
            "time": df[self.schema.duration_col].values,
            "event": df[self.schema.event_col].values,
            "feature_names": X_cols,
        }

    def _get_feature_names(self):
        """Helper to get feature names from ColumnTransformer."""
        try:
            return self.preprocessor.get_feature_names_out()
        except AttributeError:
            cat_features = (
                self.preprocessor.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(self.schema.categorical_cols)
            )
            return list(self.schema.numerical_cols) + list(cat_features)

    def inspect_censoring(self):
        """Log statistics about censoring rates."""
        self.logger.substep("Inspecting Censoring Logic")

        if not self.schema:
            self.logger.warning("Schema not detected. Run load_data() first.")
            return

        dfs = [("Train", self.train_data)]
        if self.test_data is not None and not self.test_data.empty:
            dfs.append(("Test", self.test_data))

        for name, df in dfs:
            total = len(df)
            events = df[self.schema.event_col].sum()
            censored = total - events
            pct_censored = (censored / total) * 100

            with self.logger.indent():
                msg = (
                    f"Dataset {name}: {pct_censored:.2f}% censored ({censored}/{total})"
                )
                self.logger.info(msg)


if __name__ == "__main__":
    # Complete flow test
    processor = SurvivalDataProcessor()

    # 1. Load (Triggers split + detection schema + duration calculation)
    processor.load_data()

    # 2. Stats
    processor.inspect_censoring()

    # 3. Process (Impute + Scale + Encode)
    df_train, df_test = processor.preprocess_data()

    # 4. Export PyMC
    pymc_input = processor.get_pymc_data("train")

    print(
        f"Ready: X shape {pymc_input['X'].shape}, "
        f"Events shape {pymc_input['event'].shape}"
    )
