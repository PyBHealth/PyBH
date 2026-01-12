"""
Data preprocessing module for Medical Survival Analysis Pipeline.

This module handles data loading, cleaning, imputation and formatting
specifically for Survival Analysis (Cox/Weibull) in PyMC and Lifelines.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils.config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    DURATION_COL,
    EVENT_COL,
    NUMERICAL_COLS,
    TEST_FILE,
    TRAIN_FILE,
)
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

    def load_data(self):
        """
        Load raw training and test datasets.
        """
        self.logger.substep("Loading Medical Data")

        try:
            self.train_data = pd.read_csv(DATA_PATH / TRAIN_FILE)
            self.test_data = pd.read_csv(DATA_PATH / TEST_FILE)

            with self.logger.indent():
                self.logger.dataframe_info(self.train_data, "Raw Training Data")
                self.logger.dataframe_info(self.test_data, "Raw Test Data")

            self.logger.success("Data loaded successfully")
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise

        return self.train_data.copy(), self.test_data.copy()

    def _build_pipeline(self):
        """
        Builds the Scikit-Learn preprocessing pipeline.

        CRITICAL FOR PYMC:
        - Numerical vars: Standard Scaled (Mean=0, Std=1).
        - Categorical vars: One-Hot Encoded (drop_first=True).
        - Missing values: Imputed (Median for nums, Mode for cats).
        """
        # Pipeline for numerical features
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Pipeline for categorical features
        # drop='first' is crucial to avoid multicollinearity
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
                ("num", numeric_transformer, NUMERICAL_COLS),
                ("cat", categorical_transformer, CATEGORICAL_COLS),
            ],
            verbose_feature_names_out=False,  # Keep clean column names
        )

    def preprocess_data(self):
        """
        Apply cleaning, imputation, encoding and scaling.
        Fits on Train, Transforms both Train and Test.
        """
        self.logger.substep("Preprocessing Data (Imputation + Scaling + Encoding)")

        if self.train_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1. Separating Covariates (X) from Survival Targets (T, E)
        X_train = self.train_data[NUMERICAL_COLS + CATEGORICAL_COLS]
        X_test = self.test_data[NUMERICAL_COLS + CATEGORICAL_COLS]

        # 2. Build and Fit Pipeline
        self._build_pipeline()

        # Fit on TRAIN only to avoid data leakage
        X_train_processed_arr = self.preprocessor.fit_transform(X_train)
        X_test_processed_arr = self.preprocessor.transform(X_test)

        # 3. Retrieve feature names for clear debugging/analysis
        feature_names = self._get_feature_names()

        # 4. Reconstruct DataFrames (Easier for Lifelines and Debugging)
        X_train_df = pd.DataFrame(
            X_train_processed_arr, columns=feature_names, index=self.train_data.index
        )
        X_test_df = pd.DataFrame(
            X_test_processed_arr, columns=feature_names, index=self.test_data.index
        )

        # 5. Re-attach Target Columns (Time & Event)
        # Ensure they are numeric
        for df_target, df_source in [
            (X_train_df, self.train_data),
            (X_test_df, self.test_data),
        ]:
            df_target[DURATION_COL] = df_source[DURATION_COL].astype(float)
            df_target[EVENT_COL] = df_source[EVENT_COL].astype(int)

        self.train_data_processed = X_train_df
        self.test_data_processed = X_test_df

        with self.logger.indent():
            self.logger.info(f"Features processed: {len(feature_names)}")
            self.logger.info(f"Columns: {feature_names[:5]} ...")

        self.logger.success("Preprocessing Pipeline Completed")

        return self.train_data_processed, self.test_data_processed

    def get_pymc_data(self, dataset="train"):
        """
        Returns data in a format optimized for PyMC modeling.

        Returns:
            dict: {
                'X': Matrix of covariates (design matrix),
                'time': Array of survival times,
                'event': Array of event indicators (0=censored, 1=event),
                'feature_names': list of feature names
            }
        """
        if dataset == "train":
            df = self.train_data_processed
        else:
            df = self.test_data_processed

        if df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        # Extract Covariates matrix (Drop Target cols)
        X_cols = [c for c in df.columns if c not in [DURATION_COL, EVENT_COL]]

        return {
            "X": df[X_cols].values,
            "time": df[DURATION_COL].values,
            "event": df[EVENT_COL].values,
            "feature_names": X_cols,
        }

    def _get_feature_names(self):
        """Helper to get feature names from ColumnTransformer."""
        try:
            return self.preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older scikit-learn versions
            cat_features = (
                self.preprocessor.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(CATEGORICAL_COLS)
            )
            return list(NUMERICAL_COLS) + list(cat_features)

    def inspect_censoring(self):
        """
        Log statistics about censoring rates (Useful for Weibull/Cox interpretation).
        """
        self.logger.substep("Inspecting Censoring Logic")

        for name, df in [("Train", self.train_data), ("Test", self.test_data)]:
            total = len(df)
            events = df[EVENT_COL].sum()
            censored = total - events
            pct_censored = (censored / total) * 100

            with self.logger.indent():
                msg = (
                    f"Dataset {name}: {pct_censored:.2f}% censored ({censored}/{total})"
                )
                self.logger.info(msg)


if __name__ == "__main__":
    processor = SurvivalDataProcessor()

    # 1. Load
    processor.load_data()

    # 2. Check stats
    processor.inspect_censoring()

    # 3. Process
    df_train, df_test = processor.preprocess_data()

    # 4. Get data specifically for PyMC
    pymc_input = processor.get_pymc_data("train")

    print(
        f"Ready: X shape {pymc_input['X'].shape}, "
        f"Events shape {pymc_input['event'].shape}"
    )
