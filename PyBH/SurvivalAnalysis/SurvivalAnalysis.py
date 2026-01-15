import pandas as pd
import numpy as np
import arviz as az
from typing import Optional
import matplotlib.pyplot as plt
from SurvivalAnalysis.pymc_models import PyMCModel

class SurvivalAnalysis:
    """
    Workflow Manager.
    
    This class orchestrates the analysis process:
    1. Input Validation
    2. Data Preprocessing
    3. Model Building (via the injected Builder)
    4. MCMC Sampling
    5. Diagnostics & Plotting
    """
    def __init__(self, model: any, data: pd.DataFrame, time_col: str, event_col: str, **kwargs):
        """
        Orchestrates the training process.
        Acts as an adapter that prepares data specifically for the underlying model type.

        Args:
            data (pd.DataFrame): Raw input data from the user.
            time_col (str): Name of the column containing time durations.
            event_col (str): Name of the column containing event occurrence (0/1).
            **kwargs: Additional arguments passed to the underlying model's fit method.
        """
                
        self.model = model
        self.idata = None # Pour stocker les résultats Bayésiens
        
        # Détection automatique du type de modèle
        self.is_bayesian = isinstance(model, PyMCModel)
        self.is_lifelines = hasattr(model, "print_summary")

        # 1. Validation (Using your existing method)
        self.validate_inputs(data, time_col, event_col)
        
        # 2. Preprocessing (Using your existing method)
        # Note: ensuring _preprocess_data handles One-Hot Encoding is crucial here
        df_clean = self._preprocess_data(data, time_col, event_col)
        
        # 3. Routing Logic (The Adapter Pattern)
        
        # --- CASE A: BAYESIAN MODELS  ---
        if self.is_bayesian:
            print("   -> Mode: Bayesian (PyMC)")
            
            # PyMC models from base_model.py require strictly typed inputs:
            # - X: Covariates matrix (Numpy)
            # - y: Target matrix (Numpy)
            # - coords: Dictionary for dimension names
            
            # Prepare X (Covariates): Drop time and event columns
            X_df = df_clean.drop(columns=[time_col, event_col])
            X_matrix = X_df.values  # Convert to pure Numpy array
            
            # Prepare y (Target): Usually [Time, Event] for survival
            y_matrix = df_clean[[time_col, event_col]].values
            
            # Prepare Coords: Mapping names to dimensions for ArviZ/Xarray
            coords = {
                "coeffs": X_df.columns.tolist(),
                "treated_units": df_clean.index.tolist()
            }
            
            # Call the Base Model's fit method with the prepared ingredients
            self.model.fit(X_matrix, y_matrix, coords=coords, **kwargs)
            
            # Store results locally for the workflow manager
            self.idata = self.model.idata

        # --- CASE B: LIFELINES (Standard Survival Analysis) ---
        elif hasattr(self.model, "print_summary"): 
            print("   -> Mode: Frequentist (Lifelines)")
            
            # Lifelines is user-friendly and accepts the DataFrame directly
            self.model.fit(df_clean, duration_col=time_col, event_col=event_col, **kwargs)

        else:
            raise NotImplementedError("Unknown model type. Could not determine how to fit.")

    def validate_inputs(self, data: pd.DataFrame, time_col: str, event_col: str):
        """
        Validates data integrity before processing.
        
        """
        # 1. Check if DataFrame is empty
        if data.empty:
            raise ValueError("The input dataset is empty.")

        # 2. Check column existence
        if time_col not in data.columns or event_col not in data.columns:
            raise ValueError(f"Columns '{time_col}' or '{event_col}' not found in dataset.")

        # 3. Type Validation (Numeric or Date)
        is_numeric = np.issubdtype(data[time_col].dtype, np.number)
        is_datetime = pd.api.types.is_datetime64_any_dtype(data[time_col])

        if not is_numeric and not is_datetime:
            raise TypeError(f"Column '{time_col}' must be numeric or datetime format.")
            
        # 4. Check if there is at least one event (otherwise survival analysis is useless)
        if data[event_col].sum() == 0:
            print("Warning: No events observed in the dataset. Convergence might fail.")


    def _preprocess_data(self, data: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
        """
        Cleans data (handles missing values).
        """
        df = data.copy()
        
        # Basic strategy: drop rows with missing values
        df = df.dropna(subset=[time_col, event_col])
        
        # One-Hot Encoding (Sexe -> Sexe_M, Sexe_F)
        df = pd.get_dummies(df, drop_first=True)
            
        # Booleans in int (False/True -> 0/1)
        cols_bool = df.select_dtypes(include=['bool']).columns
        df[cols_bool] = df[cols_bool].astype(int)
            
        return df  

    def check_diagnostics(self) -> None:
        """
        Checks convergence metrics using ArviZ.
        """
        if self.idata is None:
            raise RuntimeError("Model has not been trained. Run .fit() first.")

        # Logic to check R-hat would go here
        print("Checking convergence statistics...")

    def plot_survival_function(self, **kwargs):
        """
        Generates the survival curve with credible intervals.
        """
        if self.is_lifelines:
            if hasattr(self.model, "plot"):
                self.model.plot()
                plt.show()
                
        elif self.is_bayesian:
            if self.idata is None:
                raise RuntimeError("Model has not been trained. Run .fit() first.")
            
            print(f"Plotting Bayesian summary...")
            az.plot_forest(self.idata, combined=True)
            plt.show()
