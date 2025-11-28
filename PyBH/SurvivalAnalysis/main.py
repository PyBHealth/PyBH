import pandas as pd
from typing import Optional

from SurvivalAnalysis.core.workflow import BayesianSurvivalWorkflow
from SurvivalAnalysis.models.cox import CoxPHBuilder
from SurvivalAnalysis.models.weibull import WeibullBuilder

def run_analysis(
    df: pd.DataFrame,
    model_type: str,
    time_col: str,
    event_col: str,
    draws: int = 2000
) -> BayesianSurvivalWorkflow:
    """
    Single entry point for the clinician.
    
    Args:
        df (pd.DataFrame): The clinical dataset.
        model_type (str): 'cox' or 'weibull'.
        time_col (str): Column name for duration.
        event_col (str): Column name for event (0/1).
        draws (int): MCMC sampling count.
        
    Returns:
        BayesianSurvivalWorkflow: The trained engine object (containing results).
    """
    
    # 1. Select the correct Model Builder based on the string input
    if model_type.lower() == "cox":
        builder = CoxPHBuilder(name="Cox Proportional Hazards")
    elif model_type.lower() == "weibull":
        builder = WeibullBuilder(name="Weibull AFT")
    else:
        raise ValueError(f"Model '{model_type}' is not supported. Choose 'cox' or 'weibull'.")
    
    # 2. Instantiate the Workflow Engine with the selected builder
    engine = BayesianSurvivalWorkflow(builder=builder)
    
    # 3. Run the full pipeline (Validate -> Build -> Fit)
    engine.fit(df, time_col, event_col, draws=draws)
    
    # 4. auto-run diagnostics and plotting (optional convenience)
    try:
        engine.check_diagnostics()
        engine.plot_survival_function()
    except Exception as e:
        print(f"Post-processing error: {e}")
    return engine