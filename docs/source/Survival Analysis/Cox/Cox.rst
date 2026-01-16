Cox Proportional Hazards Model
-------------------------------

.. autoclass:: PyBH.SurvivalAnalysis.pymc_models.Cox
   :members:
   :show-inheritance:
   :special-members: predict_survival_function
   
   The Cox model uses a piecewise constant baseline hazard and exploits the mathematical 
   equivalence between the Cox Proportional Hazards model and Poisson regression.
   
   The docstring above contains detailed mathematical formulations showing how the Cox model
   is equivalent to a Poisson regression model under the piecewise constant hazard assumption.
