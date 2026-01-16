PyBH.SurvivalAnalysis.pymc_models module
==========================================

This module contains Bayesian survival models implemented with PyMC, designed to work with the 
:class:`PyBH.SurvivalAnalysis.SurvivalAnalysis` workflow manager.

Base Class
----------

.. autoclass:: PyBH.SurvivalAnalysis.pymc_models.PyMCModel
   :members:
   :show-inheritance:
   :undoc-members:

Cox Proportional Hazards Model
-------------------------------

.. autoclass:: PyBH.SurvivalAnalysis.pymc_models.Cox
   :members:
   :show-inheritance:
   :special-members: __init__, build_model, fit, predict_survival_function
   :exclude-members: __weakref__
   
   The Cox model uses a piecewise constant baseline hazard and exploits the mathematical 
   equivalence between the Cox Proportional Hazards model and Poisson regression.
   
   The docstring above contains detailed mathematical formulations showing how the Cox model
   is equivalent to a Poisson regression model under the piecewise constant hazard assumption.

