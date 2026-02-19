Weibull
========

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   Hypothesis
   weibull_mastectomy

Hypothesis
----------
The Weibull model is a parametric model that makes strong assumptions about the distribution of survival times.
Unlike semi-parametric models, it assumes that the hazard rate follows a specific shape (monotonic increase or decrease). 
Validating these assumptions is crucial to ensure your predictions are not biased.

:doc:`You will find here an extensive list of these hypotheses <Hypothesis>` along with existing solutions to test them.

Example Notebook
----------------

:doc:`You'll find here <weibull_mastectomy>` an example notebook displaying how you could use the
Weibull model using the SurvivalAnalysis class.

Model
-----

.. autoclass:: PyBH.SurvivalAnalysis.pymc_models.Weibull
   :members:
   :special-members: predict_survival_function
   :exclude-members: build_model
