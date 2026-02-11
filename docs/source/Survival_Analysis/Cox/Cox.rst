Cox Proportional Hazards
=========================

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   Hypothesis
   cox_mastectomy

Hypothesis
----------

**The Cox model relies on multiple hypotheses that need to be verified for it to be valid.**
There are multiple ways to verify each of them. While the solutions to verify these assumptions
range from quick approximate checks to sophisticated statistical tests, none are implemented in
this library at the moment.
:doc:`Here, you will find an extensive list of these hypotheses along with existing solutions to test them. <Hypothesis>`

Example Notebook
----------------

:doc:`You'll find here <cox_mastectomy>` an example notebook displaying how you could use the
Cox model using the SurviavlAnalysis class.

Model
-----

.. autoclass:: PyBH.SurvivalAnalysis.pymc_models.Cox
   :members:
   :special-members: predict_survival_function
   :exclude-members: build_model
