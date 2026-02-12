Kaplan Meier
============

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   Hypothesis
   kaplan_meier

Hypothesis
----------

**The Kaplan-Meier estimator relies on several hypotheses that must be verified for it to be valid.**
There are multiple ways to verify each of them. While the solutions to verify these assumptions range from quick approximate checks to sophisticated statistical tests, none are implemented in this library at the moment.

:doc:`Here, you will find an extensive list of these hypotheses along with existing solutions to test them. <Hypothesis>`

Example Notebook
----------------

:doc:`You'll find here <kaplan_meier>` an example notebook displaying how you could use the Kaplan-Meier estimator using the SurvivalAnalysis class.

Model
-----

.. autoclass:: lifelines.KaplanMeierFitter
   :members: