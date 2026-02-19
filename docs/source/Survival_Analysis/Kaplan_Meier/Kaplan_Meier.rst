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
There are multiple ways to verify each of them. While the solutions to verify these assumptions range from quick approximate checks to sophisticated statistical tests, **none are implemented in this library at the moment.** Users must ensure these assumptions are met through manual data verification and external statistical validation.

:doc:`Here, you will find an extensive list of these hypotheses along with existing solutions to test them. <Hypothesis>`

Example Notebook
----------------

:doc:`You'll find here <Kaplan Meier>` an example notebook displaying how you could use the Kaplan-Meier estimator using the SurvivalAnalysis class.

Model
-----

The Kaplan-Meier estimator is a non-parametric statistic used to estimate the survival function from lifetime data.
The formula for the survival function :math:`S(t)` is defined as:

.. math::

    S(t) = \prod_{i: t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)

Where:
- :math:`d_i` is the number of events (e.g., deaths) at time :math:`t_i`.
- :math:`n_i` is the number of subjects at risk (still alive and not censored) just before time :math:`t_i`.

.. autoclass:: lifelines.KaplanMeierFitter
   :members:
