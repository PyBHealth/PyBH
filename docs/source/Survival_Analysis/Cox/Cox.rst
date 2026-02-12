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
   :no-members:
   :special-members: predict_survival_function
   :exclude-members: build_model

**Mathematical Equivalence:**

    The hazard rate for individual :math:`i` in time interval :math:`j` is:

    .. math::
        \lambda_{ij} = \lambda_j \exp(X_i \beta)

    where :math:`\lambda_j` is the baseline hazard for interval :math:`j`.
    In a survival model, the log-likelihood contribution of an observation
    is given by:

    .. math::
        \log L_{ij} = d_{ij} \log(\lambda_{ij}) - \int_{t \in I_j} \lambda_{ij} dt

    Under the assumption that :math:`\lambda_j` is constant over the interval
    duration :math:`\Delta t_{ij}`, the integral simplifies to:

    .. math::
        \log L_{ij} = d_{ij} (\log(\Delta t_{ij}) + \log(\lambda_j) +
        X_i \beta) - (\Delta t_{ij} \lambda_j e^{X_i \beta})

    This is identical (up to a constant :math:`\log(\Delta t_{ij})`) to the
    log-likelihood of a Poisson distribution :math:`\text{Poisson}(\mu_{ij})`
    where:
