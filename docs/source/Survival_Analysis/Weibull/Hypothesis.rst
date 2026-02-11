Hypothesis
==========

**Here is an extensive list of hypotheses you should check on your data before using the Weibull model**

    - :ref:`Non-informative censoring <NIC>`
    - :ref:`Weibull Distribution <WD>`
    - :ref:`Linearity (Log-Linear Relationship) <LLR>`

.. _NIC:
1 - Non-informative Censoring
-----------------------------

Just like the Cox model, the Weibull model relies heavily on the assumption of **Non-informative
(or Random) Censoring**. A patient is "censored" when we stop following them before the event
of interest happens.

The core idea
^^^^^^^^^^^^^

This assumption implies that the mechanism causing a patient to leave the study (censoring) is
completely independent of the probability of the event occurring.

If patients drop out because they are feeling too sick (or conversely, because they feel completely
cured and stop showing up), the censoring is **informative**. This biases the estimation of the
Weibull parameters (shape and scale), leading to incorrect predictions of survival times.

How to tackle this issue
^^^^^^^^^^^^^^^^^^^^^^^^

As with semi-parametric models, this assumption **cannot be statistically tested** using the data
alone because the future of censored patients is unknown.
To ensure this hypothesis holds:
    - **Study Design**: Ensure rigorous follow-up protocols.
    - **Sensitivity Analysis**: Compare "Best Case" vs. "Worst Case" scenarios for the censored data.
    - **Documentation**: Record reasons for dropout meticulously.

.. _WD:
2 - Weibull Distribution (Parametric Assumption)
------------------------------------------------

Unlike the Cox model, which leaves the baseline hazard unspecified (semi-parametric), the Weibull
model is **parametric**. It assumes that the baseline hazard follows a specific mathematical shape
determined by the parameters :math:`\alpha` (shape) and :math:`\lambda` (scale).

The core idea
^^^^^^^^^^^^^

The Weibull distribution assumes that the hazard rate is **monotonic**. This means the risk of the
event must be either:
    - **Constantly increasing** over time (e.g., wear and tear, aging).
    - **Constantly decreasing** over time (e.g., post-surgery recovery, infant mortality).
    - **Constant** (Exponential case).

If your data exhibits a "Bathtub" hazard (high risk at start, low middle, high end) or a hump shape,
the Weibull assumption is violated.

How to tackle this issue
^^^^^^^^^^^^^^^^^^^^^^^^

You must visually verify that the data fits a Weibull distribution using the **Log-Cumulative Hazard Plot**.

1. Estimate the survival function :math:`S(t)` using the Kaplan-Meier method.
2. Plot :math:`\ln(-\ln(S(t)))` against :math:`\ln(t)`.

**Interpretation**:
    - If the plot forms a roughly **straight line**, the Weibull assumption is valid.
    - The slope of this line corresponds to the shape parameter :math:`k` (or :math:`\alpha`).
    - If the line curves significantly, a different parametric model (like Log-Normal or Log-Logistic) might be more appropriate.

.. _LLR:
3 - Linearity (Log-Linear Relationship)
---------------------------------------

The Weibull model is often interpreted as an **Accelerated Failure Time (AFT)** model. This assumes
a log-linear relationship between the covariates and the survival time.

The core idea
^^^^^^^^^^^^^

The model assumes that the covariates act multiplicatively on the time scale. In mathematical terms,
we assume that the logarithm of the survival time :math:`\ln(T)` is linearly related to the
covariates :math:`X`.

.. math::
    \ln(T) = \mu + \beta X + \sigma W

Where :math:`W` is the error term. This implies that the effect of a covariate is to "accelerate" or
"decelerate" the passage of time towards the event.

How to tackle this issue
^^^^^^^^^^^^^^^^^^^^^^^^

To check this, we use **Cox-Snell Residuals**.

If the model fits the data well (including the linearity assumption), the Cox-Snell residuals should
follow a standard Exponential distribution with a mean of 1.

**Interpretation of the Plot**:
    - Plot the **Cumulative Hazard of the Residuals** against the **Residuals themselves**.
    - If the assumption holds, the points should align closely with the **45-degree diagonal line**.
    - Significant deviations suggest that the functional form of the covariates (e.g., linearity) or the distributional assumption is incorrect.

.. [Collett2015] Collett, D. (2015). Modelling Survival Data in Medical Research.
   CRC press.
.. [Carroll2003] Carroll, K. J. (2003). On the use and utility of the Weibull model in the analysis
   of survival data. Controlled clinical trials, 24(6), 682-701.
|
|
|
    | *Signé :*
    | **Jean Van Dyk**
    | *Etudiant IMT Atlantique*