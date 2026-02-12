Kaplan Meier
==================

Cette section contient des examples pratiques d'utilisation de PyBH pour l'analyse de survie.

.. toctree::
   :maxdepth: 1
   :caption: Notebooks:
   :titlesonly:
   :hidden:

   kaplan_meier.ipynb


Overview
--------

This notebook demonstrates how to perform **Kaplan-Meier survival analysis** using PyBH's `SurvivalAnalysis` class integrated with the `lifelines` library. It provides a practical guide to analyzing survival data, particularly in medical research contexts where understanding patient outcomes over time is critical.

The notebook uses the **mastectomy dataset**, which contains survival information for breast cancer patients, making it an excellent real-world example for survival analysis.

Key Concepts
------------

The Kaplan-Meier Estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Kaplan-Meier estimator is a non-parametric method that estimates the **survival function** :math:`S(t)`, representing the probability that a subject survives longer than time :math:`t`.

Key advantages:
- Handles **censored data** (e.g., patients still alive at study end)
- Non-parametric (no assumption about data distribution)
- Intuitive and widely used in medical research

The Kaplan-Meier formula is:

.. math::

    S(t) = \prod_{i: t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)

where:
- :math:`d_i` = number of events at time :math:`t_i`
- :math:`n_i` = number of subjects at risk just before time :math:`t_i`

Censoring
^^^^^^^^^

Censoring occurs when we don't observe the event of interest during the study. For example:
- A patient moves away (lost to follow-up)
- The study ends before the event occurs
- The patient experiences an unrelated death

The Kaplan-Meier estimator properly accounts for these censored observations, making it ideal for real-world medical data.

Notebook Contents
-----------------

1. **Imports and Setup**
   - Required libraries: pandas, numpy, matplotlib, PyMC, lifelines, and PyBH
   - The `KaplanMeierFitter` from lifelines handles the statistical computation
   - PyBH's `SurvivalAnalysis` class manages the workflow

2. **Dataset Loading**
   - Uses the mastectomy dataset via PyMC's data utility
   - Dataset contains 3 main columns:
     
     - ``time``: Survival time in months
     - ``event``: Whether death was observed (1) or censored (0)
     - ``metastasized``: Categorical variable indicating cancer spread (yes/no)

3. **Data Exploration**
   - Summary statistics on survival times
   - Event and censoring breakdown
   - Distribution of categorical variables

4. **Model Fitting with PyBH**
   - Demonstrates using PyBH's `SurvivalAnalysis` class
   - The class provides:
     
     - Automatic data validation
     - Preprocessing (e.g., one-hot encoding for categorical variables)
     - Unified interface for different survival models

5. **Visualization**
   - Kaplan-Meier survival curves showing overall survival trajectory
   - Comparison of survival curves between groups (with/without metastasis)
   - Visual interpretation of survival probabilities over time

6. **Stratified Analysis**
   - Compares survival outcomes between patient groups
   - Demonstrates how metastasis status affects survival probability
   - Shows the power of stratification in revealing group differences

7. **Median Survival Time**
   - Extracts the time at which 50% of patients have experienced the event
   - Provides a single summary statistic for comparing groups

Usage Instructions
------------------

Running the Notebook
^^^^^^^^^^^^^^^^^^^^

1. Ensure all dependencies are installed (pandas, numpy, matplotlib, PyMC, lifelines, PyBH)
2. Execute cells in order, starting with imports
3. The notebook automatically loads the mastectomy dataset
4. PyBH's `SurvivalAnalysis` class handles all preprocessing

Key Parameters
^^^^^^^^^^^^^^

When initializing a `SurvivalAnalysis` object:

.. code-block:: python

    survival_analysis = SurvivalAnalysis(
        model=kmf,              # The fitted model (KaplanMeierFitter)
        data=df,                # Input data (pandas DataFrame)
        time_col='time',        # Column name for survival times
        event_col='event'       # Column name for event indicator
    )

Interpretation
^^^^^^^^^^^^^^

**Survival Curves:**
- The y-axis represents the probability of survival
- The x-axis represents time (months in this case)
- Horizontal segments indicate periods with no events
- Vertical drops occur when events are observed

**Stratified Curves:**
- Curves that are further apart indicate strong group differences
- Curves that converge suggest similar survival patterns
- Compare curves visually or use log-rank tests for statistical testing

**Median Survival:**
- The time at which the survival curve crosses 0.5 probability
- Useful for comparing groups with a single number
- May be undefined if more than 50% are censored

Practical Applications
----------------------

This analysis approach is valuable for:

- **Clinical Research**: Comparing treatment outcomes
- **Epidemiology**: Understanding disease progression and prognosis
- **Engineering**: Reliability and failure time analysis
- **Business Analytics**: Customer churn and retention analysis
- **Public Health**: Monitoring population survival trends

Extensions and Advanced Topics
------------------------------

Beyond the basic Kaplan-Meier analysis shown in this notebook:

1. **Log-Rank Tests**: Formally test differences between survival curves
2. **Cox Proportional Hazards**: Parametric regression for survival data
3. **Weibull Regression**: Assumes specific probability distribution
4. **Confidence Intervals**: Quantify uncertainty in survival estimates
5. **Multiple Comparisons**: Handle more than two groups

Refer to the PyBH documentation and lifelines library for advanced analyses.

Related Resources
-----------------

- **Cox Regression Notebook**: Parametric survival analysis with covariates
- **Weibull Regression Notebook**: Distribution-based survival modeling
- **PyBH Documentation**: API reference and examples
- **lifelines Library**: Comprehensive survival analysis package

Code Examples
-------------

**Basic Kaplan-Meier Fit**

.. code-block:: python

    from lifelines import KaplanMeierFitter
    from PyBH.SurvivalAnalysis.SurvivalAnalysis import SurvivalAnalysis
    
    kmf = KaplanMeierFitter()
    survival_analysis = SurvivalAnalysis(
        model=kmf,
        data=data,
        time_col='time',
        event_col='event'
    )

**Stratified Analysis**

.. code-block:: python

    kmf_no_meta = KaplanMeierFitter()
    survival_no_meta = SurvivalAnalysis(
        model=kmf_no_meta,
        data=data[data['metastasized'] == 'no'],
        time_col='time',
        event_col='event'
    )

**Visualization**

.. code-block:: python

    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function(label='All Patients')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Survival Curve')
    plt.show()

Common Issues and Solutions
-----------------------------

**Issue: Median survival time is very large or infinite**
- Indicates that more than 50% of subjects are censored
- The event may be rare or the follow-up period too short
- Consider using confidence intervals or other percentiles instead

**Issue: Curves are jagged**
- Normal behavior when sample sizes are small
- Represents actual event occurrences in the data
- Consider increasing follow-up time or sample size

**Issue: Different results with different software**
- Ensure consistent handling of ties and censoring
- Verify that time and event columns are correctly specified
- Check data preprocessing steps

Conclusion
----------

This notebook provides a practical introduction to Kaplan-Meier survival analysis using PyBH and lifelines. It demonstrates:

✓ Loading and exploring survival data
✓ Fitting Kaplan-Meier models with PyBH
✓ Visualizing survival curves
✓ Comparing survival between groups
✓ Extracting summary statistics

The workflow is generalizable to other survival analysis datasets, making it a template for future analyses in medical research, reliability engineering, and other fields where time-to-event data is important.

| *Signé :*
    | **Eve Bodot**
    | *Etudiant IMT Atlantique*