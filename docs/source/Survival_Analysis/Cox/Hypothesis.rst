Hypothesis
==========

**Here is an extensive list of hypotheses you should check on your data before using Cox's model**

Non-informative Censoring
-------------------------

In survival analysis, we often deal with **censoring**. A patient is "censored" when we stop
following them before the event of interest (like a relapse or recovery) happens.

However, for a Cox Regression model to be accurate, it relies on a silent but vital assumption:
**Non-informative (or Random) Censoring.**

The core idea
^^^^^^^^^^^^^

In a clinical setting, informative censoring occurs when a **patient’s exit from a study is directly
tied to their medical condition or the treatment itself**, such as dropping out due to intolerable
side effects or becoming too weak to attend follow-up appointments.

This creates a major problem for Cox regression because the **"censored" patients are not a
random loss**. They are often the ones for whom the treatment is failing or causing harm.
By removing them from the analysis, the study is left with a "filtered" group of the healthiest
survivors, **which artificially inflates the drug's perceived safety and efficacy**.

How to tackle this issue
^^^^^^^^^^^^^^^^^^^^^^^^

The trickiest part about this assumption is that **it cannot be tested with math**. Why? Because
once a patient is censored, we have **no data on what happened to them next**. We cannot "prove"
they didn't have the event five minutes after leaving.

Because we can't test it with a formula, we must guarantee it through **rigorous data collection**:
- Researchers must try their best to keep patients in the study.
- They must document why someone left.
- If a patient leaves due to a reason related to the study (like side effects), the assumption is
violated, and the results become "biased."
