Hypothesis
==========

**Here is the list of fundamental hypotheses you should check on your data before using the Kaplan-Meier estimator:**

    - :ref:`Non-informative censoring <NIC_KM>`
    - :ref:`Independence between survival and recruitment <ISR>`
    - :ref:`Precise events in time <EPT>`

.. _NIC_KM:

1 - Non-informative (or Random) Censoring
-----------------------------------------

Just like the Cox model, the accuracy of the Kaplan-Meier estimator relies on the assumption that censoring is **non-informative**.

The core idea
^^^^^^^^^^^^^

This means that the fact that an individual is censored should not provide information about their future probability of survival. In other words, patients who leave the study prematurely must have the same risk profile as those who remain.

How to tackle this issue
^^^^^^^^^^^^^^^^^^^^^^^^

**This hypothesis cannot be tested mathematically** because we lack data after the moment of censoring. As this library does not provide automated checks for this, validity must be guaranteed through the quality of data collection:
    - Precisely document the reason for each study exit.
    - Ensure that exits are not related to a deterioration in the patient's health status (e.g., exiting because the patient is too ill to continue).

.. _ISR:

2 - Independence between survival and recruitment
-------------------------------------------------

The model assumes that survival probabilities are the same for individuals recruited at the beginning or at the end of the study.

The core idea
^^^^^^^^^^^^^

If care protocols improve significantly during the study period, a patient recruited late might have better survival than a patient recruited at the beginning, which would bias the overall estimate. Users should audit their data for changes in treatment standards over the study duration.

.. _EPT:

3 - Precise events in time
--------------------------

The Kaplan-Meier estimator assumes that events (death, failure, etc.) occur at specific points in time.

The core idea
^^^^^^^^^^^^^

Although the method handles "tied" data (events occurring at the same time), it is **more effective when time is measured continuously**. Measuring time in very wide intervals (such as years) can mask the true shape of the survival curve and reduce the granularity of the estimator.

|
|
|
    | *Signé:*
    | **Eve Bodot**
    | *IMT Atlantique Student*