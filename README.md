# Inactivity-Based Emergency Detection (IBED) in Private Households Using Activity Information

## Overview
This repository contains the implementation of five different algorithms for detecting emergencies in private households based on activity data from various sensors. These algorithms have been re-implemented based on scientific papers. While every effort was made to adhere to the criteria outlined in the original works, 100% conformity cannot be guaranteed. It is essential to note that these implementations were created as part of the scientific work mentioned below.

This repository is supplementary material for the original research paper: **"Emergency Detection in Smart Homes Using an Inactivity Score for Uncertain Sensor Data" by Wilhelm & Wahl (2024)**.

> **Please Note:** If you use this code or any part of it in your work, you must cite the original paper by Wilhelm & Wahl (2024) as referenced below.

## Original Paper
For a comprehensive understanding of the methodologies and evaluations presented in this repository, please refer to the original paper:

- **Title:** Emergency Detection in Smart Homes Using an Inactivity Score for Uncertain Sensor Data
- **Authors:** Sebastian Wilhelm & Florian Wahl
- **Publication Date:** TBA

The algorithms by _Cuddihy et al. (2007)_, _Floeck & Litz (2009)_, _Floeck et al. (2011)_, and _Moshtaghi et al. (2015)_ are extensively discussed in the original sources listed below. We explicitly refer to these sources for detailed explanations and discussions.

## Algorithms Included
This repository includes the implementation of the following algorithms:

1. **Wilhelm & Wahl (2024):**
    - **File:** `IBED_WilhelmWahl_2024.py`
    - **Description:** The newly developed Inactivity Score-based algorithm that considers sensor uncertainties for emergency detection.

2. **Cuddihy et al. (2007):**
    - **File:** `IBED_CuddihyEtAl_2007.py`
    - **Description:** An early algorithm for detecting unusually long periods of inactivity based on historical data and fixed thresholds.
    - **Original Source:**
        - Cuddihy, P., Weisenberg, J., Graichen, C., & Ganesh, M. (2007). Algorithm to automatically detect abnormally long periods of inactivity in a home. *In Proceedings of the 1st ACM SIGMOBILE International Workshop on Systems and Networking Support for Healthcare and Assisted Living Environments (HealthNet '07)*. ACM. [DOI:10.1145/1248054.1248081](https://doi.org/10.1145/1248054.1248081)

3. **Floeck & Litz (2009):**
    - **File:** `IBED_FloeckLitz_2009.py`
    - **Description:** An approach using a polynomial function for threshold determination to analyze inactivity continuously.
    - **Original Source:**
        - Floeck, M., & Litz, L. (2009). Inactivity patterns and alarm generation in senior citizens’ houses. *In 2009 European Control Conference (ECC)*. IEEE. [DOI:10.23919/ecc.2009.7074979](https://doi.org/10.23919/ecc.2009.7074979)

4. **Floeck et al. (2011):**
    - **File:** `IBED_FloeckEtAl_2011.py`
    - **Description:** An algorithm using a finite-state machine to monitor room stay duration and trigger alarms based on historical data analysis.
    - **Original Source:**
        - Floeck, M., Litz, L., & Rodner, T. (2011). An Ambient Approach to Emergency Detection Based on Location Tracking. *In Toward Useful Services for Elderly and People with Disabilities* (pp. 296–302). Springer Berlin Heidelberg. [DOI:10.1007/978-3-642-21535-3_45](https://doi.org/10.1007/978-3-642-21535-3_45)

5. **Moshtaghi et al. (2015):**
    - **File:** `IBED_Moshtaghi_2015.py`
    - **Description:** A complex statistical model to detect abnormal periods of inactivity, considering historical inactivity data and continuous adaptation.
    - **Original Source:**
        - Moshtaghi, M., Zukerman, I., & Russell, R.A. (2015). Statistical models for unobtrusively detecting abnormal periods of inactivity in older adults. *User Model User-Adap Inter, 25*(3), 231–265. [DOI:10.1007/s11257-015-9162-6](https://doi.org/10.1007/s11257-015-9162-6)

6. **Utility and Supporting Classes:**
    - **Files:** `IBED.py`, `Objects.py`
    - **Description:** These files contain supporting classes and functions used across different algorithms to facilitate a uniform framework.

## Usage
Each algorithm file can be run independently to test its functionality. A unified framework for all algorithms is defined in the class `IBED.py`. After initialization, the algorithms are intended to be invoked continuously, i.e., in a tick-wise manner. For each time step, the method `def tick(self, events: List[Event])` should be called, which contains all new activity events for that time step (possibly an empty list). The return value will indicate the current score/threshold and whether an alarm is triggered.

## Contact
For questions, suggestions, or further information, please contact the authors of the original paper. We welcome feedback and, in particular, the independent verification of the re-implementations.

