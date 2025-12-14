# Credit Risk Probability Model for Alternative Data

An **end-to-end implementation** for building, deploying, and automating a **credit risk scoring model** using alternative eCommerce transaction data.  
The system supports a **Buy-Now-Pay-Later (BNPL)** use case by estimating customer risk probability, generating credit scores, and recommending optimal loan terms.

---

## Table of Contents

- [Business Overview](#business-overview)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Credit Scoring Framework](#credit-scoring-framework)
- [System Architecture](#system-architecture)
- [API Usage](#api-usage)
- [MLOps & Model Management](#mlops--model-management)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

---

## Business Overview

Bati Bank is partnering with an eCommerce platform to launch a **Buy-Now-Pay-Later (BNPL)** service.  
The objective is to assess customer creditworthiness **without traditional credit bureau data**, relying instead on **behavioral transaction data**.

This project transforms raw transaction logs into:

- A **risk probability score**
- A **credit score**
- Recommended **loan amount and duration**

These outputs are used to automate credit decisions while remaining compliant with financial regulations.

---

## Credit Scoring Business Understanding

### Basel II and Model Interpretability

The **Basel II Capital Accord** emphasizes accurate measurement, monitoring, and control of credit risk.  
This directly impacts model design by requiring:

- **Interpretability** for audit and regulatory review
- **Clear documentation** of features, assumptions, and transformations
- **Reproducibility and governance** across the model lifecycle

As a result, the modeling pipeline prioritizes transparency alongside predictive performance.

---

### Proxy Default Variable

The dataset does not contain an explicit **default** or **loan repayment** label.  
To enable supervised learning, a **proxy default variable** is engineered using **RFM (Recency, Frequency, Monetary)** customer behavior.

- **High Risk (Bad):**

  - Long inactivity periods
  - Low transaction frequency
  - Declining or minimal monetary value

- **Low Risk (Good):**
  - Recent activity
  - Consistent transaction frequency
  - Stable or increasing spend

#### Business Risks of Proxy Labels

- Proxy labels may not perfectly reflect true repayment behavior
- Potential bias against new or low-activity customers
- Requires continuous monitoring and recalibration once real repayment data is available

---

### Model Complexity Trade-offs

| Aspect              | Interpretable Models (Logistic + WoE) | Complex Models (Gradient Boosting) |
| ------------------- | ------------------------------------- | ---------------------------------- |
| Explainability      | High                                  | Medium–Low                         |
| Regulatory Approval | Easier                                | Harder                             |
| Predictive Power    | Moderate                              | High                               |
| Governance Cost     | Low                                   | High                               |

This project evaluates both approaches and selects models based on **performance, explainability, and regulatory suitability**.

---

## Project Objectives

- Define a **proxy default variable** from behavioral data
- Engineer predictive features from raw transactions
- Train models to estimate **probability of default**
- Convert probabilities into **credit scores**
- Predict **optimal loan amount and duration**
- Deploy the model via an **API**
- Implement **CI/CD and MLOps** best practices

---

## Dataset Description

**Source:** Xente eCommerce Transaction Data (Kaggle)

### Key Fields

- `TransactionId`: Unique transaction identifier
- `AccountId`: Customer account identifier
- `CustomerId`: Unique customer identifier
- `ProductCategory`: Category of product purchased
- `ChannelId`: Web, Android, iOS, Pay Later, Checkout
- `Amount`: Signed transaction value
- `Value`: Absolute transaction amount
- `TransactionStartTime`: Timestamp of transaction
- `FraudResult`: Fraud indicator (1 = Fraud, 0 = Clean)

---

## Feature Engineering

Feature engineering is performed at the **customer level** and includes:

### Behavioral Features

- Recency (days since last transaction)
- Transaction frequency
- Total and average spend
- Transaction volatility

### Risk Signals

- Fraud history ratio
- Channel usage distribution
- Product category diversity

### Aggregations

- Time-based rolling windows
- Customer-level normalization

---

## Modeling Approach

1. **Train/Test Split** at customer level
2. Handle class imbalance using:
   - Class weighting
   - Threshold tuning
3. Models evaluated:
   - Logistic Regression (WoE)
   - Gradient Boosting
4. Metrics:
   - ROC-AUC
   - Precision-Recall
   - KS Statistic

---

## Credit Scoring Framework

Risk probabilities are transformed into a **credit score** using a monotonic scaling function:

```text
Credit Score = Base Score - (Factor × log(odds))
Higher score → Lower risk
Score bands mapped to approval rules
```

## System Architecture

Raw Data → Feature Engineering → Model Training → MLflow Registry
↓
FastAPI Inference API
↓
BNPL Decision Engine
