# 📐 Probabilistic Default Modelling for Loan Portfolios
## Formula & Data Reference Sheet

---

## 🏦 Project Overview

This project applies **Statistics, Probability, and Machine Learning** to predict whether a bank loan will default.
By analysing customer data (income, credit score, loan amount, employment, repayment history), we estimate:

> **How likely is it that this customer will NOT repay their loan?**

---

## 📊 Section 1 — Core Probability

### 1.1 Probability of Default (PD)

$$
P(\text{Default}) = \frac{\text{Number of Defaults}}{\text{Total Loans}}
$$

**Example:** 20 defaults out of 100 loans → `PD = 0.20` (20%)

---

### 1.2 Complement Rule

$$
P(\text{No Default}) = 1 - P(\text{Default})
$$

---

### 1.3 Conditional Probability

$$
P(\text{Default} \mid \text{Low Credit Score}) = \frac{P(\text{Default} \cap \text{Low Credit})}{P(\text{Low Credit})}
$$

Used to find default probability for specific customer segments.

---

## 📈 Section 2 — Descriptive Statistics

### 2.1 Mean (Average)

$$
\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}
$$

Used for: Average loan amount, average credit score, average income.

---

### 2.2 Variance

$$
\sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}
$$

Measures spread/uncertainty in the portfolio.

---

### 2.3 Standard Deviation

$$
\sigma = \sqrt{\sigma^2} = \sqrt{\frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}}
$$

Used to measure portfolio risk volatility.

---

## 💰 Section 3 — Expected Loss Model (Most Important)

$$
\boxed{EL = PD \times LGD \times EAD}
$$

| Symbol | Meaning | Description |
|---|---|---|
| **PD** | Probability of Default | Chance customer fails to repay |
| **LGD** | Loss Given Default | % of loan lost if customer defaults (1 − Recovery Rate) |
| **EAD** | Exposure at Default | Total outstanding loan balance at time of default |
| **EL** | Expected Loss | Expected monetary loss per loan |

**Example:**
- PD = 0.20, LGD = 0.60, EAD = ₹5,00,000
- EL = 0.20 × 0.60 × 5,00,000 = **₹60,000**

---

### 3.1 Loss Given Default

$$
LGD = 1 - \text{Recovery Rate}
$$

---

### 3.2 Portfolio Expected Loss

$$
EL_{\text{Portfolio}} = \sum_{i=1}^{N} PD_i \times LGD_i \times EAD_i
$$

Sum of expected losses across all N loans.

---

## 🔗 Section 4 — Correlation

$$
r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

Where:
$$
\text{Cov}(X, Y) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{n}
$$

| r value | Interpretation |
|---|---|
| r ≈ +1 | Strong positive correlation |
| r ≈ −1 | Strong negative correlation |
| r ≈ 0 | No correlation |

**Used for:** Finding relationship between credit score & default rate, income & loan size, etc.

---

## 🔔 Section 5 — Normal Distribution

$$
X \sim \mathcal{N}(\mu, \sigma^2)
$$

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**Used in:** CreditMetrics model, credit score distribution, loan amount distribution.

- **μ (mu)** = Mean (centre of distribution)
- **σ (sigma)** = Standard deviation (width/spread)

---

## 🤖 Section 6 — Logistic Regression & Sigmoid Function

### 6.1 Logistic Regression Model

The log-odds (logit) of default:

$$
\log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

Where:
- **p** = probability of default
- **βᵢ** = model coefficients
- **xᵢ** = feature values (credit score, income, loan amount, etc.)

---

### 6.2 Sigmoid / Logistic Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where `z` is the linear combination: `z = β₀ + β₁x₁ + β₂x₂ + …`

- Output range: **(0, 1)** → directly interpreted as probability
- If σ(z) > 0.5 → **Predict Default**
- If σ(z) ≤ 0.5 → **Predict No Default**

---

### 6.3 Decision Boundary

$$
\hat{y} = \begin{cases} 1 & \text{if } P(\text{Default}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

---

## 📉 Section 7 — Model Evaluation Metrics

### 7.1 Confusion Matrix

|  | Predicted Default | Predicted No Default |
|---|---|---|
| **Actual Default** | TP (True Positive) | FN (False Negative) |
| **Actual No Default** | FP (False Positive) | TN (True Negative) |

---

### 7.2 Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

---

### 7.3 Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

### 7.4 Recall (Sensitivity)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

---

### 7.5 F1 Score

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

### 7.6 ROC-AUC Score

Area under the Receiver Operating Characteristic curve.
- AUC = 1.0 → Perfect model
- AUC = 0.5 → Random guessing

---

## 📂 Section 8 — Dataset Features Used

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Customer age in years |
| `income` | Numerical | Annual income (₹) |
| `loan_amount` | Numerical | Loan amount requested (₹) |
| `credit_score` | Numerical | CIBIL/credit score (300–850) |
| `employment_years` | Numerical | Years in current employment |
| `debt_to_income` | Numerical | Debt-to-income ratio |
| `num_credit_lines` | Numerical | Number of active credit accounts |
| `interest_rate` | Numerical | Loan interest rate (%) |
| `loan_term` | Numerical | Loan term in months |
| `default` | Binary (0/1) | Target variable — 1=Default, 0=No Default |

---

## 🔄 Section 9 — Data Generation (Synthetic)

Since real bank data is private, we simulate realistic loan data using:

$$
\text{Credit Score} \sim \mathcal{N}(650, 80^2) \quad \text{clipped to } [300, 850]
$$

$$
\text{Income} \sim \mathcal{N}(60000, 20000^2) \quad \text{clipped to } [20000, 150000]
$$

$$
\text{Default Probability} = \sigma(\beta_0 + \beta_1 \cdot \text{CreditScore} + \beta_2 \cdot \text{Income} + \cdots)
$$

---

## 📝 Section 10 — Key Takeaways for Viva

> **"This project uses probability to estimate PD, descriptive statistics (mean, variance) to understand the portfolio, Expected Loss (PD × LGD × EAD) to quantify monetary risk, logistic regression and the sigmoid function to classify customers as high/low risk, correlation analysis to identify feature relationships, and the normal distribution to model credit score behaviour."**

### Most Important Formulas to Remember:
1. `P(Default) = Defaults / Total Loans`
2. `EL = PD × LGD × EAD`
3. `σ(z) = 1 / (1 + e⁻ᶻ)` ← Sigmoid
4. `σ² = Σ(x − x̄)² / n` ← Variance
5. `r = Cov(X,Y) / (σₓ · σᵧ)` ← Correlation

---

*Document generated for Probabilistic Default Modelling for Loan Portfolios — Statistics Project*
