# 🩺 Medical Cost Prediction

Predict annual medical insurance charges from demographic and lifestyle data. Compare regularised linear models (Ridge vs. Lasso), choose the most stable high-performing model, and persist it for reuse.

| Stage               | Test R² | Test RMSE (US $) | Notes                                                  |
|---------------------|:-------:|-----------------:|--------------------------------------------------------|
| Polynomial OLS      | **0.851** | 4,579           | Baseline with degree-2 polynomial features             |
| **RidgeCV (α = 1)** | **0.851** | **4,575**       | Matches accuracy, shrinks coefficients → safer choice  |
| LassoCV (α = 5)     | 0.815   |          | Over-sparse; useful predictors zeroed out              |

---

## 1. Dataset

* **Source:** 
* **Rows:** 1,338  
* **Target:** `charges` (annual medical insurance cost in US dollars)  
* **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`

---

## 2. Methodology

```mermaid
flowchart LR
    A[Load data] --> B[EDA & sanity checks]
    B --> C[PolynomialFeatures (deg=2)]
    C --> D[StandardScaler]
    D --> E{Model selection}
    E -->|GridSearchCV α grid| F[Ridge]
    E -->|GridSearchCV α grid| G[Lasso]
    F --> H[Evaluate & save best model]
    G --> H
