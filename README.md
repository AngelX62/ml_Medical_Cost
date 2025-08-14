# Medical Cost Prediction

Predict annual medical insurance charges from demographic and lifestyle variables using classic linear-model baselines and regularised regressors (Ridge, Lasso).

---

## Dataset

`Dataset/insurance.csv` (1,338 rows)

* **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
* **Target:** `charges` (USD, annual)

> Header example:
> `age,sex,bmi,children,smoker,region,charges`

---

## Methods

```mermaid
flowchart LR
    A[Load data] --> B[Sanity checks & EDA]
    B --> C[PolynomialFeatures (deg=2)]
    C --> D[StandardScaler + OneHotEncoder]
    D --> E{Model selection}
    E -->|GridSearchCV α grid| F[Ridge]
    E -->|GridSearchCV α grid| G[Lasso]
    F --> H[Evaluate on test set]
    G --> H
```

**Models compared**

* Polynomial OLS (degree‑2 baseline)
* Ridge Regression (RidgeCV)
* Lasso Regression (LassoCV)

**Metrics**

* **R²** (higher is better)
* **RMSE** (USD; lower is better)

---

## Results

|               Model |  Test R²  | Test RMSE (USD) | Notes             |
| ------------------: | :-------: | --------------: | ----------------- |
|      Polynomial OLS |   0.851   |           4,579 | Degree‑2 baseline |
| **RidgeCV (α = 1)** | **0.851** |       **4,575** | Stable, preferred |
|     LassoCV (α = 5) |   0.815   |               — | Over‑sparse       |

> Takeaway: Ridge matches baseline accuracy while shrinking coefficients, so it’s the safer choice for reuse.

---

## Quick start

**Python 3.9+ recommended**

```bash
git clone https://github.com/AngelX62/ml_Medical_Cost.git
cd ml_Medical_Cost
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

```txt
joblib==1.5.0
numpy==2.2.5
scikit-learn==1.6.1
scipy==1.15.3
threadpoolctl==3.6.0
```

## Reproduce the training/eval (minimal example)

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load
df = pd.read_csv("Dataset/insurance.csv")
X = df.drop(columns=["charges"])
y = df["charges"]

# Columns
cat = ["sex", "smoker", "region"]
num = ["age", "bmi", "children"]

# Preprocess + features
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("poly", PolynomialFeatures(degree=2, include_bias=False), num),
    ]
)

# Ridge with α search
pipe = Pipeline([("prep", pre), ("model", Ridge())])
grid = GridSearchCV(
    pipe, {"model__alpha": [0.1, 1, 10]},
    cv=5, scoring="neg_root_mean_squared_error"
)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
grid.fit(Xtr, ytr)

# Evaluate
pred = grid.predict(Xte)
print("Best alpha:", grid.best_params_["model__alpha"])
print("R2:", r2_score(yte, pred))
print("RMSE:", mean_squared_error(yte, pred, squared=False))

# Save model
joblib.dump(grid.best_estimator_, "best_model.joblib")
```

---

## Repository structure

```
ml_Medical_Cost/
├── Dataset/
│   └── insurance.csv
├── requirements.txt
└── README.md   ← you are here
```

> Consider adding:
>
> * `src/` for scripts
> * `notebooks/` for EDA
> * `models/` to store trained artifacts

---


## License

MIT (add a `LICENSE` file if not present).

## Acknowledgements

This repo adapts a well‑known public insurance dataset to compare linear regression techniques and illustrate the impact of regularisation on stability and interpretability.

