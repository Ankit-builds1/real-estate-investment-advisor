# 🏠 Real Estate Investment Advisor
### Predicting Property Profitability & Future Value

A machine learning application that helps investors decide:
1. **Is this property a Good Investment?** (Classification)
2. **What will it be worth in 5 years?** (Regression)

**Live demo:** https://real-estate-investment-advisor.vercel.app

![Web app](docs/web_app.png)

Built with Python, Pandas, Scikit-learn, XGBoost, MLflow and Streamlit on the `india_housing_prices.csv` dataset (250,000 properties, 23 columns, 20 states, 42 cities).

---

## Project structure
```
Real_Estate_Investment_Advisor/
├── app.py                     # Streamlit app (Predict / Market Insights / Model Info)
├── requirements.txt
├── data/
│   ├── raw/india_housing_prices.csv
│   └── processed/housing_processed.csv   # created by preprocess.py
├── src/
│   ├── features.py            # feature engineering + target creation (shared with the app)
│   ├── preprocess.py          # Step 1: cleaning, outliers, features, targets
│   ├── eda.py                 # Step 2: the 20 EDA questions -> reports/eda/*.png
│   ├── train.py               # Step 3: 6 models, MLflow tracking + model registry
│   └── export_web.py          # exports XGBoost models to JSON for the web app
├── web/                       # Vercel site: index.html, styles.css, app.js, model.js, data/*.json
├── models/                    # best models (.joblib), metadata, metrics summary
├── reports/
│   ├── eda/q01.png ... q20.png
│   └── EDA_findings.md
├── mlflow.db                  # MLflow tracking + registry (SQLite)
└── mlruns/                    # MLflow artifacts
```

## How to run (Windows)
```bash
cd Real_Estate_Investment_Advisor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

cd src
python preprocess.py      # ~30 s
python eda.py             # ~1 min
python train.py           # ~10-20 min (6 models on 250k rows)
cd ..

streamlit run app.py      # opens http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlflow.db   # opens http://localhost:5000
```

---

## Methodology

### 1. Data preprocessing
- Checked missing values and duplicates (dataset had none; code still handles them with median/mode imputation and de-duplication).
- `Price_per_SqFt` in the raw file was in lakhs; recomputed in rupees (`Price × 100000 / Size`).
- Outlier detection with the IQR rule: ~19.7k high outliers in price per sq ft → capped at the 1st–99th percentile instead of dropping rows.
- Numeric features scaled with `StandardScaler`; categoricals one-hot encoded (inside an sklearn `Pipeline`, so the app uses the exact same transformations).

### 2. Engineered features
| Feature | Logic |
|---|---|
| `School_Density_Score` | Nearby_Schools / 10 |
| `Hospital_Density_Score` | Nearby_Hospitals / 10 |
| `Infrastructure_Score` | mean of school score, hospital score, transport level (0–1) |
| `Amenity_Count`, `Has_Gym` … `Has_Playground` | parsed from the Amenities text |
| `Parking_Flag`, `Security_Flag` | Yes/No → 1/0 |
| `Floor_Ratio` | Floor_No / Total_Floors |

> The brief mentions *crime rate*, which is not in the dataset, so `Infrastructure_Score` is used as the location-quality proxy.

### 3. Target creation
**Future_Price_5Y (regression)** – feature-based growth, as the brief suggests:
`Future = Current × (1 + r)^5`, where `r` starts at **8%** and is adjusted for infrastructure score (±2%), under-construction (+1%), age > 25 yrs (−1%), amenities (+0.2% each) and villas (+0.5%).

**Good_Investment (classification)** – multi-factor score, 1 point each for:
price/sq ft ≤ city median · BHK ≥ 3 · Ready to move · Infrastructure score ≥ 0.5 · growth rate ≥ 8%.
Score ≥ 3 → **Good Investment** (≈69% of properties).

### 4. Models (all tracked in MLflow)
| Task | Models compared |
|---|---|
| Classification | Logistic Regression, Random Forest, XGBoost |
| Regression | Linear Regression, Random Forest, XGBoost |

Best models are selected by F1 (classification) and RMSE (regression), registered in the MLflow Model Registry as
`RealEstate_GoodInvestment_Classifier` and `RealEstate_FuturePrice_Regressor`, and saved to `models/` for the app.

## Results (20% held-out test set, 50,000 properties)

**Classification - Good Investment**

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.8748 | 0.9107 | 0.9001 | 0.9216 | 0.9425 |
| RandomForest | 0.9838 | 0.9884 | 0.9837 | 0.9931 | 0.9985 |
| XGBoost ✅ | 0.9989 | 0.9992 | 0.9991 | 0.9993 | 1.0000 |

Best model confusion matrix: TN 15,319 · FP 31 · FN 23 · TP 34,627 (only 54 errors out of 50,000).

**Regression - Price after 5 years (₹ Lakhs)**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LinearRegression | 11.98 | 9.01 | 0.9970 |
| RandomForest | 3.20 | 2.29 | 0.9998 |
| XGBoost ✅ | 1.24 | 0.99 | 1.0000 |

XGBoost wins both tasks: an RMSE of about ₹1.2 L on prices averaging ~₹390 L.


## Key EDA findings
- The dataset is close to uniformly distributed: price, size, BHK, owner type and availability are evenly spread, and city averages differ by only a few percent.
- Size and price are almost uncorrelated (r ≈ 0), so price per sq ft varies widely.
- Schools, hospitals, furnishing, facing and parking barely move price per sq ft.
- Public transport access strongly raises the Good Investment share (48% Low → 72% Medium → 87% High).

Full answers to all 20 questions: [`reports/EDA_findings.md`](reports/EDA_findings.md).

## Streamlit app
![Streamlit app](docs/streamlit_app.png)

- **Predict** – form for property details → Good Investment verdict with confidence, 5-year price, implied annual growth, price/sq ft vs city median, projected price chart.
- **Market Insights** – filter by state, BHK, price and size; KPIs, city × property-type heatmap, price trend, top cities by investment share.
- **Model Info** – model comparison tables, confusion matrix, feature importance.

## Business use cases
- Investors assess long-term returns before buying.
- Buyers find high-return properties in developing areas.
- Real estate platforms automate investment analysis for listings and build trust with data-backed predictions.

## Limitations
- Both targets are **rule-based** (the dataset has no real future prices or investment labels), so the models learn these rules; scores are high for that reason. With real historical sale prices, the same pipeline can be retrained on true targets.
- The dataset appears synthetic (uniform distributions, `Locality_N` names), so insights describe this data, not the real Indian market.

## Live web app (Vercel)
`web/` is a static version of the advisor for public deployment. The XGBoost models are exported to JSON (`src/export_web.py`) and run directly in the browser (`web/model.js`), so no server is needed. Browser predictions were checked against Python on 200 test properties: all 200 verdicts match, and prices differ by under ₹0.001 L.

Deploy:
```bash
npm i -g vercel
cd web
vercel --prod
```
After retraining, run `python src/export_web.py` to refresh `web/data/`.
