"""Step 3 - Train classification + regression models, log every run to MLflow, register the best ones."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features import CATEGORICAL, NUMERIC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "housing_processed.csv"
MODELS = ROOT / "models"
os.chdir(ROOT)  # keep mlruns/ and mlflow.db in the project root
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def feature_names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())


def run_experiment(name, models, X_tr, X_te, y_tr, y_te, task):
    mlflow.set_experiment(name)
    results = {}
    for model_name, model in models.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=model_name):
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            if task == "clf":
                proba = pipe.predict_proba(X_te)[:, 1]
                m = {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred),
                     "precision": precision_score(y_te, pred), "recall": recall_score(y_te, pred),
                     "roc_auc": roc_auc_score(y_te, proba)}
                cm = confusion_matrix(y_te, pred).tolist()
                mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
                m["confusion_matrix"] = cm
            else:
                m = {"rmse": root_mean_squared_error(y_te, pred), "mae": mean_absolute_error(y_te, pred),
                     "r2": r2_score(y_te, pred)}
            mlflow.log_params({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str))})
            mlflow.log_metrics({k: v for k, v in m.items() if k != "confusion_matrix"})
            mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3),
                                     serialization_format="cloudpickle")
            results[model_name] = {"pipe": pipe, "metrics": m, "run_id": mlflow.active_run().info.run_id}
            print(f"[{name}] {model_name}: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()} }")
    return results


def importance(pipe):
    model = pipe.named_steps["model"]
    vals = getattr(model, "feature_importances_", None)
    if vals is None:
        vals = abs(model.coef_).ravel()
    s = pd.Series(vals, index=feature_names(pipe)).sort_values(ascending=False).head(15)
    return {k.split("__", 1)[1]: round(float(v), 5) for k, v in s.items()}


def main():
    df = pd.read_csv(DATA)
    X = df[NUMERIC + CATEGORICAL]

    # ---- Classification: Good_Investment ----
    y = df["Good_Investment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = run_experiment("Good_Investment_Classification", {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=14, max_samples=0.3, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
    }, X_tr, X_te, y_tr, y_te, "clf")

    # ---- Regression: Future_Price_5Y ----
    y = df["Future_Price_5Y"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = run_experiment("Future_Price_Regression", {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=14, max_samples=0.3, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42),
    }, X_tr, X_te, y_tr, y_te, "reg")

    best_clf = max(clf, key=lambda k: clf[k]["metrics"]["f1"])
    best_reg = min(reg, key=lambda k: reg[k]["metrics"]["rmse"])
    print(f"\nBest classifier: {best_clf} | Best regressor: {best_reg}")

    for res, best, reg_name in [(clf, best_clf, "RealEstate_GoodInvestment_Classifier"),
                                (reg, best_reg, "RealEstate_FuturePrice_Regressor")]:
        mlflow.register_model(f"runs:/{res[best]['run_id']}/model", reg_name)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(clf[best_clf]["pipe"], MODELS / "classifier.joblib")
    joblib.dump(reg[best_reg]["pipe"], MODELS / "regressor.joblib")
    # XGBoost pipelines are always saved too: the Vercel web app runs them in the browser
    joblib.dump(clf["XGBoost"]["pipe"], MODELS / "classifier_xgb.joblib")
    joblib.dump(reg["XGBoost"]["pipe"], MODELS / "regressor_xgb.joblib")
    summary = {
        "classification": {k: v["metrics"] for k, v in clf.items()},
        "regression": {k: v["metrics"] for k, v in reg.items()},
        "best_classifier": best_clf, "best_regressor": best_reg,
        "clf_importance": importance(clf[best_clf]["pipe"]),
        "reg_importance": importance(reg[best_reg]["pipe"]),
    }
    (MODELS / "model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
