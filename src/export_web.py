"""Export trained XGBoost pipelines + market aggregates to JSON for the Vercel web app (web/)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import CATEGORICAL, NUMERIC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data"


def compact_tree(t, i=0):
    """Turn a raw XGBoost tree into nested arrays: [feat, thr, yes, no, missing] or a leaf value.
    Uses the raw model JSON so thresholds keep their exact float32 values (text dumps round them)."""
    left, right = t["left_children"][i], t["right_children"][i]
    if left == -1:
        return t["split_conditions"][i]  # leaf value
    return [t["split_indices"][i], t["split_conditions"][i], compact_tree(t, left), compact_tree(t, right),
            "y" if t["default_left"][i] else "n"]


def export_pipe(pipe):
    prep = pipe.named_steps["prep"]
    scaler = prep.named_transformers_["num"]
    ohe = prep.named_transformers_["cat"]
    booster = pipe.named_steps["model"].get_booster()
    cfg = json.loads(booster.save_config())
    base = float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
    raw = json.loads(booster.save_raw("json"))
    trees = [compact_tree(t) for t in raw["learner"]["gradient_booster"]["model"]["trees"]]
    return {
        "numeric": NUMERIC, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "categorical": CATEGORICAL, "categories": [c.tolist() for c in ohe.categories_],
        "base_score": base, "trees": trees,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The web app always uses the XGBoost pipelines (best / near-best and exportable to JS)
    for name in ["classifier", "regressor"]:
        pipe = joblib.load(ROOT / "models" / f"{name}_xgb.joblib")
        data = export_pipe(pipe)
        (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
        print(name, len(data["trees"]), "trees")

    df = pd.read_csv(ROOT / "data" / "processed" / "housing_processed.csv")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    summary = json.loads((ROOT / "models" / "model_summary.json").read_text())
    r = lambda s, n=1: {k: round(float(v), n) for k, v in s.items()}  # noqa: E731
    market = {
        "rows": len(df), "states": df.State.nunique(), "cities": df.City.nunique(),
        "good_share": round(float(df.Good_Investment.mean()), 4),
        "city_by_state": meta["city_by_state"], "city_median_ppsf": meta["city_median_ppsf"],
        "state_ppsf": r(df.groupby("State").Price_per_SqFt.mean().sort_values(ascending=False), 0),
        "city_good": r(df.groupby("City").Good_Investment.mean().sort_values(ascending=False), 4),
        "city_price": r(df.groupby("City").Price_in_Lakhs.mean().sort_values(ascending=False)),
        "transport_good": r(df.groupby("Public_Transport_Accessibility").Good_Investment.mean(), 4),
        "bhk_good": r(df.groupby("BHK").Good_Investment.mean(), 4),
        "status_good": r(df.groupby("Availability_Status").Good_Investment.mean(), 4),
        "type_ppsf": r(df.groupby("Property_Type").Price_per_SqFt.median(), 0),
        "growth_hist": np.histogram(df.Growth_Rate * 100, bins=20)[0].tolist(),
        "growth_edges": np.histogram(df.Growth_Rate * 100, bins=20)[1].round(2).tolist(),
        "metrics": {k: summary[k] for k in ["classification", "regression", "best_classifier", "best_regressor"]},
        "importance": {"clf": summary["clf_importance"], "reg": summary["reg_importance"]},
    }
    (OUT / "market.json").write_text(json.dumps(market, separators=(",", ":")))

    # Test cases so the JS implementation can be verified against Python
    X = df[NUMERIC + CATEGORICAL].sample(200, random_state=7)
    clf = joblib.load(ROOT / "models" / "classifier_xgb.joblib")
    reg = joblib.load(ROOT / "models" / "regressor_xgb.joblib")
    tests = {"rows": X.to_dict(orient="records"), "proba": clf.predict_proba(X)[:, 1].tolist(),
             "price": reg.predict(X).tolist()}
    (ROOT / "web" / "test_cases.json").write_text(json.dumps(tests))
    print("market + tests exported")


if __name__ == "__main__":
    main()
