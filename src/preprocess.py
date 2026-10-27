"""Step 1 - Data preprocessing: clean, engineer features, create targets, save processed CSV."""
import json
from pathlib import Path

import pandas as pd

from features import add_features, add_targets, clean

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india_housing_prices.csv"
OUT = ROOT / "data" / "processed" / "housing_processed.csv"
APP = ROOT / "data" / "app" / "app_data.csv.gz"


def iqr_outliers(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def main():
    df = pd.read_csv(RAW)
    print(f"Raw shape: {df.shape}")
    print(f"Missing values: {int(df.isna().sum().sum())}  |  Duplicate rows: {int(df.duplicated().sum())}")

    df = clean(df)
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]:
        flag = iqr_outliers(df[col])
        print(f"IQR outliers in {col}: {int(flag.sum())}")
    # Cap extreme price-per-sqft values (1st-99th percentile) instead of dropping rows
    lo, hi = df["Price_per_SqFt"].quantile([0.01, 0.99])
    df["Price_per_SqFt"] = df["Price_per_SqFt"].clip(lo, hi)

    df = add_features(df)
    df = add_targets(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Processed shape: {df.shape} -> {OUT}")
    print(f"Good_Investment share: {df['Good_Investment'].mean():.2%}")

    # Small compressed copy with only the columns the Streamlit app needs (committed to GitHub for deployment)
    app_cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                "Price_per_SqFt", "Year_Built", "Future_Price_5Y", "Good_Investment"]
    APP.parent.mkdir(parents=True, exist_ok=True)
    df[app_cols].to_csv(APP, index=False, compression="gzip")
    print(f"App data -> {APP}")

    # Lookup tables the Streamlit app needs
    meta = {
        "city_by_state": df.groupby("State")["City"].unique().apply(sorted).apply(list).to_dict(),
        "city_median_ppsf": df.groupby("City")["Price_per_SqFt"].median().round(2).to_dict(),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
