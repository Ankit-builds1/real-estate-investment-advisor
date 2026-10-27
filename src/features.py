"""Feature engineering and target creation shared by training and the Streamlit app."""
import numpy as np
import pandas as pd

AMENITY_LIST = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}

# Growth assumptions for the 5-year price target
BASE_GROWTH = 0.08   # 8% per year, as suggested in the project brief
YEARS = 5

CATEGORICAL = ["State", "City", "Property_Type", "Furnished_Status",
               "Public_Transport_Accessibility", "Facing", "Owner_Type",
               "Availability_Status"]
NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
           "Age_of_Property", "Floor_No", "Total_Floors", "Nearby_Schools",
           "Nearby_Hospitals", "Parking_Flag", "Security_Flag",
           "Amenity_Count", "School_Density_Score", "Hospital_Density_Score",
           "Infrastructure_Score", "Floor_Ratio"] + [f"Has_{a}" for a in AMENITY_LIST]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fill missing values, fix types."""
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"]).copy()
    for c in df.select_dtypes(include="number").columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].fillna(df[c].mode()[0]).astype(str).str.strip()
    # Price_per_SqFt in the raw file is in lakhs per sq ft; recompute in rupees for readability
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(2)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features. Works on the full dataset or a single app row."""
    df = df.copy()
    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    amen = df["Amenities"].fillna("").astype(str)
    for a in AMENITY_LIST:
        df[f"Has_{a}"] = amen.str.contains(a).astype(int)
    df["Amenity_Count"] = df[[f"Has_{a}" for a in AMENITY_LIST]].sum(axis=1)
    df["School_Density_Score"] = df["Nearby_Schools"] / 10
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / 10
    transport = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP).fillna(2)
    # 0-1 score combining schools, hospitals and transport (proxy for "infrastructure score")
    df["Infrastructure_Score"] = ((df["School_Density_Score"] + df["Hospital_Density_Score"]
                                  + (transport - 1) / 2) / 3).round(3)
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].replace(0, 1)).clip(0, 1)
    return df


def growth_rate(df: pd.DataFrame) -> pd.Series:
    """Feature-based annual growth rate: base 8% adjusted by location quality and property state."""
    r = pd.Series(BASE_GROWTH, index=df.index)
    r += (df["Infrastructure_Score"] - 0.5) * 0.04          # better infra -> up to +2%
    r += np.where(df["Availability_Status"] == "Under_Construction", 0.01, 0)
    r -= np.where(df["Age_of_Property"] > 25, 0.01, 0)     # old buildings appreciate slower
    r += df["Amenity_Count"] * 0.002
    r += np.where(df["Property_Type"] == "Villa", 0.005, 0)
    return r


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create Future_Price_5Y (regression) and Good_Investment (classification)."""
    df = df.copy()
    df["Growth_Rate"] = growth_rate(df).round(4)
    df["Future_Price_5Y"] = (df["Price_in_Lakhs"] * (1 + df["Growth_Rate"]) ** YEARS).round(2)

    city_med_ppsf = df.groupby("City")["Price_per_SqFt"].transform("median")
    score = (
        (df["Price_per_SqFt"] <= city_med_ppsf).astype(int)          # cheaper than city median
        + (df["BHK"] >= 3).astype(int)
        + (df["Availability_Status"] == "Ready_to_Move").astype(int)
        + (df["Infrastructure_Score"] >= 0.5).astype(int)
        + (df["Growth_Rate"] >= BASE_GROWTH).astype(int)
    )
    df["Investment_Score"] = score
    df["Good_Investment"] = (score >= 3).astype(int)
    return df
