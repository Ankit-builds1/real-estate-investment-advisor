"""Step 2 - EDA: answers the 20 questions from the project brief, saving one chart per question."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "housing_processed.csv"
OUT = ROOT / "reports" / "eda"
sns.set_theme(style="whitegrid", palette="viridis")

findings = []


def save(n, title, note):
    plt.title(f"Q{n}. {title}", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / f"q{n:02d}.png", dpi=110)
    plt.close()
    findings.append(f"**Q{n}. {title}**  \n{note}\n")
    print(f"Q{n} done")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    sample = df.sample(20000, random_state=42)  # for scatter plots

    # ---------- 1-5: Price & size ----------
    plt.figure(figsize=(8, 4)); sns.histplot(df["Price_in_Lakhs"], bins=50)
    save(1, "Distribution of property prices",
         f"Prices range {df.Price_in_Lakhs.min():.0f}-{df.Price_in_Lakhs.max():.0f} lakhs, "
         f"mean {df.Price_in_Lakhs.mean():.0f}, median {df.Price_in_Lakhs.median():.0f}; roughly uniform.")

    plt.figure(figsize=(8, 4)); sns.histplot(df["Size_in_SqFt"], bins=50)
    save(2, "Distribution of property sizes",
         f"Sizes range {df.Size_in_SqFt.min()}-{df.Size_in_SqFt.max()} sq ft, median {df.Size_in_SqFt.median():.0f}.")

    plt.figure(figsize=(8, 4)); sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", showfliers=False)
    m = df.groupby("Property_Type")["Price_per_SqFt"].median().round(0).to_dict()
    save(3, "Price per sq ft by property type", f"Median price/sq ft by type: {m}.")

    plt.figure(figsize=(8, 4)); sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", s=5, alpha=.3)
    save(4, "Property size vs price",
         f"Correlation size-price = {df.Size_in_SqFt.corr(df.Price_in_Lakhs):.3f} (near zero, size alone does not drive price).")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df["Price_per_SqFt"], ax=ax[0]); sns.boxplot(y=df["Size_in_SqFt"], ax=ax[1])
    save(5, "Outliers in price per sq ft and size",
         "Price/sq ft is right-skewed with many high-end outliers (capped at the 1st-99th percentile in preprocessing); size has no outliers.")

    # ---------- 6-10: Location ----------
    s = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5)); sns.barplot(x=s.values, y=s.index)
    save(6, "Average price per sq ft by state", f"Highest: {s.index[0]} ({s.iloc[0]:.0f}); lowest: {s.index[-1]} ({s.iloc[-1]:.0f}).")

    c = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 9)); sns.barplot(x=c.values, y=c.index)
    save(7, "Average property price by city", f"Highest: {c.index[0]} ({c.iloc[0]:.1f} L); lowest: {c.index[-1]} ({c.iloc[-1]:.1f} L). Spread is narrow.")

    a = df.groupby("Locality")["Age_of_Property"].median()
    plt.figure(figsize=(8, 4)); sns.histplot(a, bins=30)
    save(8, "Median property age by locality (distribution over 500 localities)",
         f"Locality median ages range {a.min():.0f}-{a.max():.0f} years; typical locality median {a.median():.0f} years.")

    top_c = df["City"].value_counts().index[:10]
    ct = pd.crosstab(df[df.City.isin(top_c)]["City"], df["BHK"])
    ct.plot(kind="bar", stacked=True, figsize=(10, 5)); plt.ylabel("Count")
    save(9, "BHK distribution across top 10 cities", "BHK 1-5 is evenly spread in every city.")

    top_loc = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
    t = df[df.Locality.isin(top_loc)].groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean().reset_index()
    plt.figure(figsize=(10, 5)); sns.lineplot(data=t, x="Year_Built", y="Price_in_Lakhs", hue="Locality")
    save(10, "Price trend (by year built) for top 5 most expensive localities",
         f"Top localities: {', '.join(top_loc)}. Prices fluctuate by build year with no clear upward trend.")

    # ---------- 11-15: Correlations ----------
    num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools",
           "Nearby_Hospitals", "Amenity_Count", "Infrastructure_Score", "Future_Price_5Y", "Good_Investment"]
    plt.figure(figsize=(10, 8)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    save(11, "Correlation between numeric features",
         "Raw features are almost uncorrelated with each other; Future_Price_5Y is driven by current price (~0.99).")

    for n, col in [(12, "Nearby_Schools"), (13, "Nearby_Hospitals")]:
        g = df.groupby(col)["Price_per_SqFt"].mean()
        plt.figure(figsize=(8, 4)); sns.barplot(x=g.index, y=g.values); plt.ylabel("Avg price / sq ft")
        save(n, f"{col.replace('_', ' ')} vs price per sq ft",
             f"Avg price/sq ft varies only {g.min():.0f}-{g.max():.0f} across counts; weak relationship.")

    plt.figure(figsize=(8, 4)); sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs")
    save(14, "Price by furnished status", f"Median price: {df.groupby('Furnished_Status')['Price_in_Lakhs'].median().round(1).to_dict()}.")

    plt.figure(figsize=(8, 4)); sns.boxplot(data=df, x="Facing", y="Price_per_SqFt", showfliers=False)
    save(15, "Price per sq ft by facing direction", f"Median: {df.groupby('Facing')['Price_per_SqFt'].median().round(0).to_dict()}.")

    # ---------- 16-20: Investment / amenities / ownership ----------
    plt.figure(figsize=(6, 4)); sns.countplot(data=df, x="Owner_Type")
    save(16, "Properties by owner type", f"{df.Owner_Type.value_counts().to_dict()}.")

    plt.figure(figsize=(6, 4)); sns.countplot(data=df, x="Availability_Status")
    save(17, "Properties by availability status", f"{df.Availability_Status.value_counts().to_dict()}.")

    plt.figure(figsize=(6, 4)); sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs")
    save(18, "Parking space vs price", f"Median price: {df.groupby('Parking_Space')['Price_in_Lakhs'].median().round(1).to_dict()}; no meaningful difference.")

    g = df.groupby("Amenity_Count")["Price_per_SqFt"].mean()
    plt.figure(figsize=(8, 4)); sns.barplot(x=g.index, y=g.values); plt.xlabel("Number of amenities")
    save(19, "Amenities vs price per sq ft", f"Avg price/sq ft by amenity count: {g.round(0).to_dict()}.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    order = ["Low", "Medium", "High"]
    sns.barplot(data=df, x="Public_Transport_Accessibility", y="Price_per_SqFt", order=order, ax=ax[0])
    sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment", order=order, ax=ax[1])
    ax[1].set_ylabel("Share of Good Investments")
    gi = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(order)
    save(20, "Public transport vs price per sq ft and investment potential",
         f"Price/sq ft barely changes, but Good Investment share rises with access: {gi.round(3).to_dict()}.")

    (ROOT / "reports" / "EDA_findings.md").write_text("# EDA Findings\n\nCharts are in `reports/eda/`.\n\n" + "\n".join(findings), encoding="utf-8")


if __name__ == "__main__":
    main()
