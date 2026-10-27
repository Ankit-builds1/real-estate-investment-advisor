"""Step 4 - Streamlit app: Real Estate Investment Advisor."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from features import AMENITY_LIST, CATEGORICAL, NUMERIC, add_features  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_models():
    return (joblib.load(ROOT / "models" / "classifier.joblib"),
            joblib.load(ROOT / "models" / "regressor.joblib"),
            json.loads((ROOT / "models" / "metadata.json").read_text()),
            json.loads((ROOT / "models" / "model_summary.json").read_text()))


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "app" / "app_data.csv.gz")


clf, reg, meta, summary = load_models()
df = load_data()

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts whether a property is a Good Investment and estimates its price after 5 years.")
tab_pred, tab_explore, tab_model = st.tabs(["🔮 Predict", "📊 Market Insights", "🧠 Model Info"])

# ---------------------------------------------------------------- Predict
with tab_pred:
    s1, s2 = st.columns(2)  # outside the form so the city list updates when the state changes
    state = s1.selectbox("State", sorted(meta["city_by_state"]))
    city = s2.selectbox("City", meta["city_by_state"][state])
    with st.form("property"):
        c1, c2, c3 = st.columns(3)
        ptype = c1.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
        bhk = c1.slider("BHK", 1, 5, 3)
        size = c2.number_input("Size (sq ft)", 300, 10000, 1500, step=50)
        price = c2.number_input("Current Price (Lakhs)", 5.0, 1000.0, 120.0, step=5.0)
        year = c1.slider("Year Built", 1990, 2025, 2015)
        furnished = c1.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
        total_floors = c2.slider("Total Floors", 1, 30, 10)
        floor = c2.slider("Floor No", 0, 30, min(3, total_floors))
        schools = c3.slider("Nearby Schools", 1, 10, 5)
        hospitals = c3.slider("Nearby Hospitals", 1, 10, 5)
        c4, c5, c6 = st.columns(3)
        transport = c4.selectbox("Public Transport", ["Low", "Medium", "High"], index=1)
        facing = c4.selectbox("Facing", ["North", "South", "East", "West"])
        parking = c5.radio("Parking Space", ["Yes", "No"], horizontal=True)
        security = c5.radio("Security", ["Yes", "No"], horizontal=True)
        owner = c5.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
        status = c6.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        amenities = c6.multiselect("Amenities", AMENITY_LIST, default=["Gym", "Garden"])
        go = st.form_submit_button("Analyse Property", type="primary", use_container_width=True)

    if go:
        row = pd.DataFrame([{
            "State": state, "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size,
            "Price_in_Lakhs": price, "Price_per_SqFt": price * 1e5 / size, "Year_Built": year,
            "Furnished_Status": furnished, "Floor_No": min(floor, total_floors), "Total_Floors": total_floors,
            "Age_of_Property": 2025 - year, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Amenities": ", ".join(amenities), "Facing": facing, "Owner_Type": owner,
            "Availability_Status": status}])
        X = add_features(row)[NUMERIC + CATEGORICAL]
        prob = float(clf.predict_proba(X)[0, 1])
        future = float(reg.predict(X)[0])
        cagr = (future / price) ** (1 / 5) - 1

        m1, m2, m3, m4 = st.columns(4)
        if prob >= 0.5:
            m1.success(f"✅ Good Investment\n\nConfidence: {prob:.1%}")
        else:
            m1.error(f"❌ Not a Good Investment\n\nConfidence: {1 - prob:.1%}")
        m2.metric("Estimated Price after 5 Years", f"₹{future:,.1f} L", f"+₹{future - price:,.1f} L")
        m3.metric("Implied Annual Growth", f"{cagr:.2%}")
        city_med = meta["city_median_ppsf"].get(city)
        ppsf = price * 1e5 / size
        m4.metric("Price / sq ft vs city median", f"₹{ppsf:,.0f}",
                  f"{(ppsf / city_med - 1):+.1%}" if city_med else None, delta_color="inverse")

        years = list(range(0, 6))
        growth = pd.DataFrame({"Year": years, "Price (Lakhs)": [price * (1 + cagr) ** y for y in years]})
        st.plotly_chart(px.line(growth, x="Year", y="Price (Lakhs)", markers=True,
                                title="Projected price path"), use_container_width=True)

# ---------------------------------------------------------------- Explore
with tab_explore:
    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(df.State.unique()))
    bhks = f2.multiselect("BHK", sorted(df.BHK.unique()))
    pr = f3.slider("Price range (Lakhs)", float(df.Price_in_Lakhs.min()), float(df.Price_in_Lakhs.max()),
                   (float(df.Price_in_Lakhs.min()), float(df.Price_in_Lakhs.max())))
    sz = st.slider("Size range (sq ft)", int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max()),
                   (int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max())))
    d = df[df.Price_in_Lakhs.between(*pr) & df.Size_in_SqFt.between(*sz)]
    if states:
        d = d[d.State.isin(states)]
    if bhks:
        d = d[d.BHK.isin(bhks)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Properties", f"{len(d):,}")
    k2.metric("Avg Price", f"₹{d.Price_in_Lakhs.mean():,.1f} L" if len(d) else "-")
    k3.metric("Avg Price / sq ft", f"₹{d.Price_per_SqFt.mean():,.0f}" if len(d) else "-")
    k4.metric("Good Investment share", f"{d.Good_Investment.mean():.1%}" if len(d) else "-")

    if len(d):
        heat = d.pivot_table(index="City", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
        st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                                  title="Location heatmap: avg price per sq ft by city and type"),
                        use_container_width=True)
        a, b = st.columns(2)
        trend = d.groupby("Year_Built")["Price_in_Lakhs"].mean().reset_index()
        a.plotly_chart(px.line(trend, x="Year_Built", y="Price_in_Lakhs", title="Avg price by year built"),
                       use_container_width=True)
        gi = d.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(15).reset_index()
        b.plotly_chart(px.bar(gi, x="Good_Investment", y="City", orientation="h",
                              title="Top cities by Good Investment share"), use_container_width=True)
        st.dataframe(d.head(200)[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
                                  "Price_in_Lakhs", "Future_Price_5Y", "Good_Investment"]],
                     use_container_width=True)

# ---------------------------------------------------------------- Model info
with tab_model:
    st.subheader("Model comparison (logged in MLflow)")
    c = pd.DataFrame(summary["classification"]).T.drop(columns="confusion_matrix")
    r = pd.DataFrame(summary["regression"]).T
    a, b = st.columns(2)
    a.markdown(f"**Classification** - best: `{summary['best_classifier']}`")
    a.dataframe(c.style.format("{:.4f}"))
    b.markdown(f"**Regression** - best: `{summary['best_regressor']}`")
    b.dataframe(r.style.format("{:.4f}"))
    cm = summary["classification"][summary["best_classifier"]]["confusion_matrix"]
    a.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                             title="Confusion matrix (best classifier)"), use_container_width=True)
    for col, key, title in [(b, "clf_importance", "Feature importance - Good Investment"),
                            (st, "reg_importance", "Feature importance - Future Price")]:
        imp = pd.Series(summary[key]).sort_values()
        col.plotly_chart(px.bar(imp, orientation="h", title=title), use_container_width=True)
