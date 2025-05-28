# app.py
import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2

# 🌐 Page setup
st.set_page_config(page_title="EVCP Query Tool", layout="centered")
st.title("🔌 EVCP Query")
st.markdown("Webapp tool to predict EVCP hotspots and analyze nearby CPs")
st.markdown("Upload an Excel file with multiple entries **or** enter single values manually")

# 🔧 Sidebar: Model selection
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Machine Learning Model", ["Random Forest", "XGBoost"])
cp_search = st.sidebar.text_input("Enter CP Code (e.g. LJ850, IE967)")
max_distance_meters = st.sidebar.slider("Maximum distance to search (meters)", min_value=10, max_value=1000, value=200, step=10)

@st.cache_resource
def load_model(model_name):
    if model_name == "Random Forest":
        return joblib.load("rf_model.pkl")
    elif model_name == "XGBoost":
        return joblib.load("xgb_model.pkl")

model = load_model(model_choice)

# 📊 Feature Importance Plot
def plot_feature_importance(model, model_choice):
    if model_choice == "Random Forest":
        importance = model.feature_importances_
        features = ["incomeperperson", "internetuserate", "urbanrate"]
        df_imp = pd.DataFrame({"Feature": features, "Importance": importance})
    elif model_choice == "XGBoost":
        booster = model.get_booster()
        raw_score_dict = booster.get_score(importance_type='weight')
        feature_map = {"f0": "incomeperperson", "f1": "internetuserate", "f2": "urbanrate"}
        df_imp = pd.DataFrame({
            "Feature": [feature_map.get(f, f) for f in raw_score_dict.keys()],
            "Importance": list(raw_score_dict.values())
        })

    df_imp = df_imp.sort_values("Importance", ascending=True)
    fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                 title=f"💡{model_choice} Feature Importance", height=300)
    st.plotly_chart(fig, use_container_width=True)

# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in meters

# 📑 Tabs
tab1, tab2 = st.tabs(["📥 Upload Excel", "✍️ Manual Entry"])

with tab1:
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type="xlsx")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            required_columns = {"incomeperperson", "internetuserate", "urbanrate", "cp-code"}
            if not required_columns.issubset(df.columns):
                st.error("❌ Excel file must include columns: 'incomeperperson', 'internetuserate', 'urbanrate', 'cp-code'")
            else:
                # 💡 Prediction
                predictions = model.predict(df[["incomeperperson", "internetuserate", "urbanrate"]])
                df["EV_Hotspot_Score"] = predictions
                st.success("✅ Predictions generated!")
                st.info(f"Model used: **{model_choice}**")

                st.dataframe(df)

                # 📊 Show feature importance
                plot_feature_importance(model, model_choice)

                # Interactive scatter map (if coordinates are available) **hover_name=df.index
                if {"latitude", "longitude"}.issubset(df.columns):
                    try:
                        fig_map = px.scatter_map(
                            df,
                            lat="latitude",
                            lon="longitude",
                            color="EV_Hotspot_Score",
                            size="EV_Hotspot_Score",
                            color_continuous_scale="YlOrRd",
                            zoom=10,
                            hover_name="cp-code",
                            map_style="open-street-map",
                            title="📍 EV Hotspot Prediction Map",
                            width=2000
                        )
                        st.plotly_chart(fig_map)
                    except Exception as e:
                        st.warning(f"Map could not be rendered: {e}")

                # 💾 Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)

                st.download_button("📥 Download Results", output, file_name="ev_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # cp_search = st.text_input("Enter CP Code (e.g. LR711, EA473)")

                if cp_search:
                    if "cp-code" in df.columns and "latitude" in df.columns and "longitude" in df.columns:
                        if cp_search in df["cp-code"].values:
                            ref_row = df[df["cp-code"] == cp_search].iloc[0]
                            lat1, lon1 = ref_row["latitude"], ref_row["longitude"]

                            def compute_distance(row):
                                return haversine(lat1, lon1, row["latitude"], row["longitude"])

                            df["distance_meters"] = df.apply(compute_distance, axis=1)
                            nearby_df = df[(df["distance_meters"] <= max_distance_meters) & (df["cp-code"] != cp_search)]

                            if not nearby_df.empty:
                                st.success(f"✅ Found {len(nearby_df)} nearby charging points within {max_distance_meters} meters:")
                                st.dataframe(nearby_df[["cp-code", "latitude", "longitude", "distance_meters"]])
                                # Plot map
                                fig = px.scatter_mapbox(
                                    nearby_df,
                                    lat="latitude",
                                    lon="longitude",
                                    color="distance_meters",
                                    color_continuous_scale="YlOrRd",
                                    size='distance_meters',
                                    zoom=15,
                                    mapbox_style="open-street-map",
                                    hover_name="cp-code",
                                    title="Nearby Charging Points",
                                )

                                # Add reference point
                                fig.add_trace(
                                    px.scatter_mapbox(
                                        pd.DataFrame([ref_row]),
                                        lat="latitude",
                                        lon="longitude",
                                        hover_name="cp-code",
                                    ).data[0]
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"No nearby charging stations found within {max_distance_meters} meters.")
                        else:
                            st.error("❌ CP Code not found in uploaded file.")
                    else:
                        st.error("❌ Missing required columns: 'cp-code', 'latitude', or 'longitude'")

        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab2:
    income = st.number_input("Income per person (USD)", min_value=0.0, step=100.0)
    internet_use = st.slider("Internet usage rate (%)", 0.0, 100.0, 50.0)
    urban_rate = st.slider("Urbanization rate (%)", 0.0, 100.0, 50.0)

    if st.button("Predict Single Entry"):
        input_data = pd.DataFrame([[income, internet_use, urban_rate]],
                                  columns=["incomeperperson", "internetuserate", "urbanrate"])
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Predicted EV Hotspot Score: **{prediction}**")
        st.info(f"Model used: **{model_choice}**")

        # 📊 Show feature importance
        plot_feature_importance(model, model_choice)
