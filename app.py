# app.py
import streamlit as st
import pandas as pd
import joblib
import io
import plotly.express as px

# 🔧 Sidebar: Model selection
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Machine Learning Model", ["Random Forest", "XGBoost"])

@st.cache_resource
def load_model(model_name):
    if model_name == "Random Forest":
        return joblib.load("rf_model.pkl")
    elif model_name == "XGBoost":
        return joblib.load("xgb_model.pkl")

model = load_model(model_choice)

# 🌐 Page setup
st.set_page_config(page_title="EV Hotspot Predictor", layout="centered")
st.title("🔌 EV Charging Hotspot Predictor")
st.markdown("Choose to enter single values manually **or** upload an Excel file with multiple entries.")

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
                 title=f"{model_choice} Feature Importance", height=300)
    st.plotly_chart(fig, use_container_width=True)

# 📑 Tabs
tab1, tab2 = st.tabs(["📥 Upload Excel", "✍️ Manual Entry"])

with tab1:
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type="xlsx")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            required_columns = {"incomeperperson", "internetuserate", "urbanrate"}
            if not required_columns.issubset(df.columns):
                st.error("❌ Excel file must include columns: 'incomeperperson', 'internetuserate', 'urbanrate'")
            else:
                # 💡 Prediction
                predictions = model.predict(df[["incomeperperson", "internetuserate", "urbanrate"]])
                df["EV_Hotspot_Score"] = predictions
                st.success("✅ Predictions generated!")
                st.info(f"Model used: **{model_choice}**")

                st.dataframe(df)

                # 📊 Show feature importance
                plot_feature_importance(model, model_choice)

                # 💾 Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)

                st.download_button("📥 Download Results", output, file_name="ev_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
