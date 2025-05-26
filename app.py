# app.py
import streamlit as st
import pandas as pd
import joblib

import io

# Load the model
model = joblib.load('model.pkl')

st.set_page_config(page_title="EV Hotspot Predictor", layout="centered")
st.title("🔌 EV Charging Hotspot Predictor")
st.markdown("Choose to enter single values manually **or** upload an Excel file with multiple entries.")

# Tabs
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
                predictions = model.predict(df[["incomeperperson", "internetuserate", "urbanrate"]])
                df["EV_Hotspot_Score"] = predictions
                st.success("✅ Predictions generated!")

                st.dataframe(df)

                # ...inside your code, after predictions are added to df
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
