import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

pipeline = joblib.load("pipeline.pkl")
feature_names = pipeline.feature_names_in_

model = list(pipeline.named_steps.values())[-1]

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("💻 Laptop Price Prediction App")

st.markdown("""
Predict the **estimated market price of a laptop** based on hardware specifications.

This application uses a **Fine-Tuned Lasso Regression Model** trained on laptop hardware datasets.
""")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
**Model:** Lasso Regression  
**Cross Validation R²:** 0.865  

Important predictors discovered during training:

• RAM  
• SSD Storage  
• Pixel Density (PPI)  
• CPU Frequency
""")

# -------------------------------------------------
# INPUT LAYOUT
# -------------------------------------------------

col1, col2 = st.columns(2)

# LEFT COLUMN
with col1:

    st.subheader("General Specifications")

    company = st.selectbox(
        "Company",
        [
        "Apple","Asus","Chuwi","Dell","Fujitsu","Google","HP","Huawei",
        "LG","Lenovo","MSI","Mediacom","Microsoft","Razer","Samsung",
        "Toshiba","Vero","Xiaomi"
        ]
    )

    typename = st.selectbox(
        "Laptop Type",
        ["Notebook","Gaming","Ultrabook","Workstation","Netbook"]
    )

    os = st.selectbox(
        "Operating System",
        ["Windows","Other"]
    )

    weight = st.slider(
        "Weight (kg)",
        0.8, 5.0, 2.0
    )

# RIGHT COLUMN
with col2:

    st.subheader("Hardware Specifications")

    ram = st.slider(
        "RAM (GB)",
        2, 64, 8
    )

    cpu_freq = st.slider(
        "CPU Frequency (GHz)",
        1.0, 4.5, 2.5
    )

    cpu = st.selectbox(
        "CPU Type",
        [
        "A12-Series","A6-Series","A8-Series","A9-Series",
        "Atom","Celeron","Core M","Core i3","Core i5","Core i7",
        "E-Series","FX","Pentium","Ryzen","Xeon"
        ]
    )

    gpu = st.selectbox(
        "GPU Brand",
        ["Intel","Nvidia"]
    )

# -------------------------------------------------
# ADVANCED SETTINGS
# -------------------------------------------------

with st.expander("Advanced Display & Storage Settings"):

    col3, col4 = st.columns(2)

    with col3:

        touchscreen = st.selectbox("Touchscreen", [0,1])
        ips = st.selectbox("IPS Display", [0,1])

        ppi = st.slider(
            "Pixel Density (PPI)",
            90, 400, 150
        )

    with col4:

        ssd = st.slider(
            "SSD Storage (GB)",
            0, 2000, 256
        )

        hdd = st.slider(
            "HDD Storage (GB)",
            0, 2000, 0
        )

# -------------------------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------------------------

input_data = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

# Numerical features
input_data["CPU_Frequency (GHz)"] = cpu_freq
input_data["RAM (GB)"] = ram
input_data["Weight (kg)"] = weight
input_data["Touchscreen"] = touchscreen
input_data["Ips"] = ips
input_data["ppi"] = ppi
input_data["SSD"] = ssd
input_data["HDD"] = hdd

# -------------------------------------------------
# ONE HOT FEATURES
# -------------------------------------------------

company_col = f"Company_{company}"
if company_col in input_data.columns:
    input_data[company_col] = 1

type_col = f"TypeName_{typename}"
if type_col in input_data.columns:
    input_data[type_col] = 1

cpu_col = f"CPU_{cpu}"
if cpu_col in input_data.columns:
    input_data[cpu_col] = 1

gpu_col = f"GPU_Company_{gpu}"
if gpu_col in input_data.columns:
    input_data[gpu_col] = 1

input_data["CPU_Company_Intel"] = 1

if os == "Windows":
    input_data["OS_Windows"] = 1
else:
    input_data["OS_Other"] = 1

# -------------------------------------------------
# CONFIG SUMMARY
# -------------------------------------------------

st.subheader("Laptop Configuration Summary")

summary = {
    "Company": company,
    "Type": typename,
    "RAM": f"{ram} GB",
    "CPU": cpu,
    "CPU Frequency": f"{cpu_freq} GHz",
    "GPU": gpu,
    "SSD": f"{ssd} GB",
    "HDD": f"{hdd} GB",
    "PPI": ppi,
    "Weight": f"{weight} kg"
}

st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

st.markdown("---")

if st.button("💰 Predict Laptop Price"):

    prediction = pipeline.predict(input_data)

    price = np.exp(prediction)[0]

    st.success("Prediction Complete")

    colA, colB = st.columns(2)

    with colA:

        st.metric(
            label="Estimated Laptop Price",
            value=f"€{price:,.2f}"
        )

    with colB:

        # simple confidence range
        lower = price * 0.9
        upper = price * 1.1

        st.metric(
            label="Estimated Price Range",
            value=f"€{lower:,.0f} – €{upper:,.0f}"
        )

    st.caption("Predicted using trained regression model")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.markdown("---")
st.subheader("Model Feature Importance")

try:

    coefficients = model.coef_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": np.abs(coefficients)
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig, ax = plt.subplots()

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.invert_yaxis()

    st.pyplot(fig)

except:
    st.write("Feature importance unavailable.")

# -------------------------------------------------
# MODEL EXPLANATION
# -------------------------------------------------

with st.expander("How the Model Works"):

    st.markdown("""
### Machine Learning Model

This application uses **Lasso Regression**, a regularized linear model.

Lasso adds a penalty to reduce unnecessary coefficients, which helps:

• prevent overfitting  
• select important features  
• improve generalization  

### Key Features Affecting Price

Based on training results:

1. RAM capacity  
2. SSD storage size  
3. Display quality (PPI)  
4. CPU frequency  

Higher values for these features generally increase laptop price.
""")