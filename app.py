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

This application uses a **polynomial-features + linear regression model** trained on laptop hardware datasets.
""")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📊 Model Information")

st.sidebar.markdown("""
**Model:** Polynomial Regression (Linear Regression on Polynomial Features)  
**Cross Validation R²:** See training logs from `train_model.py`  

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
        "Apple","Asus","Acer","Chuwi","Dell","Fujitsu","Google","HP",
        "Huawei","LG","Lenovo","MSI","Mediacom","Microsoft","Razer",
        "Samsung","Toshiba","Vero","Xiaomi"
        ]
    )

    typename = st.selectbox(
        "Laptop Type",
        ["Notebook","Gaming","Ultrabook","2 in 1 Convertible","Netbook","Workstation"]
    )

    os = st.selectbox(
        "Operating System",
        ["Windows","Mac","Other"]
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

    gpu_company = st.selectbox(
        "GPU Brand",
        ["Intel","Nvidia","AMD"]
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

        cpu_tier = st.slider(
            "CPU Tier (1 = entry, 4 = high-end)",
            1, 4, 3
        )

        gpu_tier = st.slider(
            "GPU Tier (1 = entry, 3 = high-end)",
            1, 3, 2
        )

# -------------------------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------------------------

input_data = pd.DataFrame(
    np.zeros((1, len(feature_names))),
    columns=feature_names
)

# Set numeric features
if "CPU_Frequency (GHz)" in input_data.columns:
    input_data["CPU_Frequency (GHz)"] = cpu_freq
if "RAM (GB)" in input_data.columns:
    input_data["RAM (GB)"] = ram
if "Weight (kg)" in input_data.columns:
    input_data["Weight (kg)"] = weight
if "Touchscreen" in input_data.columns:
    input_data["Touchscreen"] = touchscreen
if "Ips" in input_data.columns:
    input_data["Ips"] = ips
if "ppi" in input_data.columns:
    input_data["ppi"] = ppi
if "SSD" in input_data.columns:
    input_data["SSD"] = ssd
if "HDD" in input_data.columns:
    input_data["HDD"] = hdd
if "CPU_Tier" in input_data.columns:
    input_data["CPU_Tier"] = cpu_tier
if "GPU_Tier" in input_data.columns:
    input_data["GPU_Tier"] = gpu_tier

# Set categorical/string features
if "Company" in input_data.columns:
    input_data["Company"] = company
if "TypeName" in input_data.columns:
    input_data["TypeName"] = typename
if "GPU_Company" in input_data.columns:
    input_data["GPU_Company"] = gpu_company
if "CPU" in input_data.columns:
    input_data["CPU"] = cpu
if "OS" in input_data.columns:
    input_data["OS"] = os

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
    "GPU Brand": gpu_company,
    "SSD": f"{ssd} GB",
    "HDD": f"{hdd} GB",
    "PPI": ppi,
    "Weight": f"{weight} kg",
    "CPU Tier": cpu_tier,
    "GPU Tier": gpu_tier,
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

This application uses a **polynomial regression model**, implemented as linear regression on polynomially
expanded numeric features together with one-hot encoded categorical features.

Numeric features (such as CPU frequency, RAM, storage sizes, and display PPI) are:

• scaled using `StandardScaler`  
• expanded using `PolynomialFeatures` with interaction terms  

Categorical features (such as company, laptop type, GPU brand, CPU family, and OS) are encoded via
`OneHotEncoder(handle_unknown="ignore")`.

The model is trained on the **log of the price** to stabilize variance, and predictions are exponentiated
in this app to return prices in Euro.
""")