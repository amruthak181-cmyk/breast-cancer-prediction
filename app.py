import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0F0F0F;
    color: #FFFFFF; app
}

/* Main Title */
h1 {
    color: #FF3B3B;
    text-align: center;
    font-weight: bold;
}

/* Sub Headings */
h2, h3 {
    color: #FF4D4D;
}

/* Normal Text */
p, label, div {
    color: #EAEAEA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1A1A1A;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div {
    color: white;
}

/* Predict Button */
div.stButton > button {
    background: linear-gradient(90deg, #B30000, #FF1A1A);
    color: white;
    border-radius: 12px;
    border: none;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #FF1A1A, #B30000);
    color: white;
}

/* Input Boxes */
div[data-baseweb="input"] input {
    background-color: #222222;
    color: white;
    border: 1px solid #FF3B3B;
}

/* Number Input Labels */
label {
    color: white !important;
    font-weight: bold;
}

/* Horizontal Line */
hr {
    border: 1px solid #FF3B3B;
}

</style>
""", unsafe_allow_html=True)

st.title("🩺 Breast Cancer Prediction System")
st.markdown("""
<h2 style="
color:white;
border-bottom:2px solid #FF3B3B;
padding-bottom:8px;">
📊 Feature Importance
</h2>
""", unsafe_allow_html=True)

st.markdown("""
Predict whether a breast tumor is **Benign** or **Malignant**
using a **Random Forest Classifier**.
""")

# Load model
model = joblib.load("breast_cancer_model.pkl")

#Sidebar
st.sidebar.title("Breast Cancer Prediction")
st.sidebar.markdown("---")

st.sidebar.write("### About Project")

st.sidebar.write("""
- Algorithm : Random Forest
- Language : Python
- Framework : Streamlit
- Dataset : Breast Cancer Dataset
""")

st.sidebar.info("""
This application predicts whether a breast cancer is:

🟢 Benign

🔴 Malignant

using Random Forest Classifier.
""")


# Load dataset
df = pd.read_csv("breast-cancer.csv")

# Training feature names
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concave points_mean",
    "radius_worst",
    "area_worst"
]

st.subheader("Dataset Information")

st.write("📊 Number of Records:", df.shape[0])
st.write("📌 Number of Features:", len(features))
st.write("🤖 Algorithm Used: Random Forest Classifier")

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Records", df.shape[0])

with col2:
    st.metric("Features", len(features))

with col3:
    st.metric("Algorithm", "Random Forest")

st.subheader("Enter Patient Details")

# input_data = {}

# for feature in features:
#     input_data[feature] = st.number_input(feature, value=0.0)
col1, col2 = st.columns(2)

input_data = {}

with col1:
    input_data["radius_mean"] = st.slider(
        "Radius Mean",
        float(df["radius_mean"].min()),
        float(df["radius_mean"].max()),
        float(df["radius_mean"].mean())
    )

    input_data["texture_mean"] = st.slider(
        "Texture Mean",
        float(df["texture_mean"].min()),
        float(df["texture_mean"].max()),
        float(df["texture_mean"].mean())
    )

    input_data["perimeter_mean"] = st.slider(
        "Perimeter Mean",
        float(df["perimeter_mean"].min()),
        float(df["perimeter_mean"].max()),
        float(df["perimeter_mean"].mean())
    )

    input_data["area_mean"] = st.slider(
        "Area Mean",
        float(df["area_mean"].min()),
        float(df["area_mean"].max()),
        float(df["area_mean"].mean())
    )

with col2:
    input_data["concave points_mean"] = st.slider(
        "Concave Points Mean",
        float(df["concave points_mean"].min()),
        float(df["concave points_mean"].max()),
        float(df["concave points_mean"].mean())
    )

    input_data["radius_worst"] = st.slider(
        "Radius Worst",
        float(df["radius_worst"].min()),
        float(df["radius_worst"].max()),
        float(df["radius_worst"].mean())
    )

    input_data["area_worst"] = st.slider(
        "Area Worst",
        float(df["area_worst"].min()),
        float(df["area_worst"].max()),
        float(df["area_worst"].mean())
    )

if st.button("🔍 Predict"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    confidence = probability.max() * 100

    if prediction[0] == 1:
        st.error("🔴 Prediction: Malignant (Cancer Detected)")
        st.warning("""
⚠️ **Recommendation**

- The prediction indicates a higher likelihood of breast cancer.
- Please consult a qualified healthcare professional as soon as possible.
- Additional clinical tests (such as biopsy or imaging) may be required.
- This application is only an AI-assisted prediction tool and should not be used as a final medical diagnosis.
        """)

    else:
        st.success("🟢 Prediction: Benign (No Cancer Detected)")
        st.info("""
✅ **Recommendation**

- The prediction indicates that the tumor is likely benign.
- Continue regular health check-ups and breast screening as advised by your doctor.
- If you notice any unusual symptoms, seek medical advice promptly.
- This application is only an AI-assisted prediction tool and should not be used as a final medical diagnosis.
        """)

    st.write(f"### 🎯 Confidence: {confidence:.2f}%")
    st.progress(confidence / 100)

st.markdown("---")

st.caption("Developed using ❤️ Python, Streamlit & Random Forest Classifier")

st.markdown("---")
st.subheader("📊 Feature Importance")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(
    importance_df.set_index("Feature")
)

st.markdown("---")
st.subheader("📋 Dataset Preview")

st.dataframe(df.head())

