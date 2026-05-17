# ============================================================
# CUSTOMER CHURN PREDICTOR - FULL WORKING CODE
# ============================================================

# RUN:
# streamlit run app.py

# INSTALL:
# pip install streamlit pandas numpy scikit-learn matplotlib

# DATASET:
# Download:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#
# Put this CSV file in same folder:
# WA_Fn-UseC_-Telco-Customer-Churn.csv
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Customer Churn Predictor")

st.sidebar.info("""
This ML web application predicts whether a telecom
customer is likely to churn or stay.
""")

# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    # LOAD DATASET
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # ========================================================
    # DATA CLEANING
    # ========================================================

    # Remove customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove null values
    df.dropna(inplace=True)

    # Convert target column
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # ========================================================
    # ENCODE CATEGORICAL COLUMNS
    # ========================================================

    df_enc = df.copy()

    label_encoders = {}

    categorical_cols = df_enc.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        le = LabelEncoder()

        df_enc[col] = le.fit_transform(df_enc[col])

        label_encoders[col] = le

    # ========================================================
    # FEATURES & TARGET
    # ========================================================

    X = df_enc.drop("Churn", axis=1)

    y = df_enc["Churn"]

    feature_columns = X.columns

    # ========================================================
    # SPLIT DATA
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    feature_imp = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    return (
        model,
        acc,
        report,
        cm,
        feature_imp,
        df,
        feature_columns
    )

# ============================================================
# LOAD MODEL
# ============================================================

(
    model,
    acc,
    report,
    cm,
    feature_imp,
    df,
    feature_columns
) = train_model()

# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Predictor")

st.caption("""
Machine Learning Web Application for predicting
telecom customer churn.
""")

# ============================================================
# BUSINESS PROBLEM
# ============================================================

st.markdown("""
### 📌 Business Problem

Telecom companies lose customers because of churn.

This ML application predicts customers who are
likely to leave the company so businesses can
take preventive action.
""")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔮 Predict",
    "📈 Model Performance",
    "🔍 Dataset Explorer"
])

# ============================================================
# TAB 1 - PREDICT
# ============================================================

with tab1:

    st.subheader("Enter Customer Details")

    c1, c2, c3 = st.columns(3)

    # ========================================================
    # COLUMN 1
    # ========================================================

    with c1:

        tenure = st.slider(
            "Tenure (months)",
            1,
            72,
            12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            20.0,
            150.0,
            70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            0.0,
            10000.0,
            float(tenure * monthly_charges)
        )

    # ========================================================
    # COLUMN 2
    # ========================================================

    with c2:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # ========================================================
    # COLUMN 3
    # ========================================================

    with c3:

        senior_citizen = st.radio(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
            "Yes" if x else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    # ========================================================
    # ENCODING MAPS
    # ========================================================

    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    internet_map = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    }

    payment_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }

    yes_no_map = {
        "No": 0,
        "Yes": 1
    }

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_df = pd.DataFrame(
        columns=feature_columns
    )

    input_df.loc[0] = 0

    # Numeric values
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = total_charges
    input_df["SeniorCitizen"] = senior_citizen

    # Encoded values
    input_df["Contract"] = contract_map[contract]

    input_df["InternetService"] = internet_map[
        internet_service
    ]

    input_df["PaymentMethod"] = payment_map[
        payment_method
    ]

    input_df["Partner"] = yes_no_map[partner]

    input_df["Dependents"] = yes_no_map[
        dependents
    ]

    # Ensure exact order
    input_df = input_df[feature_columns]

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    ):

        pred = model.predict(input_df)[0]

        proba = model.predict_proba(input_df)[0]

        st.divider()

        # Progress Bar
        st.progress(int(proba[1] * 100))

        st.write(
            f"### Churn Risk Score: "
            f"{proba[1]*100:.1f}%"
        )

        # ====================================================
        # RESULTS
        # ====================================================

        if pred == 1:

            st.error(
                f"⚠️ HIGH CHURN RISK — "
                f"{proba[1]*100:.1f}% chance"
            )

            st.info("""
            💡 Recommended Actions:
            - Offer discount
            - Improve support
            - Convert to yearly plan
            """)

        else:

            st.success(
                f"✅ LOW CHURN RISK — "
                f"{proba[0]*100:.1f}% chance"
            )

            st.info("""
            💡 Recommended Actions:
            - Maintain engagement
            - Upsell services
            """)

        ca, cb = st.columns(2)

        ca.metric(
            "Churn Probability",
            f"{proba[1]*100:.1f}%"
        )

        cb.metric(
            "Retention Probability",
            f"{proba[0]*100:.1f}%"
        )

# ============================================================
# TAB 2 - MODEL PERFORMANCE
# ============================================================

with tab2:

    st.subheader("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{acc*100:.1f}%"
    )

    c2.metric(
        "Precision",
        f"{report['1']['precision']*100:.1f}%"
    )

    c3.metric(
        "Recall",
        f"{report['1']['recall']*100:.1f}%"
    )

    c4.metric(
        "F1 Score",
        f"{report['1']['f1-score']*100:.1f}%"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown("""
    ### 📈 Feature Importance
    """)

    st.bar_chart(feature_imp)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown("""
    ### 📋 Classification Report
    """)

    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown("""
    ### 🔲 Confusion Matrix
    """)

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

# ============================================================
# TAB 3 - DATASET EXPLORER
# ============================================================

with tab3:

    st.subheader("Dataset Explorer")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    # ========================================================
    # CHURN DISTRIBUTION
    # ========================================================

    with c1:

        st.markdown("""
        ### Customer Churn Distribution
        """)

        fig2, ax2 = plt.subplots()

        df["Churn"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax2
        )

        ax2.set_ylabel("")

        st.pyplot(fig2)

    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    with c2:

        st.markdown("""
        ### Summary Statistics
        """)

        st.dataframe(
            df.describe().round(2),
            use_container_width=True
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
### 🚀 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Random Forest Classifier
""")