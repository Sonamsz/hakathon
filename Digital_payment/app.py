import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Digital Payment Analytics",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("digital_payments.csv")

    return data


df = load_data()


# =========================================================
# CLEAN DATA
# =========================================================

df["Is_High_Value"] = (
    df["Is_High_Value"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)


# =========================================================
# LOAD / TRAIN MODEL
# =========================================================

@st.cache_resource
def train_models(data):

    data = data.copy()

    # Remove rows where target is missing
    data = data.dropna(
        subset=["Is_High_Value"]
    )

    # Separate features and target
    X = data.drop(
        columns=["Is_High_Value"]
    )

    y = data["Is_High_Value"].astype(int)

    # Transaction ID is not useful for prediction
    if "Transaction_ID" in X.columns:

        X = X.drop(
            columns=["Transaction_ID"]
        )

    # Detect categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Detect numerical columns
    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numerical",
                StandardScaler(),
                numerical_features
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Models
    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }

    trained_models = {}

    results = {}

    predictions = {}

    # Train every model
    for name, model in models.items():

        pipeline = Pipeline(
            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )
            ]
        )

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Predictions
        y_pred = pipeline.predict(
            X_test
        )

        # Probability
        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        # Store model
        trained_models[name] = pipeline

        # Store predictions
        predictions[name] = {

            "y_pred": y_pred,

            "y_prob": y_prob
        }

        # Store evaluation metrics
        results[name] = {

            "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "ROC AUC":
                roc_auc_score(
                    y_test,
                    y_prob
                )
        }

    return (
        trained_models,
        results,
        X_test,
        y_test,
        predictions
    )


(
    trained_models,
    results,
    X_test,
    y_test,
    predictions
) = train_models(df)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💳 Digital Payments")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Prediction",
        "Model Evaluation",
        "Data Explorer"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title(
        "💳 Digital Payment Analytics Dashboard"
    )

    st.write(
        """
        This application analyzes digital payment transactions
        and predicts whether a transaction is classified as
        high-value or normal-value.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    total_transactions = len(df)

    high_value = (
        df["Is_High_Value"] == 1
    ).sum()

    normal_value = (
        df["Is_High_Value"] == 0
    ).sum()

    average_amount = df[
        "Transaction_Amount"
    ].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "High-Value Transactions",
        high_value
    )

    col3.metric(
        "Normal Transactions",
        normal_value
    )

    col4.metric(
        "Average Amount",
        f"{average_amount:,.2f}"
    )

    st.divider()

    # -----------------------------------------------------
    # TRANSACTION AMOUNT DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "Transaction Amount Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        data=df,
        x="Transaction_Amount",
        hue="Is_High_Value",
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        "Transaction Amount"
    )

    ax.set_ylabel(
        "Number of Transactions"
    )

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # PAYMENT METHOD AND LOCATION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Transactions by Payment Method"
        )

        payment_counts = (
            df["Payment_Method"]
            .value_counts()
        )

        st.bar_chart(
            payment_counts
        )

    with col2:

        st.subheader(
            "Transactions by Location"
        )

        location_counts = (
            df["Location"]
            .value_counts()
        )

        st.bar_chart(
            location_counts
        )


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.title(
        "🔮 Digital Payment Prediction"
    )

    st.write(
        """
        Enter transaction information below to predict
        whether the transaction is high-value.
        """
    )

    # -----------------------------------------------------
    # GET OPTIONS FROM DATASET
    # -----------------------------------------------------

    locations = sorted(
        df["Location"]
        .dropna()
        .unique()
        .tolist()
    )

    payment_methods = sorted(
        df["Payment_Method"]
        .dropna()
        .unique()
        .tolist()
    )

    merchant_categories = sorted(
        df["Merchant_Category"]
        .dropna()
        .unique()
        .tolist()
    )

    device_types = sorted(
        df["Device_Type"]
        .dropna()
        .unique()
        .tolist()
    )

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        location = st.selectbox(
            "Location",
            locations
        )

        payment_method = st.selectbox(
            "Payment Method",
            payment_methods
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            merchant_categories
        )

    with col2:

        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=float(
                df["Transaction_Amount"].median()
            ),
            step=10.0
        )

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        device_type = st.selectbox(
            "Device Type",
            device_types
        )

        # IMPORTANT:
        # Dataset column is Customer_Tenure_Months

        customer_tenure = st.number_input(
            "Customer Tenure (Months)",
            min_value=0,
            max_value=600,
            value=36,
            step=1
        )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button(
        "🔍 Predict Transaction",
        use_container_width=True
    ):

        # Create input dataframe
        # Column names MUST match training data

        input_data = pd.DataFrame({

            "Age": [
                age
            ],

            "Location": [
                location
            ],

            "Payment_Method": [
                payment_method
            ],

            "Merchant_Category": [
                merchant_category
            ],

            "Transaction_Amount": [
                transaction_amount
            ],

            "Transaction_Hour": [
                transaction_hour
            ],

            "Device_Type": [
                device_type
            ],

            "Customer_Tenure_Months": [
                customer_tenure
            ]
        })

        # Use Random Forest
        model = trained_models[
            "Random Forest"
        ]

        # Prediction
        prediction = model.predict(
            input_data
        )[0]

        # Probability
        probability = model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.error(
                "🚨 HIGH-VALUE TRANSACTION"
            )

        else:

            st.success(
                "✅ NORMAL-VALUE TRANSACTION"
            )

        st.metric(
            "Probability of High-Value Transaction",
            f"{probability * 100:.2f}%"
        )

        st.subheader(
            "Entered Transaction"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )


# =========================================================
# MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":

    st.title(
        "📊 Model Evaluation"
    )

    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    ).T

    results_df = results_df.round(
        4
    )

    st.subheader(
        "Model Performance Comparison"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # F1 SCORE CHART
    # -----------------------------------------------------

    st.subheader(
        "F1 Score Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=results_df.index,
        y=results_df["F1 Score"],
        ax=ax
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "F1 Score"
    )

    ax.set_ylim(
        0,
        1.05
    )

    plt.xticks(
        rotation=30
    )

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # SELECT MODEL
    # -----------------------------------------------------

    selected_model = st.selectbox(
        "Select model for confusion matrix",
        list(trained_models.keys())
    )

    y_pred = predictions[
        selected_model
    ]["y_pred"]

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    st.subheader(
        f"Confusion Matrix - {selected_model}"
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # MODEL METRICS
    # -----------------------------------------------------

    st.subheader(
        "Model Metrics"
    )

    selected_results = results[
        selected_model
    ]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{selected_results['Accuracy']:.3f}"
    )

    col2.metric(
        "Precision",
        f"{selected_results['Precision']:.3f}"
    )

    col3.metric(
        "Recall",
        f"{selected_results['Recall']:.3f}"
    )

    col4.metric(
        "F1 Score",
        f"{selected_results['F1 Score']:.3f}"
    )

    col5.metric(
        "ROC-AUC",
        f"{selected_results['ROC AUC']:.3f}"
    )


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.title(
        "🔎 Data Explorer"
    )

    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.subheader(
        "Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(
            df.isnull().sum().sum()
        )
    )

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    st.subheader(
        "Numerical Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    st.subheader(
        "Missing Values"
    )

    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.subheader(
        "Correlation Heatmap"
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    )

    correlation = numeric_columns.corr()

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.info(
    """
    Digital Payment Analytics

    Machine Learning Project

    Models:
    • Logistic Regression
    • Decision Tree
    • Random Forest
    • Gradient Boosting
    """
)