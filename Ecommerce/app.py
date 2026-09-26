import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("ecommerce.csv")

    return data


df = load_data()


# =========================================================
# CLEAN DATA
# =========================================================

df["High_Value_Order"] = (
    df["High_Value_Order"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models(data):

    data = data.copy()

    # Remove rows without target
    data = data.dropna(
        subset=["High_Value_Order"]
    )

    # Separate features and target
    X = data.drop(
        columns=["High_Value_Order"]
    )

    y = data["High_Value_Order"].astype(int)


    # -----------------------------------------------------
    # REMOVE DATA LEAKAGE
    # -----------------------------------------------------

    # Order_Value directly determines High_Value_Order.
    # Therefore, do not use it as a model feature.

    if "Order_Value" in X.columns:

        X = X.drop(
            columns=["Order_Value"]
        )


    # Order ID is only an identifier

    if "Order_ID" in X.columns:

        X = X.drop(
            columns=["Order_ID"]
        )


    # -----------------------------------------------------
    # IDENTIFY COLUMNS
    # -----------------------------------------------------

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()


    numerical_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()


    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TRAIN ALL MODELS
    # -----------------------------------------------------

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


        # Train model

        pipeline.fit(
            X_train,
            y_train
        )


        # Predictions

        y_pred = pipeline.predict(
            X_test
        )


        # Probabilities

        y_probability = pipeline.predict_proba(
            X_test
        )[:, 1]


        # Store trained model

        trained_models[name] = pipeline


        # Store predictions

        predictions[name] = {

            "y_pred": y_pred,

            "y_prob": y_probability
        }


        # Store metrics

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
                    y_probability
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

st.sidebar.title(
    "🛒 E-Commerce Analytics"
)


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
        "🛒 E-Commerce Analytics Dashboard"
    )


    st.write(
        """
        This application analyzes e-commerce orders
        and predicts whether an order is classified
        as a high-value order.
        """
    )


    st.divider()


    # =====================================================
    # KPI CARDS
    # =====================================================

    total_orders = len(df)


    high_value_orders = (
        df["High_Value_Order"] == 1
    ).sum()


    normal_orders = (
        df["High_Value_Order"] == 0
    ).sum()


    average_order_value = df[
        "Order_Value"
    ].mean()


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Orders",
        total_orders
    )


    col2.metric(
        "High-Value Orders",
        high_value_orders
    )


    col3.metric(
        "Normal Orders",
        normal_orders
    )


    col4.metric(
        "Average Order Value",
        f"{average_order_value:,.2f}"
    )


    st.divider()


    # =====================================================
    # ORDER VALUE DISTRIBUTION
    # =====================================================

    st.subheader(
        "Order Value Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.histplot(
        data=df,
        x="Order_Value",
        hue="High_Value_Order",
        kde=True,
        ax=ax
    )


    ax.set_xlabel(
        "Order Value"
    )


    ax.set_ylabel(
        "Number of Orders"
    )


    st.pyplot(fig)


    plt.close(fig)


    # =====================================================
    # CATEGORY AND REGION
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Orders by Category"
        )


        category_counts = (
            df["Category"]
            .value_counts()
        )


        st.bar_chart(
            category_counts
        )


    with col2:

        st.subheader(
            "Orders by Customer Region"
        )


        region_counts = (
            df["Customer_Region"]
            .value_counts()
        )


        st.bar_chart(
            region_counts
        )


    # =====================================================
    # PAYMENT METHOD
    # =====================================================

    st.subheader(
        "Orders by Payment Method"
    )


    payment_counts = (
        df["Payment_Method"]
        .value_counts()
    )


    st.bar_chart(
        payment_counts
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.title(
        "🔮 High-Value Order Prediction"
    )


    st.write(
        """
        Enter customer and order information below.
        The machine-learning model will predict whether
        the order is likely to be classified as high-value.
        """
    )


    # =====================================================
    # GET DATASET OPTIONS
    # =====================================================

    categories = sorted(
        df["Category"]
        .dropna()
        .unique()
        .tolist()
    )


    regions = sorted(
        df["Customer_Region"]
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


    # =====================================================
    # INPUT FORM
    # =====================================================

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with col1:

        category = st.selectbox(
            "Product Category",
            categories
        )


        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )


        customer_region = st.selectbox(
            "Customer Region",
            regions
        )


        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=float(
                df["Product_Price"].median()
            ),
            step=100.0
        )


        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )


    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with col2:

        discount = st.number_input(
            "Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )


        delivery_days = st.number_input(
            "Delivery Days",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )


        payment_method = st.selectbox(
            "Payment Method",
            payment_methods
        )


        customer_rating = st.number_input(
            "Customer Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )


    st.divider()


    # =====================================================
    # CALCULATE ORDER VALUE
    # =====================================================

    estimated_order_value = (
        product_price
        * quantity
        * (1 - discount / 100)
    )


    st.metric(
        "Estimated Order Value",
        f"{estimated_order_value:,.2f}"
    )


    st.divider()


    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🔍 Predict Order",
        use_container_width=True
    ):


        # IMPORTANT:
        # These column names must match the training data.

        input_data = pd.DataFrame({

            "Category": [
                category
            ],

            "Customer_Age": [
                customer_age
            ],

            "Customer_Region": [
                customer_region
            ],

            "Product_Price": [
                product_price
            ],

            "Quantity": [
                quantity
            ],

            "Discount_Percent": [
                discount
            ],

            "Delivery_Days": [
                delivery_days
            ],

            "Payment_Method": [
                payment_method
            ],

            "Customer_Rating": [
                customer_rating
            ]
        })


        # Use Random Forest

        model = trained_models[
            "Random Forest"
        ]


        # Make prediction

        prediction = model.predict(
            input_data
        )[0]


        # Probability

        probability = model.predict_proba(
            input_data
        )[0][1]


        st.divider()


        # =================================================
        # SHOW RESULT
        # =================================================

        if prediction == 1:

            st.error(
                "🚨 HIGH-VALUE ORDER"
            )

        else:

            st.success(
                "✅ NORMAL-VALUE ORDER"
            )


        st.metric(
            "Probability of High-Value Order",
            f"{probability * 100:.2f}%"
        )


        st.subheader(
            "Entered Order Details"
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


    # =====================================================
    # RESULTS TABLE
    # =====================================================

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


    # =====================================================
    # F1 SCORE CHART
    # =====================================================

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


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

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


    # =====================================================
    # SELECTED MODEL METRICS
    # =====================================================

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
        "🔎 E-Commerce Data Explorer"
    )


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df,
        use_container_width=True
    )


    st.divider()


    # =====================================================
    # DATASET INFORMATION
    # =====================================================

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


    # =====================================================
    # STATISTICS
    # =====================================================

    st.subheader(
        "Numerical Statistics"
    )


    st.dataframe(
        df.describe(),
        use_container_width=True
    )


    # =====================================================
    # MISSING VALUES
    # =====================================================

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


    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

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
# SIDEBAR FOOTER
# =========================================================

st.sidebar.divider()


st.sidebar.info(
    """
    🛒 E-Commerce Analytics

    Machine Learning Project

    Models:

    • Logistic Regression
    • Decision Tree
    • Random Forest
    • Gradient Boosting
    """
)