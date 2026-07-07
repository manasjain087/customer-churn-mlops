import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv(r"E:\customer-churn-mlops\data\Telco_customer_churn.csv")
print(df.head())

# ..Fix Datatype
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
print(df.info())

# ..Remove Missing values.
df["Churn Reason"] = df["Churn Reason"].fillna(df["Churn Reason"].mode()[0])
print(df.isnull().sum())

df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())
print(df.isnull().sum())

df.drop("CustomerID", axis=1, inplace=True)


# Separate Features and Target.
drop_cols = [
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CLTV"
]

X = df.drop(columns=drop_cols)
y = df["Churn Value"]  # output feature (already 0/1)


# Select Numerical and Categorical Columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# Create Pipelines.
# Numerical pipeline:
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical pipeline:
cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine Pipelines.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

# Train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit and Transform (fit only on train, transform both)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# ===================== TRAINING ML MODELS =====================

# ---- Logistic Regression ----
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_lr, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_lr, pos_label=1))
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))


# ---- Decision Tree ----
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_dt, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_dt, pos_label=1))


# ---- Random Forest ----
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_rf, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_rf, pos_label=1))


# ---- XGBoost ----
# NOTE: reuse the SAME X_train/X_test/y_train/y_test built above.
# "Churn Value" is already numeric (0/1) - no need to re-encode it,
# and no need to rebuild X from df (that would reintroduce leakage
# columns like Churn Label / Churn Score / CLTV / geo columns).
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n--- XGBoost ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_xgb, pos_label=1))
print("F1 Score:", f1_score(y_test, y_pred_xgb, pos_label=1))
print("ROC AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))


# ===================== SAVE BEST MODEL =====================
# Save the model + preprocessor together so inference uses the same
# transformations that were fit during training.
joblib.dump(xgb, "xgb_churn_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("\nModel and preprocessor saved.")