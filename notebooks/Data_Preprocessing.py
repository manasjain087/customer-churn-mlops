import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import joblib

df=pd.read_csv(r"E:\customer-churn-mlops\data\Telco_customer_churn.csv")
print(df.head())
print(df.info())

df["Total Charges"]=pd.to_numeric(df["Total Charges"],errors="coerce") # . fix datatypes

print(df.info())

print(df.isnull().sum())

df["Churn Reason"].fillna(df["Churn Reason"].mode()[0], inplace=True)
print(df.isnull().sum())

#.remove unnecessary column:
df.drop("CustomerID",axis=1,inplace=True)
print(df.info())

#. Seperate Features and Target.

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
y = df["Churn Value"]  #  output features.


#. Select Numerical and Categorical Column
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:",num_cols)
print("Categorical Columns:",cat_cols)

#.Create Pipelines.
#.Numerical pipeline:-
num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

#.Categorical pipeline:
cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

#.Combines Pipelines.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ]
)

#.Train-test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#. Fit and Transform
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


#..save preprocessor.

import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(preprocessor, "models/preprocessor.pkl")

print("✅ Saved at:", os.path.abspath("models/preprocessor.pkl"))


#.verify shapes..
print(X_train.shape)
print(X_test.shape)





