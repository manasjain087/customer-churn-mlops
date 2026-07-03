import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"E:\customer-churn-mlops\data\Telco_customer_churn.csv")

print(df.head())

print(df.shape)  #.No of rows and columns in dataset.

print(df.columns) # check column name.

print(df.info()) #. 1.Numerical data, 2.Categorical data , 3.Missing values , 4.memory usage
                    
print(df.describe())  #. Statistical Summary

print(df.isnull().sum()) #.missing values check

print(df.duplicated().sum()) #.duplicate rows.

print(df["Churn Score"].value_counts())

#.Plot Churn Distribution.
df["Churn Score"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Score")
plt.ylabel("Count")
plt.show()

#. Numerical features
print(df.select_dtypes(include="number").columns)

#. Categorical features
print(df.select_dtypes(include="object").columns)

#. HISTORGRAMS
df.hist(figsize=(15,10))
plt.show()
