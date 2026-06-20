import pandas as pd
df = pd.read_csv("C:/Users/anand/Documents/Customer_Churn.csv")
print(df.head)
"""

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())


print(df.describe())

print(df["Churn"].value_counts())


print(df.isnull().sum())

print(df.duplicated().sum())

print(df.dtypes)

print(df["TotalCharges"].head(10))
"""
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

#print(df.dtypes)
print(df.isnull().sum())
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

#print(df["TotalCharges"].isnull().sum())

print(df["Churn"].value_counts())

print(df["Churn"].value_counts(normalize=True) * 100)

print(df.groupby("Churn")["tenure"].mean())


print(df.groupby("Churn")["MonthlyCharges"].mean())



import matplotlib.pyplot as plt

plt.hist(df["tenure"])
plt.title("Tenure Distribution")
plt.xlabel("Tenure")
plt.ylabel("Count")
plt.show()



plt.hist(df["MonthlyCharges"])
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Count")
plt.show()

print(pd.crosstab(df["Contract"], df["Churn"]))





print(df.select_dtypes(include="object").columns)


df.drop("customerID", axis=1, inplace=True)


df["Churn"] = df["Churn"].map({
    "No":0,
    "Yes":1
})

print(df["Churn"].head())



print(df.head())

print(df["Churn"].unique())



for col in df.columns:
    print(col)
    print(df[col].unique())
    print()
    
    
df = pd.get_dummies(
    df,
    drop_first=True
)

print(df.shape)

print(df.head())



X = df.drop("Churn", axis=1)

y = df["Churn"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[:5])





from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(classification_report(
    y_test,
    y_pred
))


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

model = LogisticRegression(
    max_iter=10000
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(scores.mean())


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")
plt.show()


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))

import matplotlib.pyplot as plt

feature_importance.head(10).plot(
    x="Feature",
    y="Importance",
    kind="bar"
)

plt.title("Top 10 Important Features")
plt.show()
