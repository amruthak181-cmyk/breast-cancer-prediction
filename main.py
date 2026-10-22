import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import joblib
import seaborn as sns
print(pd.__version__)


df=pd.read_csv("breast-cancer.csv")
print(df)
print(df.head())
print(df.shape)
df.info()
print(df.isnull().sum())
print(df.duplicated().sum())

x = df.drop("diagnosis", axis=1)  #X contains all input features used to train the model.
top_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concave points_mean",
    "radius_worst",
    "area_worst"
]

x = df[top_features]
y = df["diagnosis"]   #y is the target variable that the model predicts.
print(x.head())
print(y.head())

y = y.map({
    "M": 1,
    "B": 0
})
print(y.head())


x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print("X Train Shape:", x_train.shape)
print("X Test Shape:", x_test.shape)
print("Y Train Shape:", y_train.shape)
print("Y Test Shape:", y_test.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x_train,y_train)
print(model)
print("Model Training Completed Successfully")

y_pred = model.predict(x_test)
print(y_pred)
print(y_test.values)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Accuracy Percentage :",accuracy*100)


joblib.dump(model, "breast_cancer_model.pkl")
print("Model Saved Successfully")

importance = model.feature_importances_

feature_names = x.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

plt.figure(figsize=(6,4))
df["diagnosis"].value_counts().plot(kind="bar")
plt.title("Breast Cancer Diagnosis Count")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,6))
plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

plt.figure(figsize=(8,6))

sns.heatmap(
    df[top_features].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()


plt.figure(figsize=(6,4))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.show()


correct = (y_test == y_pred).sum()
wrong = (y_test != y_pred).sum()

plt.figure(figsize=(5,5))

plt.pie(
    [correct, wrong],
    labels=["Correct", "Wrong"],
    autopct="%1.1f%%"
)

plt.title("Prediction Accuracy")

plt.show()


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))