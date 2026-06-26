import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Student Dropout Prediction System")

df = pd.read_csv("student_dropout_dataset_v3.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop("Dropout", axis=1)
y = df["Dropout"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

st.success("Model trained successfully!")

st.write("Model Accuracy:",
         model.score(X_test, y_test))

if st.button("Predict Sample Student"):
    prediction = model.predict(X_test.iloc[[0]])

    if prediction[0] == "Yes":
        st.error("Student is likely to Drop Out")
    else:
        st.success("Student is likely to Continue Studies")