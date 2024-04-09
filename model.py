import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score,recall_score

# Function to load data
def load_data(file_path):
    return pd.read_excel(file_path)

# Function to train RandomForestClassifier
def train_model(X_train, X_test, y_train):
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train)
    return rf_clf

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Function to perform feature selection and retrain the model
def select_features_and_retrain(model, X_train, X_test, y_train):
    selector = SelectFromModel(model, threshold='median')
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    model.fit(X_train_selected, y_train)
    return model, X_test_selected

# Main function to run Streamlit app
# Load data
file_path = "datasets/dataset1.xlsx"
data_encoded = load_data(file_path)

# Define features (X) and target (y)
X = data_encoded.drop(columns=['Risk Level_High', 'Risk Level_Low', 'Risk Level_Medium'])
y_columns = [ 'Risk Level_High', 'Risk Level_Low', 'Risk Level_Medium']
y = data_encoded[y_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize empty dictionaries to store models and evaluations for each target
models = {}
accuracies = {}
precisions = {}
f1s={}
recalls={}

# Iterate over each target variable
for column in y_columns:
    # Train the model for the current target variable
    model = train_model(X_train_scaled, X_test_scaled, y_train[column])
    models[column] = model
    
    # Evaluate the model's performance
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_scaled, y_test[column])
    accuracies[column] = accuracy
    precisions[column] = precision
    f1s [column] = f1
    recalls[column] = recall

    # Perform feature selection and retrain the model
    model, X_test_selected = select_features_and_retrain(model, X_train_scaled, X_test_scaled, y_train[column])
    
    # Evaluate the model's performance with selected features
    accuracy_selected, precision_selected,f1_score_selected,recall_score_selected = evaluate_model(model, X_test_selected, y_test[column])


    # Display results
    st.write(f"### Evaluation for {column}")
    st.write(f"- Accuracy: {accuracy}")
    st.write(f"- Precision: {precision}")
    st.write(f"- F1 Score: {f1}")
    st.write(f"- Recall Score: {recall}")
    st.write(f"- Accuracy with selected features: {accuracy_selected}")
    st.write(f"- Precision with selected features: {precision_selected}")
    st.write(f"- F1 Score with selected features: {f1_score_selected}")
    st.write(f"- Recall Score with selected features: {recall_score_selected}")