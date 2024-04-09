import streamlit as st
import pandas as pd
import pandas as pd
import re
import joblib
import pandas as pd
import streamlit as st

def run_python_file(filename):
    exec(open(filename).read())

def preprocess_return_earned(return_earned):
    if pd.isna(return_earned):
        return None
    elif "Negative" in return_earned:
        return 0
    elif "More than" in return_earned:
        return int(re.findall(r'\d+', return_earned)[0]) + 1
    else:
        numbers = re.findall(r'\d+', return_earned)
        return (int(numbers[0]) + int(numbers[1])) // 2

def extract_numerical_value(text):
    pattern = r"\d+"
    matches = re.findall(pattern, text)
    if len(matches) == 1:
        return int(matches[0])
    elif len(matches) == 2:
        return (int(matches[0]) + int(matches[1])) // 2
    elif len(matches) > 2:
        return sum(int(match) for match in matches) // len(matches)
    else:
        return None

# Load the training dataset
train_file_path = "datasets/dataset.xlsx"
train_data = pd.read_excel(train_file_path)
st.title("Recommendation System")
# Display the first three columns
st.write("First Three Columns:")
st.write(train_data.iloc[:3, ])

# Create input boxes for each column
inputs = {}
for col in train_data.columns[1:]:
    if col == 'Risk Level':
        continue
    else:
        unique_values = train_data[col].unique()
        inputs[col] = st.selectbox(col, unique_values)

# Execute button
if st.button('Execute'):
    # Process the inputs
    processed_data = []
    processed_data.append(len(train_data) + 1)
    for col in train_data.columns[1:]:
        if col == 'Risk Level':
            processed_data.append("Low")
        else:
            processed_data.append(inputs[col])
    
    # Add processed data to dataset.xlsx
    new_data = pd.DataFrame([processed_data], columns=train_data.columns)
    combined_data = pd.concat([train_data, new_data], ignore_index=True)
    df=combined_data
    
    df.to_excel("datasets/dataset.xlsx", index=False)
    run_python_file('preprocessing.py')
    
    # Load the model
    data = pd.read_excel('datasets/dataset1.xlsx')
    data = data.iloc[[-1]]
    model = joblib.load("model.pkl")
    # Remove the columns 'Risk Level_High', 'Risk Level_Low', 'Risk Level_Medium' from new_data
    columns_to_remove = ['Risk Level_High', 'Risk Level_Low', 'Risk Level_Medium']
    columns_to_remove = [col for col in columns_to_remove if col in data.columns]
    data.drop(columns_to_remove, axis=1, inplace=True)
    
    # Predict the risk value for new_data
    risk_prediction = model.predict(data)
    if risk_prediction[0][1]==1:
        st.subheader("Risk Level: High")
        st.write("You are at a high risk of losing your investment. Please consult a financial advisor.")
    elif risk_prediction[0][2]==1:
        st.subheader("Risk Level: Low")
        st.write("You are at a low risk of losing your investment. You can invest in this scheme.")
    else:
        st.subheader("Risk Level: Medium")
        st.write("You are at a medium risk of losing your investment. Think Once before you Invest !!!.")
    
    
    
    
    
