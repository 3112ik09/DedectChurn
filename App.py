import streamlit as st
import numpy as np  # for generating a fake CustomerID
import joblib
import pandas as pd
import numpy as np
import os

def preprocess_inference_data(input_data):
    # Create a DataFrame with the input data
    df = pd.DataFrame([input_data])

    df['Gender'].replace({"Male":1 , "Female":0} , inplace=True)
    
    df["Total_Spend"] = df['Subscription_Length_Months'] * df['Monthly_Bill']

    df['Data_Value'] = round(df['Total_Usage_GB'] / df['Monthly_Bill'])

    bins = [0, 35, 55, 71]
    labels = ['Young', 'Middle-aged', 'Senior']

    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    ag = df['Age_Group'].values
    # print(ag)
    # Drop Age column
    df = pd.concat([df, pd.get_dummies(df[['Location' , 'Age_Group']]).astype(int)], axis=1)
    df.drop("Age", axis=1, inplace=True)

    # Identify Loyal_Customer
    df['Loyal_Customer'] = df['Subscription_Length_Months'].apply(lambda x: 1 if x > 20 else 0)

    # Calculate Monthly_Usage_per_Age
    
    age_data ={
    'Young': 274.289397,
    'Middle-aged': 274.421263,
    'Senior': 274.481208
    }

    # Calculate Usage_Difference
    df['Usage_Difference'] = df['Total_Usage_GB'] -age_data[ag[-1]]
    # df.drop('Monthly_Usage_per_Age', axis=1, inplace=True)
    
    col_names = df.select_dtypes(include=['int64', 'float64']).columns
    

    expected_columns = ['Gender', 'Subscription_Length_Months', 'Monthly_Bill',
           'Total_Usage_GB', 'Total_Spend', 'Data_Value', 'Loyal_Customer',
           'Usage_Difference', 'Location_Chicago', 'Location_Houston',
           'Location_Los Angeles', 'Location_Miami', 'Location_New York',
           'Age_Group_Young', 'Age_Group_Middle-aged', 'Age_Group_Senior']

    # Check and add any missing columns with a default value of 0
    for column in expected_columns:
        if column not in df.columns:
            df[column] = 0
    
    df.drop(['CustomerID' ,'Age_Group','Location'] , axis=1 , inplace=True) 
    return df

# Example usage of the function with input data
input_data = {
    "CustomerID": "14dfhhg",
    "Age": 63,
    "Gender": "Male",
    "Location": "Los Angeles",
    "Subscription_Length_Months": 17,
    "Monthly_Bill": 73.36,
    "Total_Usage_GB": 236,
}

processed_input = preprocess_inference_data(input_data)
processed_input.head()

# model load
script_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the file path relative to the script's directory
model_file_path = os.path.join(script_dir, 'best_model.pkl')
try:
    best_model = joblib.load(model_file_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {str(e)}")



# Create a Streamlit app
st.title("Churn Prediction App")


st.header("Enter Customer Data")
customer_id = st.text_input("Customer ID", key="customer_id")
age = st.number_input("Age", min_value=1, max_value=100, key="age")
gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
location = st.selectbox("Location", ["Los Angeles", "New York", "Miami", "Chicago", "Houston"], key="location")
subscription_length = st.number_input("Subscription Length (Months)", min_value=1, key="subscription_length")
monthly_bill = st.number_input("Monthly Bill", min_value=1.0, key="monthly_bill")
total_usage_gb = st.number_input("Total Usage (GB)", min_value=1.0, key="total_usage_gb")

# Encode categorical features


# Make predictions
if st.button("Predict Churn"):
    input_data = {
        "CustomerID": customer_id,
        "Age": age,
        "Gender": gender,
        "Location": location,
        "Subscription_Length_Months": subscription_length,
        "Monthly_Bill": monthly_bill,
        "Total_Usage_GB": total_usage_gb,
    }

    # Reshape input data into a 2D array
    processed_input = preprocess_inference_data(input_data)
    expected_column_order = [
    'Gender', 'Subscription_Length_Months', 'Monthly_Bill',
    'Total_Usage_GB', 'Total_Spend', 'Data_Value', 'Loyal_Customer',
    'Usage_Difference', 'Location_Los Angeles', 'Location_Chicago',
    'Location_Houston', 'Location_Miami', 'Location_New York',
    'Age_Group_Young', 'Age_Group_Middle-aged', 'Age_Group_Senior'
]


    processed_input_reordered = {col: processed_input[col] for col in expected_column_order}
    # Make predictions
    input_data = np.array([processed_input_reordered[col] for col in expected_column_order]).reshape(1, -1)
    
    churn_prediction = best_model.predict(input_data)

   
    if churn_prediction[0] == 1:
        st.error("Churn Prediction: Customer is likely to churn.")
    else:
        st.success("Churn Prediction: Customer is likely to stay.")

