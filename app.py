import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
model = joblib.load('model/fraud_detector.joblib')


# Function to preprocess input data
def preprocess_data(data):
    input_df = data.copy()
    preprocessed_data = pd.DataFrame()
    le = LabelEncoder()
    preprocessed_data['step'] = input_df['step']
    preprocessed_data['type'] = le.fit_transform(input_df['type'])
    preprocessed_data['amount'] = np.log1p(input_df['amount'])
    preprocessed_data['oldbalanceOrg'] = input_df['oldbalanceOrg']
    preprocessed_data['newbalanceOrig'] = input_df['newbalanceOrig']
    preprocessed_data['oldbalanceDest'] = input_df['oldbalanceDest']
    preprocessed_data['newbalanceDest'] = input_df['newbalanceDest']
    return preprocessed_data


# Function to make predictions using the loaded model
def predict(data):
    preprocessed_data = preprocess_data(data)
    predictions = model.predict(preprocessed_data)
    return predictions


# Create the Streamlit app
def main():
    st.title('Fraud Detection App')

    # Create input form
    st.header('Enter transaction details:')
    _type = st.selectbox(label='Select payment type', options=['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
    step = st.number_input('Step', min_value=1)
    amount = st.number_input('Amount', min_value=0.0)
    oldbalanceOrg = st.number_input('Old Balance')
    newbalanceOrig = st.number_input('New Balance Origin')
    oldbalanceDest = st.number_input('Old Balance Destination')
    newbalanceDest = st.number_input('New Balance Destination')

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({'type': [_type],
                               'step': [step],
                               'amount': [amount],
                               'oldbalanceOrg': [oldbalanceOrg],
                               'newbalanceOrig': [newbalanceOrig],
                               'oldbalanceDest': [oldbalanceDest],
                               'newbalanceDest': [newbalanceDest]
                               })

    # Make predictions on button click
    if st.button('Predict'):
        predictions = predict(input_data)
        st.write('Predicted Class:', predictions[0])


# Run the app
if __name__ == '__main__':
    main()
