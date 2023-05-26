import base64
import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the saved model
model = joblib.load('model/fraud_detector.joblib')


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
    background-size: 180%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)




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
    _type = st.selectbox(label='Select payment type', options=['TRANSFER', 'CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT'])
    step = st.number_input('Step', min_value=1)
    amount = st.number_input('Amount', min_value=0.0)
    oldbalanceOrg = st.number_input('Old Balance Origin')
    newbalanceOrig = oldbalanceOrg - amount
    oldbalanceDest = st.number_input('Old Balance Destination')
    newbalanceDest = oldbalanceDest + amount

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
        is_fraud = ''
        if predictions[0] == 1:
            is_fraud = 'Fraud'
        else: 
            is_fraud = 'Normal'
        st.write(f'Predicted Class : {is_fraud}')


# Run the app
if __name__ == '__main__':
    main()
