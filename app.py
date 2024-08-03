import streamlit as st
import pickle
import pandas as pd
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("House Price Prediction")

# Input fields
st.write("Enter the input features:")

# Assume your model expects two features 'feature1' and 'feature2'
feature1 = st.text_input('Alley')
# feature2 = st.number_input('BedroomAbvGr')

# Predict button
if st.button('Predict'):
    # Create a DataFrame for the inputs
    input_data = pd.DataFrame([[feature1]], columns=['Alley'])

    # Make prediction
    prediction = model.predict(input_data)

    # Show prediction
    st.write(f"Prediction: {prediction[0]}")

if st.button('Show model parameters'):
    st.write(f"Model parameters: {model.get_params()}")
print(model)
