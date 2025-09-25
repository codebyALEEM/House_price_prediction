import streamlit as st
import numpy as np
import pickle

#Load the dataset
model =pickle.load(open(r'C:\Users\VICTUS\Desktop\mastering git\Practise git\House_price_prediction\LRM_house_price.pkl','rb'))

# Set the title of the streamlit app
st.title('House Price Prediction')

# Add a brief description 
st.write('This app predicts the Price based on size in square feet using a simple linear regression model.')


# Add input widget for user to enter years of experience
size_sq_feet = st.number_input('Enter area in sq feet :',min_value=0.0,value=1000.0,step=500.0)


# When the button is clicked , make predictions
if st.button("Predict House Price"):
    #Make a prediction using the trained model
    sq_feet_input = np.array([[size_sq_feet]]) # Convert the input to a 2D array for prediction
    prediction = model.predict(sq_feet_input)
    
    # Display the result
    st.success(f"The predicted house price for {size_sq_feet} is : ${prediction[0]:,.2f}")
    

#Display information about the model
st.write("The model was trained using a dataset of Price and area in sq feet.")

