🏡 House Price Prediction App
This project builds a SIMPLE LINEAR REGRESSION model to predict house prices based on property    size (in square feet). 

It includes both a training pipeline and an interactive Streamlit web app for real-time predictions.


📊 Dataset
- Source: House_data.csv
- Features used:
- Sq_feet: Independent variable (area of the house)
- Price: Dependent variable (target)


⚙️ Model Workflow
- Data preprocessing using pandas
- Train-test split with scikit-learn
- Model training using LinearRegression
- Model serialization with pickle
- Streamlit app for user input and prediction
