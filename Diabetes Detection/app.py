import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the saved model
loaded_model = pickle.load(open('C:\\Users\\akhil\\OneDrive\\Desktop\\project\\Diabetes Detection\\regression.pkl', 'rb'))
scaler = StandardScaler()

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert input data to numpy array and reshape for prediction
    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
    # Scale the input data
    scaled_input = scaler.fit_transform(input_data_as_numpy_array)
    prediction = loaded_model.predict(scaled_input)
    return prediction[0]

def main():
    # Title of the web app
    st.title('Diabetes Prediction Web App')
    
    # Display header image
    # st.image(r'C:\Users\akhil\OneDrive\Desktop\project\Diabetes Detection\header.png')
    st.image('https://static.vecteezy.com/system/resources/previews/029/607/426/non_2x/diabetes-awareness-month-is-observed-every-year-in-november-november-is-diabetes-awareness-month-template-for-banner-greeting-card-poster-with-background-illustration-vector.jpg')

    
    # Input fields for user data
    pregnancies = st.text_input('Number of Pregnancies')
    glucose = st.text_input('Glucose Level')
    blood_pressure = st.text_input('Blood Pressure value')
    skin_thickness = st.text_input('Skin Thickness value')
    insulin = st.text_input('Insulin Level')
    bmi = st.text_input('BMI value')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value')
    age = st.text_input('Age of the Person')
    
    # Button to trigger prediction
    if st.button('Diabetes Test Result'):
        # Convert input values to float and make prediction
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        input_data = [float(val) for val in input_data]
        output = diabetes_prediction(input_data)
        
        # Display prediction result
        if output == 0:
            st.success('The person is not diabetic')
        else:
            st.error('The person is diabetic')
        
        # Print prediction result in terminal
        print("Prediction Result:", output)

if __name__ == '__main__':
    main()
