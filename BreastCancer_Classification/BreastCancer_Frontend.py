import pickle
import streamlit as st
import numpy as np
import base64  # Don't forget to import base64!
from PIL import Image

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
        encoded_image = base64.b64encode(data).decode()

    background_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)
# Call the function to set the background
set_background(r"C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\breastcancer.png")
# Dictionary to map model names to their file paths
model_files = {
    'K-Nearest Neighbors': r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\knn_breast_cancer.pkl',
    'Logistic Regression': r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\logistic_regression_breast_cancer.pkl',
    'Decision Tree': r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\decision_tree_breast_cancer.pkl',
    'Random Forest': r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\random_forest_breast_cancer.pkl',
    'SVC': r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\svc_breast_cancer.pkl'
}

# Load the saved scaler
scaler = pickle.load(open(r'C:\Users\91939\Desktop\AI&DS\GITHUB REPOSITORIES\BreastCancer_Classification\scaler.pkl', 'rb'))

st.header("Breast Cancer Predictor🩺")

st.write("""
This user-friendly web application leverages machine learning to assist in the early detection of breast cancer. By inputting specific characteristics about a patient's tumor, the tool provides a prediction regarding whether the tumor is benign or malignant. This information can be invaluable for healthcare professionals in making timely and informed treatment decisions.
""")

# Function to collect user input
def get_user_input():
    clump_thickness = st.number_input("Clump Thickness", min_value=1, max_value=10)
    uniformity_cell_size = st.number_input("Uniformity Cell Size", min_value=1, max_value=10)
    uniformity_cell_shape = st.number_input("Uniformity Cell Shape", min_value=1, max_value=10)
    marginal_adhesion = st.number_input("Marginal Adhesion", min_value=1, max_value=10)
    single_epithelial_cell_size = st.number_input("Single Epithelial Cell Size", min_value=1, max_value=10)
    bare_nuclei = st.number_input("Bare Nuclei", min_value=1, max_value=10)
    bland_chromatin = st.number_input("Bland Chromatin", min_value=1, max_value=10)
    normal_nucleoli = st.number_input("Normal Nucleoli", min_value=1, max_value=10)
    mitoses = st.number_input("Mitoses", min_value=1, max_value=10)
    
    # Return a list of input values
    return [clump_thickness, uniformity_cell_size, uniformity_cell_shape, marginal_adhesion, 
            single_epithelial_cell_size, bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]

# Function to scale the input and make predictions
def make_prediction(model, input_data):
    # Scale the input data using the scaler
    scaled_input = scaler.transform([input_data])
    
    # Make a prediction using the chosen model
    prediction = model.predict(scaled_input)
    return prediction

# Function to display the result
def display_result(prediction):
    if prediction[0] == 2:  # Assuming '2' is for benign
        st.success("Predicted Class: Benign (Non-cancerous)")
    else:  # Assuming '4' is for malignant
        st.error("Predicted Class: Malignant (Cancerous)")

# Sidebar to select classifier
st.sidebar.header("Choose Classifier")
classifier_name = st.sidebar.selectbox("Select the classifier for prediction", list(model_files.keys()))

# Load the chosen model
model = pickle.load(open(model_files[classifier_name], 'rb'))

# Classifier descriptions
classifier_descriptions = {
    'K-Nearest Neighbors': (
        "K-Nearest Neighbors (KNN) is a simple, non-parametric algorithm used for classification and regression. "
        "It works by finding the 'k' training samples closest to the input sample, based on a distance metric (usually Euclidean). "
        "The predicted class is the majority class among these 'k' neighbors."
    ),
    'Logistic Regression': (
        "Logistic Regression is a statistical method used for binary classification that models the probability of a class label. "
        "It estimates the parameters of a logistic function and outputs a probability between 0 and 1, which can be thresholded to predict the class."
    ),
    'Decision Tree': (
        "A Decision Tree is a flowchart-like tree structure used for classification and regression. "
        "It splits the data into branches based on feature values, making decisions at each node until a leaf node is reached, which provides the prediction."
    ),
    'Random Forest': (
        "Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (for classification). "
        "It reduces overfitting and improves accuracy by averaging the predictions from multiple trees."
    ),
    'SVC': (
        "Support Vector Classification (SVC) is a supervised learning model that finds the hyperplane that best separates the classes in feature space. "
        "It aims to maximize the margin between the closest points of different classes, known as support vectors."
    )
}

# Display classifier description in the sidebar
st.sidebar.subheader(f"About {classifier_name}")
st.sidebar.write(classifier_descriptions[classifier_name])

# Get user input
input_data = get_user_input()

# Add a button for prediction
if st.button("Predict"):
    # Make a prediction
    prediction = make_prediction(model, input_data)
    
    # Display the result
    display_result(prediction)

# Sidebar with additional model info
st.sidebar.markdown("[Learn more about the classifiers](https://scikit-learn.org/stable/supervised_learning.html)")
