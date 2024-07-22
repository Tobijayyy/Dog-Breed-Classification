import streamlit as st
import numpy as np
from PIL import Image # I am using PIL here instead of cv2 because of ease of use. cv2 requires converting the image from its file format to a byte-stream form.
import tensorflow as tf

# Function to load my model
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# Function to preprocess the image
def load_and_preprocess_image(image):
    img = Image.open(image)
    img = img.resize((64, 64))  # Resize the image to the required input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to predict what class the image falls into
def predict_image(image, model):
    processed_image = load_and_preprocess_image(image)
    confidence = np.max(model.predict(processed_image)) * 100
    prediction = model.predict(processed_image).argmax(axis=1)
    return prediction, confidence
    
# Streamlit app
st.title('Dog Breed Classification App')
st.write('Upload the image of a dog to predict its breed')

# A note for the users of the app
disclaimer = """
    <p style='font-size: 12px; color: gray; font-weight: bold;'>Please note that the model used was trained on the data from only 10 dog breeds. These are the Beagle, Boxer, Bulldog, Dachshund, German Shepherd, Golden Retriever, Labrador Retriever, Poodle, Rottweiler, Yorkshire Terrier breeds. </p>
    """
st.markdown(disclaimer, unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    # Simulate prediction
    prediction, confidence = predict_image(uploaded_file, model)

    # Display predictions
    breed_dict = {
        0: 'Beagle',
        1: 'Boxer',
        2: 'Bulldog',
        3: 'Dachshund',
        4: 'German Shepherd',
        5: 'Golden Retriever',
        6: 'Labrador Retriever',
        7: 'Poodle',
        8: 'Rottweiler',
        9: 'Yorkshire Terrier' 
        }

    for k, v in breed_dict.items():
        if prediction == k:
            st.write(f'There is a {confidence:.2f}% chance that the dog in the image is a {v}')
