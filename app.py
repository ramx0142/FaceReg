import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the trained model
# We use @st.cache_resource so the model loads once and stays in memory
@st.cache_resource
def load_model():
    try:
        # Load the model file 'DataAugmentation.h5'
        model = tf.keras.models.load_model('DataAugmentation.h5')
        return model

model = load_model()

# 2. Define Class Names
# Based on Page 1 of your PDF: {'NANDHINI': 0, 'PRAVEENA': 1, 'RAMPRASATH': 2}
class_names = ['NANDHINI', 'PRAVEENA', 'RAMPRASATH']

# 3. App Title and Layout
st.set_page_config(page_title="Face Recognition App", page_icon="👤")
st.title("👤 Face Recognition / Classification")
st.write("Upload an image to identify if it is Nandhini, Praveena, or Ramprasath.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing...")

    # 5. Preprocessing (Matching Page 4 of your PDF)
    try:
        # Resize to 224x224 (IMG_SIZE from Page 1)
        img = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Expand dims to make it (1, 224, 224, 3) - Page 4
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize by dividing by 255 - Page 4
        img_array = img_array / 255.0
        
        # 6. Make Prediction
        if model:
            prediction = model.predict(img_array)
            
            # Get the index with the highest probability
            ind = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Get the name from our list
            result_name = class_names[ind]
            
            # 7. Display Result
            st.markdown("---")
            st.subheader("Prediction Result:")
            st.success(f"**{result_name}**")
            st.info(f"Confidence: {confidence * 100:.2f}%")
                
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Footer
st.caption("Model: CNN with Data Augmentation")
