import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the trained model with error handling
st.sidebar.header("🔍 Model Status")
try:
    model = load_model('plant_disease_model.h5')
    st.sidebar.success("✅ Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# Define plant disease classes
CLASS_NAMES = [
    "Tomato-Bacterial_spot", "Tomato-Early_blight", "Tomato-Late_blight",
    "Potato-Early_blight", "Potato-Late_blight", "Potato-Healthy",
    "Corn-Common_rust", "Corn-Gray_leaf_spot", "Corn-Leaf_blight",
    "Apple-Apple_scab", "Apple-Black_rot", "Apple-Cedar_rust", "Apple-Healthy"
]

# Disease prevention tips
DISEASE_PREVENTION = {
    "bacterial_spot": "🛑 Avoid overhead watering, use copper-based fungicides, and remove infected leaves.",
    "early_blight": "🌱 Rotate crops, remove infected leaves, and apply fungicides containing chlorothalonil.",
    "late_blight": "🌦️ Avoid wet leaves, use resistant varieties, and apply fungicides containing mancozeb.",
    "common_rust": "🍂 Remove infected plants, increase air circulation, and use fungicides with sulfur.",
    "gray_leaf_spot": "🌞 Ensure proper spacing, avoid overhead irrigation, and use resistant plant varieties.",
    "leaf_blight": "🌿 Apply nitrogen-based fertilizers, prune infected leaves, and improve soil drainage.",
    "apple_scab": "🍏 Use resistant apple varieties, remove fallen leaves, and apply sulfur-based fungicides.",
    "black_rot": "🌳 Avoid overhead irrigation, prune infected branches, and ensure proper plant ventilation.",
    "cedar_rust": "🍂 Remove infected leaves, avoid planting near junipers, and apply fungicides early."
}

# Streamlit UI Styling
st.markdown("""
    <style>
        .stButton>button {
            color: white !important;
            background: linear-gradient(to right, #00c853, #b2ff59);
            font-weight: bold;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: green;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='title'>🌿 Plant Disease Detection App 🌿</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>📷 Upload a plant leaf image to check its health status</h4>", unsafe_allow_html=True)

# Upload image
plant_image = st.file_uploader("📸 Choose an image...", type=["jpg", "png", "jpeg", "webp"])

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit = st.button("🔍 Predict Disease")

# On predict button click
if submit:
    if plant_image is not None:
        try:
            # Convert file to OpenCV image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Check if the image is valid
            if opencv_image is None or len(opencv_image.shape) != 3:
                st.error("⚠️ Invalid image! Please upload a valid plant leaf image.")
            else:
                # Convert to RGB
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

                # Display the uploaded image
                st.image(opencv_image, caption="🖼 Uploaded Image", width=250)

                # Preprocess image for model
                processed_image = cv2.resize(opencv_image, (256, 256))
                processed_image = processed_image / 255.0  # Normalize pixel values
                processed_image = np.expand_dims(processed_image, axis=0)  # Reshape for model input

                # Make prediction
                Y_pred = model.predict(processed_image)
                confidence = np.max(Y_pred) * 100
                predicted_class = CLASS_NAMES[np.argmax(Y_pred, axis=1)[0]]

                # Extract plant name and disease
                plant_name, disease = predicted_class.split("-")

                # Display results
                if disease.lower() == "healthy":
                    st.success(f"✅ The {plant_name} plant is *healthy*! 🌱 (Confidence: {confidence:.2f}%)")
                else:
                    disease_key = disease.lower().replace(" ", "_")
                    prevention_tips = DISEASE_PREVENTION.get(disease_key, "📝 No specific prevention tips available.")
                    st.error(f"⚠️ The {plant_name} plant has *{disease.replace('_', ' ')}* disease! ❗ (Confidence: {confidence:.2f}%)")
                    st.warning(prevention_tips)

        except Exception as e:
            st.error(f"❌ An error occurred while processing the image: {e}")
    else:
        st.warning("⚠️ Please upload an image before clicking Predict.")
