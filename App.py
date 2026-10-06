import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 CIFAR-10 CNN Image Classifier")
st.write("Upload an image and let the CNN predict the class.")


class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_model.keras")

model = load_model()

st.success("CNN model loaded successfully!")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

    image_resized = image.resize((32, 32))

    # Convert to NumPy
    image_array = np.array(image_resized)

    # Normalize
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    

    if st.button("🔮 Predict"):

        prediction = model.predict(image_array)

        predicted_class = np.argmax(prediction[0])

        confidence = prediction[0][predicted_class] * 100

        st.subheader("Prediction")

        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    

        probabilities = prediction[0] * 100

        st.subheader("Class Probabilities")

        for i, probability in enumerate(probabilities):

            st.write(
                f"{class_names[i]}: {probability:.2f}%"
            )

            st.progress(
                float(probability / 100)
            )
