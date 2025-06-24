import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.layers import TFSMLayer

# Page config
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .emotion-result {
        font-size: 2.5rem;
        font-weight: bold;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None

# Emotion configuration
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_COLORS = [
    "#ff6b6b",
    "#4ecdc4",
    "#45b7d1",
    "#96ceb4",
    "#ffeaa7",
    "#dda0dd",
    "#95a5a6",
]


@st.cache_resource
def load_saved_model(model_path):
    """Load TensorFlow SavedModel using TFSMLayer"""
    try:
        # Load as TFSMLayer for Keras 3 compatibility
        tfsm_layer = tf.keras.layers.TFSMLayer(
            model_path, call_endpoint="serving_default"
        )
        return tfsm_layer
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None


def preprocess_face(face_image):
    """Preprocess face image for FER2013 model"""
    # Convert to grayscale if needed
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_image

    # Resize to 48x48 (FER2013 standard)
    resized = cv2.resize(gray, (48, 48))

    # Normalize pixel values to [0, 1]
    normalized = resized.astype("float32") / 255.0

    # Add batch and channel dimensions
    return normalized.reshape(1, 48, 48, 1)


def predict_emotion(model, face_image):
    """Predict emotion from face image"""
    processed_face = preprocess_face(face_image)
    # TFSMLayer returns a dictionary, get the output tensor
    predictions = model(processed_face)
    # Extract the actual predictions (adjust key name if needed)
    if isinstance(predictions, dict):
        # Try common output names
        for key in ["dense", "output", "predictions", "dense_1"]:
            if key in predictions:
                return predictions[key].numpy()[0]
        # If no standard key found, take the first value
        return list(predictions.values())[0].numpy()[0]
    else:
        return predictions.numpy()[0]


def detect_faces(image):
    """Detect faces in image using OpenCV"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Check if image is grayscale or color
    if len(image.shape) == 2:
        gray = image  # Already grayscale
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces


def create_confidence_chart(predictions):
    """Create interactive confidence chart"""
    fig = go.Figure(
        data=[
            go.Bar(
                x=EMOTIONS,
                y=predictions,
                marker_color=EMOTION_COLORS,
                text=[f"{p:.1%}" for p in predictions],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis_title="Emotions",
        yaxis_title="Confidence",
        showlegend=False,
        height=400,
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def draw_faces_on_image(image, faces, predictions_list):
    """Draw bounding boxes and emotions on detected faces"""
    img_with_faces = image.copy()

    for i, (x, y, w, h) in enumerate(faces):
        # Draw rectangle around face
        cv2.rectangle(img_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Add emotion label if predictions available
        if i < len(predictions_list):
            emotion_idx = np.argmax(predictions_list[i])
            emotion = EMOTIONS[emotion_idx]
            confidence = predictions_list[i][emotion_idx]

            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(
                img_with_faces,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    return img_with_faces


# Main UI
st.markdown(
    '<h1 class="main-header">🎭 Emotion Detection AI</h1>', unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Settings")

    # Model path input
    model_path = st.text_input(
        "SavedModel Folder Path",
        placeholder="./best_model",
        help="Enter the path to your SavedModel folder (e.g., ./best_model)",
    )

    # Load model button
    if st.button("🔄 Load Model"):
        if model_path:
            with st.spinner("Loading SavedModel..."):
                model = load_saved_model(model_path)
                if model:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.session_state.model_loaded = False
        else:
            st.warning("Please enter model path")

    # Model status indicator
    if st.session_state.model_loaded:
        st.markdown(
            '<span class="status-indicator" style="background-color: #4CAF50;"></span>**Model Ready**',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-indicator" style="background-color: #f44336;"></span>**Model Not Loaded**',
            unsafe_allow_html=True,
        )

    st.divider()

    # Detection settings
    st.subheader("🎯 Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.3, 0.05
    )  # Default 30% confidence
    show_face_boxes = st.checkbox("Show Face Bounding Boxes", value=True)
    show_all_emotions = st.checkbox("Show All Emotion Scores", value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Image Upload")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload an image containing faces for emotion detection",
    )

    if uploaded_file:
        # Load and display image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display image info
        st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.header("🧠 Emotion Analysis")

    if uploaded_file and st.session_state.model_loaded:
        with st.spinner("Analyzing emotions..."):
            # Detect faces
            faces = detect_faces(img_array)

            if len(faces) > 0:
                st.success(f"🔍 Detected {len(faces)} face(s)")

                # Process each face
                all_predictions = []
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face region
                    face_roi = img_array[y : y + h, x : x + w]

                    # Predict emotion
                    predictions = predict_emotion(st.session_state.model, face_roi)
                    all_predictions.append(predictions)

                    # Get dominant emotion
                    emotion_idx = np.argmax(predictions)
                    emotion = EMOTIONS[emotion_idx]
                    confidence = predictions[emotion_idx]

                    # Display result for each face
                    if confidence >= confidence_threshold:
                        st.markdown(
                            f"""
                        <div class="emotion-result" style="background-color: {EMOTION_COLORS[emotion_idx]}20; 
                             border: 3px solid {EMOTION_COLORS[emotion_idx]};">
                            Face {i+1}: <span style="color: {EMOTION_COLORS[emotion_idx]};">{emotion}</span><br>
                            <span style="font-size: 1.5rem;">{confidence:.1%} confidence</span>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(
                            f"Face {i+1}: Low confidence - {emotion} ({confidence:.1%})"
                        )

                # Show image with face boxes
                if show_face_boxes:
                    img_with_faces = draw_faces_on_image(
                        img_array, faces, all_predictions
                    )
                    st.image(
                        img_with_faces, caption="Detected Faces", use_column_width=True
                    )

                # Show confidence chart for first face
                if show_all_emotions and len(all_predictions) > 0:
                    fig = create_confidence_chart(all_predictions[0])
                    st.plotly_chart(fig, use_container_width=True)

                    if len(faces) > 1:
                        st.info("📊 Chart shows emotions for the first detected face")

            else:
                st.error("❌ No faces detected in the image")
                st.info("💡 Tips: Upload an image with clear, front-facing faces")

    elif uploaded_file and not st.session_state.model_loaded:
        st.error("❌ Please load the model first")

    elif not uploaded_file:
        st.info("📤 Upload an image to start emotion detection")

# Footer
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>CNN Model trained on FER2013 Dataset</strong> • TensorFlow SavedModel Format</p>
    <p>Supported emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral</p>
</div>
""",
    unsafe_allow_html=True,
)

# Model info expandable section
if st.session_state.model_loaded:
    with st.expander("📋 Model Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Details:**")
            st.write("• Input Shape: (48, 48, 1)")
            st.write("• Output Classes: 7 emotions")
            st.write("• Format: TensorFlow SavedModel")
        with col2:
            st.write("**Dataset:**")
            st.write("• FER2013 (Facial Expression Recognition)")
            st.write("• Grayscale images")
            st.write("• 48x48 pixel resolution")
