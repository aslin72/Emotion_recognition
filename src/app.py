import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import plotly.graph_objects as go
from tensorflow.keras.layers import TFSMLayer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .debug-info {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
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
    "#ff6b6b",  # Angry - Red
    "#4ecdc4",  # Disgust - Teal
    "#45b7d1",  # Fear - Blue
    "#96ceb4",  # Happy - Green
    "#ffeaa7",  # Sad - Yellow
    "#dda0dd",  # Surprise - Purple
    "#95a5a6",  # Neutral - Gray
]


@st.cache_resource
def load_saved_model(model_path):
    """Load TensorFlow SavedModel with better error handling"""
    try:
        # Try loading as SavedModel first
        model = tf.saved_model.load(model_path)
        st.success("✅ Model loaded as SavedModel")
        return model, "savedmodel"
    except Exception as e1:
        try:
            # Try loading with TFSMLayer
            tfsm_layer = tf.keras.layers.TFSMLayer(
                model_path, call_endpoint="serving_default"
            )
            st.success("✅ Model loaded as TFSMLayer")
            return tfsm_layer, "tfsmlayer"
        except Exception as e2:
            try:
                # Try loading as Keras model
                model = tf.keras.models.load_model(model_path)
                st.success("✅ Model loaded as Keras model")
                return model, "keras"
            except Exception as e3:
                st.error(f"All loading methods failed:")
                st.error(f"SavedModel: {str(e1)}")
                st.error(f"TFSMLayer: {str(e2)}")
                st.error(f"Keras: {str(e3)}")
                return None, None


def enhanced_preprocess_face(face_image, debug=False):
    """Enhanced preprocessing pipeline that matches FER2013 training"""
    original_shape = face_image.shape

    # Convert to grayscale if needed
    if len(face_image.shape) == 3:
        if face_image.shape[2] == 4:  # RGBA
            face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_image.copy()

    # Apply histogram equalization to improve contrast (common in FER2013 preprocessing)
    gray = cv2.equalizeHist(gray)

    # Resize to 48x48 with better interpolation
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur to reduce noise (optional, test both)
    # resized = cv2.GaussianBlur(resized, (3, 3), 0)

    # Normalize to [0, 1] - ensure float32
    normalized = resized.astype(np.float32) / 255.0

    # Optional: Apply standardization (zero mean, unit variance)
    # This depends on how your model was trained
    # normalized = (normalized - np.mean(normalized)) / (np.std(normalized) + 1e-8)

    # Add batch and channel dimensions
    processed = normalized.reshape(1, 48, 48, 1)

    if debug:
        st.write(f"Original shape: {original_shape}")
        st.write(f"Processed shape: {processed.shape}")
        st.write(f"Pixel value range: [{processed.min():.3f}, {processed.max():.3f}]")
        st.write(f"Mean: {processed.mean():.3f}, Std: {processed.std():.3f}")

        # Show preprocessing steps
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray, caption="Grayscale", width=150)
        with col2:
            st.image(cv2.equalizeHist(gray), caption="Hist Equalized", width=150)
        with col3:
            st.image(resized, caption="Resized (48x48)", width=150)

    return processed


def robust_predict_emotion(model, face_image, model_type, debug=False):
    """Robust emotion prediction with proper output handling"""
    try:
        processed_face = enhanced_preprocess_face(face_image, debug=debug)

        if model_type == "savedmodel":
            # For SavedModel, use infer function
            predictions = model.signatures["serving_default"](
                tf.constant(processed_face, dtype=tf.float32)
            )
            # Extract predictions from the output dictionary
            if isinstance(predictions, dict):
                # Common output names to try
                output_keys = list(predictions.keys())
                if debug:
                    st.write(f"Output keys: {output_keys}")

                for key in ["output_0", "dense", "predictions", "logits"]:
                    if key in predictions:
                        pred_array = predictions[key].numpy()[0]
                        break
                else:
                    # Take the first output
                    pred_array = list(predictions.values())[0].numpy()[0]
            else:
                pred_array = predictions.numpy()[0]

        elif model_type == "tfsmlayer":
            # For TFSMLayer
            predictions = model(processed_face)
            if isinstance(predictions, dict):
                output_keys = list(predictions.keys())
                if debug:
                    st.write(f"TFSMLayer output keys: {output_keys}")

                for key in ["output_0", "dense", "predictions", "logits"]:
                    if key in predictions:
                        pred_array = predictions[key].numpy()[0]
                        break
                else:
                    pred_array = list(predictions.values())[0].numpy()[0]
            else:
                pred_array = predictions.numpy()[0]

        elif model_type == "keras":
            # For Keras model
            predictions = model.predict(processed_face, verbose=0)
            pred_array = predictions[0]

        # Apply softmax if predictions are logits (not probabilities)
        if np.max(np.abs(pred_array)) > 1.0:  # Likely logits
            pred_array = tf.nn.softmax(pred_array).numpy()

        # Ensure we have 7 classes
        if len(pred_array) != 7:
            raise ValueError(f"Expected 7 emotion classes, got {len(pred_array)}")

        if debug:
            st.write(f"Raw predictions: {pred_array}")
            st.write(f"Prediction sum: {pred_array.sum():.3f}")
            st.write(f"Max prediction: {pred_array.max():.3f}")

        return pred_array

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        logger.error(f"Prediction failed: {str(e)}")
        return np.array([1 / 7] * 7)  # Return uniform distribution as fallback


def detect_faces_advanced(image, debug=False):
    """Advanced face detection with multiple cascade classifiers"""
    # Try multiple cascade classifiers
    cascade_files = [
        cv2.data.haarcascades + r"models\haarcascade_frontalface_default.xml",
        cv2.data.haarcascades + r"models\haarcascade_frontalface_alt.xml",
        cv2.data.haarcascades + r"models\haarcascade_frontalface_alt2.xml",
    ]

    all_faces = []

    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply histogram equalization to improve detection
    gray = cv2.equalizeHist(gray)

    for cascade_file in cascade_files:
        try:
            face_cascade = cv2.CascadeClassifier(cascade_file)

            # Try different scale factors and min neighbors
            for scale_factor in [1.1, 1.05, 1.3]:
                for min_neighbors in [3, 5, 7]:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30),
                        maxSize=(300, 300),
                    )

                    if len(faces) > 0:
                        all_faces.extend(faces)
        except Exception as e:
            if debug:
                st.write(f"Cascade {cascade_file} failed: {e}")
            continue

    # Remove duplicate faces using Non-Maximum Suppression
    if len(all_faces) > 0:
        all_faces = np.array(all_faces)
        # Simple NMS - remove faces that overlap significantly
        unique_faces = []
        for face in all_faces:
            is_duplicate = False
            for unique_face in unique_faces:
                # Calculate overlap
                x1, y1, w1, h1 = face
                x2, y2, w2, h2 = unique_face

                # Calculate intersection
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

                if xi1 < xi2 and yi1 < yi2:
                    intersection = (xi2 - xi1) * (yi2 - yi1)
                    area1, area2 = w1 * h1, w2 * h2
                    union = area1 + area2 - intersection
                    overlap = intersection / union if union > 0 else 0

                    if overlap > 0.3:  # 30% overlap threshold
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_faces.append(face)

        return np.array(unique_faces)

    return np.array([])


def create_enhanced_confidence_chart(predictions, threshold=0.1):
    """Create enhanced confidence chart with threshold line"""
    fig = go.Figure()

    # Add bars
    fig.add_trace(
        go.Bar(
            x=EMOTIONS,
            y=predictions,
            marker_color=EMOTION_COLORS,
            text=[f"{p:.1%}" for p in predictions],
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>",
            name="Confidence",
        )
    )

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold:.1%}",
    )

    fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis_title="Emotions",
        yaxis_title="Confidence",
        showlegend=False,
        height=400,
        yaxis=dict(tickformat=".0%", range=[0, max(1.0, predictions.max() * 1.1)]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def draw_faces_on_image(image, faces, predictions_list, threshold=0.3):
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

            # Color code based on confidence
            color = (
                (0, 255, 0) if confidence >= threshold else (0, 165, 255)
            )  # Green if confident, orange if not

            label = f"{emotion}: {confidence:.2f}"

            # Add background rectangle for better text visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(
                img_with_faces, (x, y - 35), (x + label_size[0], y - 5), (0, 0, 0), -1
            )

            cv2.putText(
                img_with_faces,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

    return img_with_faces


# Main UI
st.markdown(
    '<h1 class="main-header">🎭 Enhanced Emotion Detection AI</h1>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Settings")

    # Model path input
    model_path = st.text_input(
        "SavedModel Folder Path",
        placeholder="./best_model",
        help="Enter the path to your SavedModel folder",
    )

    # Load model button
    if st.button("🔄 Load Model"):
        if model_path:
            with st.spinner("Loading model..."):
                model, model_type = load_saved_model(model_path)
                if model:
                    st.session_state.model = model
                    st.session_state.model_type = model_type
                    st.session_state.model_loaded = True
                else:
                    st.session_state.model_loaded = False
        else:
            st.warning("Please enter model path")

    # Model status indicator
    if st.session_state.model_loaded:
        st.markdown(
            f'<span class="status-indicator" style="background-color: #4CAF50;"></span>**Model Ready** ({st.session_state.get("model_type", "unknown")})',
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
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)
    show_face_boxes = st.checkbox("Show Face Bounding Boxes", value=True)
    show_all_emotions = st.checkbox("Show All Emotion Scores", value=True)
    debug_mode = st.checkbox(
        "Debug Mode", value=False, help="Show detailed preprocessing info"
    )

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        apply_enhancement = st.checkbox("Apply Image Enhancement", value=True)
        enhancement_factor = st.slider("Enhancement Factor", 0.5, 2.0, 1.2, 0.1)

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

        # Apply enhancement if enabled
        if apply_enhancement:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhancement_factor)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(enhancement_factor)

        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display image info
        st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")

with col2:
    st.header("🧠 Emotion Analysis")

    if uploaded_file and st.session_state.model_loaded:
        with st.spinner("Analyzing emotions..."):
            # Detect faces with advanced method
            faces = detect_faces_advanced(img_array, debug=debug_mode)

            if len(faces) > 0:
                st.success(f"🔍 Detected {len(faces)} face(s)")

                # Process each face
                all_predictions = []
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face region with some padding
                    padding = 10
                    y1 = max(0, y - padding)
                    y2 = min(img_array.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(img_array.shape[1], x + w + padding)

                    face_roi = img_array[y1:y2, x1:x2]

                    if debug_mode:
                        st.write(f"**Face {i+1} Debug Info:**")
                        col_debug1, col_debug2 = st.columns(2)
                        with col_debug1:
                            st.image(face_roi, caption=f"Face {i+1} ROI", width=150)

                    # Predict emotion
                    predictions = robust_predict_emotion(
                        st.session_state.model,
                        face_roi,
                        st.session_state.model_type,
                        debug=debug_mode,
                    )
                    all_predictions.append(predictions)

                    # Get dominant emotion
                    emotion_idx = np.argmax(predictions)
                    emotion = EMOTIONS[emotion_idx]
                    confidence = predictions[emotion_idx]

                    # Get second highest for comparison
                    sorted_indices = np.argsort(predictions)[::-1]
                    second_emotion = EMOTIONS[sorted_indices[1]]
                    second_confidence = predictions[sorted_indices[1]]

                    # Display result for each face
                    if confidence >= confidence_threshold:
                        st.markdown(
                            f"""
                        <div class="emotion-result" style="background-color: {EMOTION_COLORS[emotion_idx]}20; 
                             border: 3px solid {EMOTION_COLORS[emotion_idx]};">
                            Face {i+1}: <span style="color: {EMOTION_COLORS[emotion_idx]};">{emotion}</span><br>
                            <span style="font-size: 1.5rem;">{confidence:.1%} confidence</span><br>
                            <span style="font-size: 1rem;">2nd: {second_emotion} ({second_confidence:.1%})</span>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(
                            f"Face {i+1}: Low confidence - {emotion} ({confidence:.1%}) | 2nd: {second_emotion} ({second_confidence:.1%})"
                        )

                    # Debug information
                    if debug_mode:
                        st.markdown(
                            f"""
                        <div class="debug-info">
                        <strong>Face {i+1} Debug:</strong><br>
                        Region: ({x}, {y}, {w}, {h})<br>
                        Top 3 predictions:<br>
                        1. {EMOTIONS[sorted_indices[0]]}: {predictions[sorted_indices[0]]:.3f}<br>
                        2. {EMOTIONS[sorted_indices[1]]}: {predictions[sorted_indices[1]]:.3f}<br>
                        3. {EMOTIONS[sorted_indices[2]]}: {predictions[sorted_indices[2]]:.3f}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                # Show image with face boxes
                if show_face_boxes:
                    img_with_faces = draw_faces_on_image(
                        img_array, faces, all_predictions, confidence_threshold
                    )
                    st.image(
                        img_with_faces, caption="Detected Faces", use_column_width=True
                    )

                # Show confidence chart for each face
                if show_all_emotions and len(all_predictions) > 0:
                    for i, predictions in enumerate(all_predictions):
                        fig = create_enhanced_confidence_chart(
                            predictions, confidence_threshold
                        )
                        fig.update_layout(
                            title=f"Face {i+1} - Emotion Confidence Scores"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("❌ No faces detected in the image")
                st.info("💡 Tips:")
                st.info("• Ensure faces are clearly visible and front-facing")
                st.info("• Try adjusting image enhancement settings")
                st.info("• Make sure the image has good lighting and contrast")

                if debug_mode:
                    st.write("**Debug: Trying face detection on enhanced image...**")
                    # Try with different preprocessing
                    gray = (
                        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        if len(img_array.shape) == 3
                        else img_array
                    )
                    enhanced_gray = cv2.equalizeHist(gray)

                    col_debug1, col_debug2 = st.columns(2)
                    with col_debug1:
                        st.image(gray, caption="Original Grayscale", width=200)
                    with col_debug2:
                        st.image(enhanced_gray, caption="Enhanced Grayscale", width=200)

    elif uploaded_file and not st.session_state.model_loaded:
        st.error("❌ Please load the model first")

    elif not uploaded_file:
        st.info("📤 Upload an image to start emotion detection")

# Footer with troubleshooting tips
st.divider()

with st.expander("🔧 Troubleshooting Guide"):
    st.markdown(
        """
    **If the model shows low confidence or wrong predictions:**
    
    1. **Model Overfitting**: Your model shows 87% training vs 77% validation accuracy, indicating overfitting
       - Consider using dropout, batch normalization, or data augmentation
       - Reduce model complexity or use early stopping
    
    2. **Preprocessing Mismatch**: Ensure inference preprocessing matches training
       - Check if histogram equalization was used during training
       - Verify normalization method (0-1 vs standardization)
       - Confirm input image dimensions and channels
    
    3. **Face Detection Issues**:
       - Try enabling image enhancement
       - Adjust confidence threshold (lower for more lenient detection)
       - Use debug mode to see preprocessing steps
    
    4. **Model Output Issues**:
       - Check if model outputs logits or probabilities
       - Verify the correct output key from SavedModel
       - Ensure model expects single-channel (grayscale) input
    
    **Best Practices:**
    - Use images with clear, front-facing faces
    - Ensure good lighting and contrast
    - Avoid heavily compressed or low-resolution images
    - Test with images similar to your training data
    """
    )

st.markdown(
    """
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Enhanced CNN Model for FER2013 Dataset</strong> • Multiple Model Format Support</p>
    <p>Supported emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral</p>
    <p>Features: Advanced face detection, robust preprocessing, debug mode</p>
</div>
""",
    unsafe_allow_html=True,
)

# Model info expandable section
if st.session_state.model_loaded:
    with st.expander("📋 Model Information & Recommendations"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Current Model Details:**")
            st.write("• Input Shape: (48, 48, 1)")
            st.write("• Output Classes: 7 emotions")
            st.write(f"• Format: {st.session_state.get('model_type', 'unknown')}")
            st.write("• Training Accuracy: 87.12%")
            st.write("• Validation Accuracy: 77.22%")
        with col2:
            st.write("**Improvement Suggestions:**")
            st.write("• **Reduce Overfitting**: Add dropout, regularization")
            st.write("• **Data Augmentation**: Rotation, shift, brightness")
            st.write("• **Ensemble Methods**: Combine multiple models")
            st.write("• **Transfer Learning**: Use pre-trained features")
            st.write("• **Class Balancing**: Handle imbalanced dataset")
