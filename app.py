import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# Page configuration
st.set_page_config(
    page_title="Math Expression Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Model parameters
img_height = 135
img_width = 155

# Symbol mapping
symbol_map = {
    'five': '5', 'div': '÷', 'three': '3', 
    'plus': '+', 'minus': '-', 'multiply': '×',
    'equal': '=', 'zero': '0', 'one': '1',
    'two': '2', 'four': '4', 'six': '6',
    'seven': '7', 'eight': '8', 'nine': '9',
    'decimal': '.'
}

@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def get_class_names():
    """Get class names from training data or use defaults"""
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            'data/train',
            image_size=(img_height, img_width),
            batch_size=32
        )
        return train_ds.class_names
    except:
        # Fallback to default class names
        return ['decimal', 'div', 'eight', 'equal', 'five', 'four', 
                'minus', 'multiply', 'nine', 'one', 'plus', 'seven', 
                'six', 'three', 'two', 'zero']

def preprocess_and_segment(image_array):
    """Preprocess image and segment characters"""
    # Convert to grayscale
    if len(image_array.shape) == 3:
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img = image_array
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Horizontal dilation to connect parts
    hor_kernel = np.ones((1, 5), np.uint8)
    thresh_hor = cv2.dilate(thresh, hor_kernel, iterations=1)
    
    # Find vertical cuts
    vertical_sum = np.sum(thresh_hor, axis=0)
    threshold = 5
    cuts = []
    start = None
    
    for i, val in enumerate(vertical_sum):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            end = i
            if end - start > 2:
                cuts.append((start, end))
            start = None
    
    if start is not None:
        cuts.append((start, len(vertical_sum)))
    
    # Extract characters
    characters = []
    for (x_start, x_end) in cuts:
        char_img = thresh[:, x_start:x_end]
        char_img = cv2.bitwise_not(char_img)
        
        # Maintain aspect ratio with padding
        h, w = char_img.shape
        max_dim = max(h, w)
        square_img = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
        
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square_img[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
        
        # Resize to model dimensions
        char_img_resized = cv2.resize(square_img, (img_width, img_height))
        char_img_rgb = cv2.cvtColor(char_img_resized, cv2.COLOR_GRAY2RGB)
        
        characters.append(char_img_rgb)
    
    return characters, thresh

def predict_characters(model, characters, class_names):
    """Predict each character using the CNN model"""
    predictions = []
    confidences = []
    all_probabilities = []
    
    for char_img in characters:
        # Prepare image for prediction
        img_array = np.expand_dims(char_img, 0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)
        probabilities = tf.nn.softmax(prediction[0])
        
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        predictions.append(predicted_class)
        confidences.append(confidence)
        all_probabilities.append(probabilities.numpy())
    
    return predictions, confidences, all_probabilities

def display_results(predictions, confidences, all_probs, class_names, show_confidence, show_top_predictions):
    """Display recognition results"""
    # Individual predictions
    if show_confidence:
        st.write("**Individual Character Predictions:**")
        result_cols = st.columns(len(predictions))
        
        for idx, (col, pred, conf) in enumerate(zip(result_cols, predictions, confidences)):
            with col:
                symbol = symbol_map.get(pred, pred)
                confidence_color = "#28a745" if conf > 0.9 else "#ffc107" if conf > 0.7 else "#dc3545"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 8px; border: 2px solid {confidence_color};">
                    <h1 style="margin: 0; color: #1E88E5; font-size: 2.5rem;">{symbol}</h1>
                    <p style="margin: 5px 0; font-size: 0.85rem; color: #666;">{pred}</p>
                    <p style="margin: 0; font-size: 1rem; font-weight: bold; color: {confidence_color};">{conf*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show top 3 predictions if enabled
                if show_top_predictions:
                    with st.expander("Top 3"):
                        top_3_idx = np.argsort(all_probs[idx])[-3:][::-1]
                        for i, top_idx in enumerate(top_3_idx):
                            st.write(f"{i+1}. {class_names[top_idx]}: {all_probs[idx][top_idx]*100:.1f}%")
    
    # Final expression
    st.markdown("---")
    expression = ' '.join([symbol_map.get(p, p) for p in predictions])
    word_expression = ''.join(predictions)
    avg_confidence = np.mean(confidences) * 100
    
    # Determine confidence level
    if avg_confidence >= 90:
        conf_text = "Excellent"
        conf_color = "#28a745"
    elif avg_confidence >= 75:
        conf_text = "Good"
        conf_color = "#ffc107"
    else:
        conf_text = "Low"
        conf_color = "#dc3545"
    
    st.markdown(f"""
    <div style="padding: 25px; border-radius: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin: 0; text-align: center; color: white;">✨ Final Expression ✨</h2>
        <h1 style="margin: 15px 0; text-align: center; font-size: 4rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{expression}</h1>
        <p style="margin: 0; text-align: center; color: #e0e0e0; font-size: 1.1rem;">({word_expression})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Confidence", 
            value=f"{avg_confidence:.1f}%",
        )
    
    with col2:
        st.metric(
            label="Characters Recognized",
            value=len(predictions)
        )
    
    with col3:
        min_conf = min(confidences) * 100
        st.metric(
            label="Lowest Confidence",
            value=f"{min_conf:.1f}%"
        )
    
    # Detailed results
    with st.expander("Detailed Results"):
        st.write("**Character-by-Character Breakdown:**")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            symbol = symbol_map.get(pred, pred)
            st.write(f"Position {i+1}: **{symbol}** ({pred}) - {conf*100:.2f}% confidence")
    
    # Download results
    st.markdown("---")
    result_text = f"""Math Expression Recognition Results
==========================================

Expression: {expression}
Word Form: {word_expression}
Average Confidence: {avg_confidence:.2f}%

Character Details:
"""
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        symbol = symbol_map.get(pred, pred)
        result_text += f"\nPosition {i+1}: {symbol} ({pred}) - {conf*100:.2f}%"
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Results (TXT)",
            data=result_text,
            file_name="recognition_results.txt",
            mime="text/plain"
        )
    
    with col2:
        csv_data = "Position,Symbol,Class,Confidence\n"
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            symbol = symbol_map.get(pred, pred)
            csv_data += f"{i+1},{symbol},{pred},{conf*100:.2f}\n"
        
        st.download_button(
            label="Download Results (CSV)",
            data=csv_data,
            file_name="recognition_results.csv",
            mime="text/csv"
        )

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🔢 Math Expression Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image or draw a mathematical expression and let AI recognize it!</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/math.png", width=100)
        
        st.header("About")
        st.info(
            """
            This app uses a **Convolutional Neural Network (CNN)** to recognize 
            handwritten mathematical expressions.
            
            **Supported Characters:**
            - **Digits**: 0-9
            - **Operators**: +, -, ×, ÷, =
            - **Special**: . (decimal)
            """
        )
        
        st.header("Instructions")
        st.markdown("""
        1. Upload or Draw
        2. Wait for processing
        3. View segmented characters
        4. See final prediction
        """)
        
        st.header("Settings")
        show_intermediate = st.checkbox("Show preprocessing steps", value=True)
        show_confidence = st.checkbox("Show confidence scores", value=True)
        show_top_predictions = st.checkbox("Show top 3 predictions", value=False)
        
    
    # Load model and class names
    model = load_model()
    class_names = get_class_names()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'model.h5' exists in the project directory.")
        st.info("Train the model first by running: `python train_model.py`")
        return
    
    st.success("Model loaded successfully!")
    st.info(f"Model recognizes {len(class_names)} different classes")
    
    # Input method selection
    st.markdown("---")
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Draw on Canvas"],
        horizontal=True,
        help="Select how you want to provide the mathematical expression"
    )
    
    # Initialize image_to_process
    image_to_process = None
    
    if input_method == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of a mathematical expression", 
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear image with handwritten digits and operators"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_to_process = np.array(image)
            
            st.markdown("---")
            st.subheader("Uploaded Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Original Expression")
            st.write(f"**Image size:** {image_to_process.shape[1]} × {image_to_process.shape[0]} pixels")
    
    else:  # Draw on Canvas
        st.markdown("---")
        st.subheader("Draw Your Mathematical Expression")
        
        # Canvas settings
        col1, col2, col3 = st.columns(3)
        with col1:
            stroke_width = st.slider("Brush Size", 1, 25, 8)
        
        # Fixed colors
        stroke_color = "#000000"  # Black
        bg_color = "#FFFFFF"  # White
        
        # Create canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=300,
            width=700,
            drawing_mode="freedraw",
            key="canvas",
            display_toolbar=True,
        )
        
        # Instructions for canvas
        st.info(" **Tips:** Draw clearly with good spacing between characters. Click the trash icon to clear and redraw.")
        
        # Process canvas drawing
        if canvas_result is not None and canvas_result.image_data is not None:
            try:
                # Convert canvas data to usable image
                canvas_image = canvas_result.image_data.astype(np.uint8)
                
                # Check if anything has been drawn (check if alpha channel has content)
                if np.sum(canvas_image[:, :, 3]) > 100:  # Threshold for meaningful drawing
                    # Convert RGBA to RGB by dropping alpha channel
                    rgb_image = canvas_image[:, :, :3].copy()
                    
                    # Set image to process
                    image_to_process = rgb_image
                    
                    # Display what was drawn
                    with st.expander("Preview Drawn Image"):
                        st.image(rgb_image, caption="Your Drawing")
                else:
                    image_to_process = None
            except Exception as e:
                st.error(f"Error processing canvas: {e}")
                image_to_process = None
    
    # Process button
    if image_to_process is not None:
        st.markdown("---")
        if st.button("Recognize Expression", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Segment characters
                    characters, thresh_img = preprocess_and_segment(image_to_process)
                    
                    if len(characters) == 0:
                        st.error("No characters detected. Please try:")
                        st.markdown("""
                        - A clearer image with better contrast
                        - Black text on white background
                        - Larger characters
                        - Better spacing between characters
                        - Less background noise
                        """)
                        return
                    
                    # Show intermediate steps
                    if show_intermediate:
                        st.markdown("---")
                        st.subheader("Preprocessing Steps")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(thresh_img, caption="Binary Threshold (Inverted)", clamp=True)
                        with col2:
                            st.metric("Characters Detected", len(characters))
                            st.write("**Processing Steps:**")
                            st.write("1. Convert to grayscale")
                            st.write("2. Apply binary threshold")
                            st.write("3. Remove noise")
                            st.write("4. Segment characters")
                            st.write("5. Maintain aspect ratio")
                    
                    # Display segmented characters
                    st.markdown("---")
                    st.subheader("Segmented Characters")
                    
                    cols = st.columns(min(len(characters), 10))
                    for idx, (col, char) in enumerate(zip(cols, characters)):
                        with col:
                            st.image(char, caption=f"#{idx+1}")
                    
                    # Predict
                    with st.spinner("Running CNN prediction..."):
                        predictions, confidences, all_probs = predict_characters(model, characters, class_names)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Recognition Results")
                    
                    display_results(predictions, confidences, all_probs, class_names, show_confidence, show_top_predictions)
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.info("Please try with a different image or check the error details above.")
    
    elif input_method == "Upload Image":
        # Show helpful info when no image is uploaded
        st.markdown("---")
        st.info("Upload an image above to get started!")
        
        with st.expander("Tips for Best Results"):
            st.markdown("""
            ### For optimal recognition:
            
            1. **Image Quality**
               - Use clear, high-contrast images
               - Ensure characters are not blurry
               
            2. **Background**
               - White or light background works best
               - Remove unnecessary background objects
               
            3. **Character Style**
               - Handwritten characters (matching training data)
               - Well-formed digits and symbols
               - Adequate spacing between characters
               
            4. **Image Size**
               - Not too small (minimum 200px width)
               - Not too large (maximum 2000px)
               
            5. **File Format**
               - PNG, JPG, JPEG, or BMP
               - Avoid heavily compressed images
            """)
    
    else:  # Canvas mode but nothing drawn yet
        st.markdown("---")
        st.info("Draw a mathematical expression on the canvas above, then click 'Recognize Expression'!")
        
        with st.expander("Drawing Tips"):
            st.markdown("""
            ### For best results when drawing:
            
            1. **Clear Writing**
               - Write digits and symbols clearly
               - Use consistent stroke width
               
            2. **Good Spacing**
               - Leave adequate space between characters
               - Don't make characters too small
               
            3. **Single Line**
               - Write expression in a single horizontal line
               - Avoid overlapping characters
               
            4. **Contrast**
               - Use black pen on white background (default)
               - Ensure good visibility
               
            5. **Size**
               - Make characters reasonably large
               - Fill up the canvas height
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⭐ Star this project on <a href="https://github.com/nikhilsshekhawat/Mathematical-Expression-Recognition-using-CNN" target="_blank">GitHub</a></p>
        <p style="font-size: 0.9rem; color: #999;">Built with ❤️ for the AI community</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()

