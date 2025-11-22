import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

st.title("🧫 Microscopy Cell Detector (RBC / WBC / Platelets) - ONNX Runtime")
st.write("Upload an image and adjust confidence threshold to detect cells.")

# Confidence slider
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
    return session

session = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((640, 640))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = img_array.transpose(2, 0, 1) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    outputs = session.run(None, {"images": img_array})

    boxes = outputs[0]  # YOLO ONNX output: [x1, y1, x2, y2, score, class]
    class_names = {0: "RBC", 1: "WBC", 2: "Platelets"}

    counts = {"RBC": 0, "WBC": 0, "Platelets": 0}

    # Draw bounding boxes
    img_draw = np.array(img_resized).copy()

    for box in boxes:
        x1, y1, x2, y2, score, cls = box
        if score >= conf_threshold:
            cls = int(cls)
            name = class_names[cls]
            counts[name] += 1

            # Draw rectangle
            img_draw = cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            img_draw = cv2.putText(img_draw, f"{name} {score:.2f}", (int(x1), int(y1)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    st.image(img_draw, caption="Detection Result", use_column_width=True)

    st.subheader("🧮 Cell Counts")
    for k, v in counts.items():
        st.write(f"**{k}** : {v}")
