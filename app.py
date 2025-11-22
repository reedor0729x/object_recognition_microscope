import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Microscopic Tiny Cell Detector (ONNX)", layout="wide")

# ----------------- MODEL INFO -----------------
MODEL_PATH = "best.onnx"
CLASSES = ["RBC", "WBC", "Platelets"]  # update if needed

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ----------------- PREPROCESSING -----------------
def preprocess(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (640, 640)) / 255.0
    tensor = resized.transpose(2, 0, 1).astype(np.float32)
    return img, np.expand_dims(tensor, axis=0)


# ----------------- POSTPROCESSING -----------------
def postprocess(pred, original):
    # Match output orientation
    if pred.shape[0] < pred.shape[1]:  # reshape (84,8400) -> (8400,84)
        pred = pred.T

    boxes = pred[:, :4]
    scores = pred[:, 4]
    class_ids = pred[:, 5:].argmax(axis=1)

    mask = scores > 0.25
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    h, w, _ = original.shape
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w / 640  # x1
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h / 640  # y1
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w / 640  # x2
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h / 640  # y2

    return xyxy, scores, class_ids


# ----------------- DRAW BOXES -----------------
def draw_boxes(image, boxes, class_ids, scores):
    counts = {"RBC": 0, "WBC": 0, "Platelets": 0}

    for box, cls, score in zip(boxes, class_ids, scores):
        label = CLASSES[int(cls)]
        counts[label] += 1

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image, counts


# ----------------- STREAMLIT UI -----------------
st.title("🧪 Microscopic Tiny Cell Detector (ONNX)")
uploaded_file = st.file_uploader("Upload microscopic image",
                                 type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    original, input_tensor = preprocess(img_bgr)

    # ------------ PRINT MODEL METADATA FOR DEBUGGING -------------
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    st.write("🔌 **Model Input Name:**", input_name)
    st.write("📦 **Model Input Shape:**", input_shape)
    st.write("📦 **Model Output Shape:**", output_shape)
    st.write("🎯 **Using tensor shape:**", input_tensor.shape)

    # ---------------- INFERENCE ----------------
    outputs = session.run([output_name], {input_name: input_tensor})
    pred = outputs[0][0]

    # ---------------- POST PROCESS ----------------
    boxes, scores, class_ids = postprocess(pred, original)
    result_img, counts = draw_boxes(original.copy(), boxes, class_ids, scores)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Original Image")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.subheader("📤 Processed Output")
        disp = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(disp, use_column_width=True)

    st.subheader("📊 Cell Counts")
    st.json(counts)
