import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Microscopic Cell Detector", layout="wide")

# ---------- Model & Classes ----------
MODEL_PATH = "best.onnx"
CLASSES = ["RBC", "WBC", "Platelets"]   # change if different

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640)) / 255.0
    img_input = img_resized.transpose(2, 0, 1).astype(np.float32)
    return img, np.expand_dims(img_input, axis=0)


def postprocess(pred, input_image):
    pred = pred.squeeze().T
    boxes = pred[:, :4]
    scores = pred[:, 4]
    class_ids = pred[:, 5:].argmax(axis=1)
    confidences = scores

    # Filter predictions
    mask = confidences > 0.5
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    # Convert xywh to xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # scale to full image
    h, w, _ = input_image.shape
    boxes_xyxy[:, [0, 2]] *= w / 640
    boxes_xyxy[:, [1, 3]] *= h / 640

    return boxes_xyxy, confidences, class_ids


def draw_boxes(image, boxes, class_ids, confs):
    counts = {"RBC": 0, "WBC": 0, "Platelets": 0}
    for box, class_id, conf in zip(boxes, class_ids, confs):
        label = CLASSES[class_id]
        counts[label] += 1

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image, counts


# ---------- UI / Streamlit ----------
st.title("🧪 Microscopic Tiny Cell Detector (ONNX)")
uploaded_file = st.file_uploader("Upload microscopic image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    orig_img, input_tensor = preprocess(img)

    outputs = session.run(None, {"images": input_tensor})
    boxes, confs, class_ids = postprocess(outputs[0], orig_img)

    result_img, counts = draw_boxes(orig_img, boxes, class_ids, confs)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Original Image")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.subheader("📤 Processed Output")
        st.image(result_img, channels="BGR", use_column_width=True)

    st.subheader("📊 Cell Counts")
    st.write(counts)
