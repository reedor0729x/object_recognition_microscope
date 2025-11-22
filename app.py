import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Microscopic Tiny Cell Detector (YOLOv8-ONNX)", layout="wide")

MODEL_PATH = "best.onnx"
CLASSES = ["RBC", "WBC", "Platelets"]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def preprocess(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (1024, 1024)) / 255.0
    tensor = resized.transpose(2, 0, 1).astype(np.float32)
    return img, np.expand_dims(tensor, axis=0)


def nms(boxes, scores, threshold=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = []
        for j in idxs[1:]:
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
            area2 = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
            iou = inter / (area1 + area2 - inter)
            ious.append(iou)
        idxs = np.delete(idxs, np.where(np.array(ious) > threshold)[0] + 1)
        idxs = np.delete(idxs, 0)
    return keep


def postprocess(pred, original):
    pred = pred.squeeze()      # (8,21504)
    pred = pred.T              # (21504,8)

    conf = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    mask = conf > 0.25

    pred = pred[mask]
    conf = conf[mask]
    cls_ids = cls_ids[mask]

    if len(pred) == 0:
        return np.array([]), np.array([]), np.array([])

    xywh = pred[:, :4]
    h, w, _ = original.shape
    xyxy = np.zeros_like(xywh)
    xyxy[:, 0] = (xywh[:, 0] - xywh[:, 2]/2) * w / 1024
    xyxy[:, 1] = (xywh[:, 1] - xywh[:, 3]/2) * h / 1024
    xyxy[:, 2] = (xywh[:, 0] + xywh[:, 2]/2) * w / 1024
    xyxy[:, 3] = (xywh[:, 1] + xywh[:, 3]/2) * h / 1024

    keep = nms(xyxy, conf)
    return xyxy[keep], conf[keep], cls_ids[keep]


def draw(image, boxes, cls_ids, scores):
    counts = {"RBC":0, "WBC":0, "Platelets":0}
    for box, c, s in zip(boxes, cls_ids, scores):
        lbl = CLASSES[c]
        counts[lbl] += 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(image, f"{lbl} {s:.2f}", (x1, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return image, counts


st.title("🧪 Microscopic Tiny Cell Detector (YOLOv8-ONNX)")
uploaded_file = st.file_uploader("Upload image")

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    original, tensor = preprocess(img_bgr)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: tensor})
    pred = outputs[0]

    boxes, scores, cls_ids = postprocess(pred, original)
    result, counts = draw(original.copy(), boxes, cls_ids, scores)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    st.json(counts)
