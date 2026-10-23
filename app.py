from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
from ultralytics import YOLO

register_heif_opener()



APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "yolo26n.pt"

st.set_page_config(
    page_title="Indian Vehicle Detection",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path: str):
    return YOLO(model_path)


st.title("🚗 Indian Vehicle Detection")
st.write(
    "Upload a traffic image to detect the classes learned by your trained YOLO26 model. "
    "The app displays annotated detections, confidence scores, and class-wise counts."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place yolo26n.pt in the same folder as app.py."
    )
    st.code(
        "vehicle-detection-app/\n"
        "├── app.py\n"
        "├── yolo26n.pt\n"
        "└── requirements.txt"
    )
    st.stop()

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05,
        help="Lower values may detect more objects but can also increase false positives.",
    )
    image_size = st.selectbox(
        "Inference image size", options=[640, 960, 1280], index=0,
        help="Higher sizes can help with small or distant vehicles, but need more memory and time.",
    )
    iou_threshold = st.slider("NMS IoU threshold", 0.10, 0.90, 0.65, 0.05)
    max_detections = st.number_input("Maximum detections", min_value=1, max_value=3000, value=300)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "heic", "heif"]
)   

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(pil_image)
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()

    original_h, original_w = image_rgb.shape[:2]
    max_side = 1920
    if max(original_h, original_w) > max_side:
        scale = max_side / max(original_h, original_w)
        resized_w = max(1, int(original_w * scale))
        resized_h = max(1, int(original_h * scale))
        image_rgb = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image")
        st.image(image_rgb, use_container_width=True)
        st.caption(f"Input dimensions: {original_w} × {original_h}")

    if st.button("Detect vehicles", type="primary"):
        try:
            model = load_model(str(MODEL_PATH))
            with st.spinner("Running object detection..."):
                result = model.predict(
                    source=image_rgb,
                    conf=confidence,
                    imgsz=image_size,
                    iou=iou_threshold,
                    max_det=int(max_detections),
                    verbose=False,
                )[0]

            annotated_bgr = result.plot(boxes=True, labels=True, conf=True)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            boxes = result.boxes
            names = result.names

            with col2:
                st.subheader("Detection result")
                st.image(annotated_rgb, use_container_width=True)

            if boxes is not None and len(boxes) > 0:
                class_ids = boxes.cls.cpu().numpy().astype(int)
                class_counts = Counter(names[class_id] for class_id in class_ids)
            else:
                class_counts = Counter()
            total = sum(class_counts.values())
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Total detections", total)
            metric_col2.metric("Classes detected", len(class_counts))

            st.subheader("Class-wise counts")
            if class_counts:
                counts_df = pd.DataFrame(
                    sorted(class_counts.items(), key=lambda item: item[1], reverse=True),
                    columns=["Class", "Count"],
                )
                st.dataframe(counts_df, use_container_width=True, hide_index=True)
            else:
                st.info("No objects detected. Try lowering the confidence threshold or using a larger image size.")

            st.subheader("Individual detections")
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                detections_df = pd.DataFrame(
                    [
                        {
                            "Class": names[class_id],
                            "Confidence": round(float(score), 4),
                            "X1": round(float(coords[0]), 1),
                            "Y1": round(float(coords[1]), 1),
                            "X2": round(float(coords[2]), 1),
                            "Y2": round(float(coords[3]), 1),
                        }
                        for coords, score, class_id in zip(xyxy, confs, class_ids)
                    ]
                )
                st.dataframe(detections_df, use_container_width=True, hide_index=True)
            else:
                st.caption("There are no individual detections to display.")

            success, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            if success:
                st.download_button(
                    "Download annotated image",
                    data=encoded.tobytes(),
                    file_name="vehicle_detection_result.jpg",
                    mime="image/jpeg",
                )
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.caption(
                "Check that yolo26n.pt is a compatible Ultralytics YOLO weights file "
                "and that the required packages are installed."
)

else:
    st.info("Upload an image to get started.")

st.divider()
st.caption("Inference only: this app uses your trained model and does not train or download a dataset.")
