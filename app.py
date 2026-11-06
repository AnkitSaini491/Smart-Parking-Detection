
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Smart Parking Detection System")
st.write("AI-Based Parking Space Monitoring System")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = np.array(image)

    st.subheader("Uploaded Parking Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Parking"):

        results = model(image)

        vehicle_count = 0
        output_image = image.copy()

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                if label in ["car", "truck", "bus", "motorcycle"]:

                    vehicle_count += 1

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0]
                    )

                    cv2.rectangle(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        output_image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        col1.metric("Vehicles Detected", vehicle_count)
        col2.metric("Detection Status", "Completed")

        st.image(
            output_image,
            caption="Vehicle Detection Output",
            use_container_width=True
        )

        st.success("Vehicle detection completed!")

