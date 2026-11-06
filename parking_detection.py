

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_vehicles(image):

    results = model(image)

    vehicle_count = 0
    detected_image = image.copy()

    vehicle_classes = ["car", "truck", "bus", "motorcycle"]

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_classes:

                vehicle_count += 1

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )

                # Draw rectangle
                cv2.rectangle(
                    detected_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display vehicle name
                cv2.putText(
                    detected_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    return detected_image, vehicle_count
