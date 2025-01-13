import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from ultralytics import YOLO
import time
from datetime import datetime

# Load the YOLO model
# model = YOLO('/home/TeamThirdAxis/Downloads/yolov8s (1) (1).pt')
model = YOLO('/home/TeamThirdAxis/Downloads/best.pt')

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 360)}))

# Function to process and detect objects in frames
def detect_objects():
    global model

    # Start the camera stream
    picam2.start()

    try:
        frame_skip = 2  # Process every 3rd frame
        frame_count = 0

        while True:
            # Capture a frame
            frame = picam2.capture_array()

            # Ensure the frame has 3 channels (convert if necessary)
            if frame.shape[2] == 4:  # If the frame has 4 channels (e.g., BGRA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Skip frames to reduce latency
            frame_count += 1
            if frame_count % (frame_skip + 1) != 0:
                continue

            # Perform object detection
            start_time = time.time()
            results = model.predict(frame, conf=0.25, device='cpu')

            detected_objects = {}
            total_count = 0  # Initialize total object count
            for result in results:
                boxes = result.boxes.xyxy.numpy()
                confidences = result.boxes.conf.numpy()
                class_ids = result.boxes.cls.numpy()

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    label = f'{model.names[int(class_id)]}: {conf:.2f}'
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update the detected objects count
                    class_name = model.names[int(class_id)]
                    if class_name in detected_objects:
                        detected_objects[class_name] += 1
                    else:
                        detected_objects[class_name] = 1
                    total_count += 1  # Increment total count

            # Display the classification, count, timestamp, and date
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display total count of objects in red at the top right
            cv2.putText(frame, f"Total Objects: {total_count}", (450, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            y_offset = 40
            for class_name, count in detected_objects.items():
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 20

            # Display the frame on the monitor
            cv2.imshow('YOLOv8 Detection', frame)

            print("Detection processed in: {:.2f} ms".format((time.time() - start_time) * 1000))

            # Wait for 1 ms and check if 'q' is pressed to break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the camera and close all OpenCV windows
        picam2.stop()
        cv2.destroyAllWindows()

# Start the detection loop
detect_objects()
