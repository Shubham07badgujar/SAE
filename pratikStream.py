import io
import logging
import socketserver
from threading import Condition, Thread
from http import server
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# Load the YOLO model
model = YOLO('/home/TeamThirdAxis/Downloads/yolov8s (1) (1).pt')

PAGE = """
<html>
<head>
<title>Picamera2 MJPEG Streaming Demo with YOLOv8 Detection</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo with YOLOv8 Detection</h1>
<img src="stream.mjpg" width="640" height="360" />
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()
        self.detected_frame = None

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

def detect_objects():
    global stream_output, model
    while True:
        with stream_output.condition:
            stream_output.condition.wait()
            frame = stream_output.frame

        # Decode and resize the frame
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        # Skip detection on every frame to improve latency (e.g., process 1 out of every 3 frames)
        if int(time.time() * 10) % 3 == 0:
            start_time = time.time()
            results = model.predict(image, conf=0.25)

            detected_objects = {}
            total_count = 0  # Initialize total object count
            for result in results:
                boxes = result.boxes.xyxy.numpy()
                confidences = result.boxes.conf.numpy()
                class_ids = result.boxes.cls.numpy()
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    label = f'{model.names[int(class_id)]}: {conf:.2f}'
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Update the detected objects count
                    class_name = model.names[int(class_id)]
                    if class_name in detected_objects:
                        detected_objects[class_name] += 1
                    else:
                        detected_objects[class_name] = 1
                    total_count += 1  # Increment total count

            # Display the classification, count, timestamp, and date
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(image, f"Time: {timestamp}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display total count of objects in red at the top right
            cv2.putText(image, f"Total Objects: {total_count}", (450, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            y_offset = 40
            for class_name, count in detected_objects.items():
                cv2.putText(image, f"{class_name}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 20

            # Display the frame on the monitor
            cv2.imshow('YOLOv8 Detection', image)

            # Encode the processed frame back to JPEG
            _, jpeg_frame = cv2.imencode('.jpg', image)
            stream_output.detected_frame = jpeg_frame.tobytes()

            print("Detection processed in: {:.2f} ms".format((time.time() - start_time) * 1000))

        # Wait for 1 ms and check if 'q' is pressed to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if stream_output.detected_frame is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(stream_output.detected_frame))
                        self.end_headers()
                        self.wfile.write(stream_output.detected_frame)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 360)}))

stream_output = StreamingOutput()
detection_thread = Thread(target=detect_objects, daemon=True)
detection_thread.start()

picam2.start_recording(MJPEGEncoder(), FileOutput(stream_output))

try:
    address = ('192.168.1.40', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on http://192.168.1.40:8000")
    server.serve_forever()

except Exception as e:
    logging.error("Error starting server: %s", str(e))

finally:
    picam2.stop_recording()
    cv2.destroyAllWindows()