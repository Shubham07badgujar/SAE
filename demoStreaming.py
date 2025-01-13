import io
import logging
import socketserver
from threading import Condition
from http import server
import cv2
import torch
import numpy as np

PAGE = """
<html>
<head>
<title>OpenCV MJPEG streaming demo with Human Detection</title>
</head>
<body>
<h1>OpenCV MJPEG Streaming Demo with Human Detection</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

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
                    with stream_output.condition:
                        stream_output.condition.wait()
                        frame = stream_output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Load the human detection model
def load_model(model_path):
    model = torch.load(model_path)  # Load the model
    model.eval()  # Set the model to evaluation mode
    return model

# Function to detect humans in a frame
def detect_humans(frame, model, conf_threshold=0.5):
    # Convert the frame to a format suitable for the model (e.g., tensor)
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV
    img_tensor = torch.from_numpy(img).float()  # Convert to tensor
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Change to CHW format

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Assuming the model outputs bounding boxes in the format [x1, y1, x2, y2, confidence]
    # This will depend on the model, so make sure you adjust for your specific model
    boxes = predictions[0]['boxes'].numpy()  # Detected bounding boxes
    scores = predictions[0]['scores'].numpy()  # Confidence scores

    # Draw bounding boxes for detections with confidence above the threshold
    for box, score in zip(boxes, scores):
        if score > conf_threshold:
            x1, y1, x2, y2 = box
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return frame

# Create a streaming output
stream_output = StreamingOutput()

# Open the camera using OpenCV
cap = cv2.VideoCapture(0)  # 0 for the default camera (adjust if needed)

# Load your custom model for human detection
model_path = '/home/TeamThirdAxis/Downloads/yolov8s (1) (1).pt'  # Path to the .pt model file
model = load_model(model_path)

# Function to process and stream MJPEG with human detection
def process_and_stream():
    while True:
        ret, frame = cap.read()  # Capture a frame from the camera

        if not ret:
            break

        # Detect humans in the captured frame
        detected_frame = detect_humans(frame, model)

        # Encode the frame back to JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', detected_frame)
        if ret:
            stream_output.frame = jpeg.tobytes()  # Update the frame with detection

try:
    # Start the server to stream the video
    address = ('192.168.1.40', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on http://localhost:8000")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    # Process the frames and detect humans simultaneously
    process_and_stream()

except Exception as e:
    logging.error("Error starting server: %s", str(e))

finally:
    cap.release()  # Release the camera
