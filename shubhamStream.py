import io
import logging
import socketserver
from threading import Condition
from http import server
import tensorflow as tf
import numpy as np
import cv2
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/home/TeamThirdAxis/Downloads/icon-classifier-master/ml-code/icons-50.tflite/efficientdet_lite0.tflite")
interpreter.allocate_tensors()

# Get input and output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setup a camera
PAGE = """
<html>
<head>
<title>picamera2 MJPEG streaming demo with Person Detection</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo with Person Detection</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def _init_(self):
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

# Initialize the camera
picam2 = Picamera2()

# Configure camera for MJPEG streaming (640x480 resolution)
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

stream_output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(stream_output))

# Function to process image and detect persons using TensorFlow Lite
def detect_persons(frame):
    # Convert the frame to RGB (as the model expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image to match model's input dimensions (usually 300x300)
    input_size = (300, 300)
    resized_frame = cv2.resize(rgb_frame, input_size)

    # Normalize the frame and expand dimensions for batch input
    input_array = np.expand_dims(resized_frame, axis=0)
    input_array = np.array(input_array, dtype=np.uint8)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    # Get the output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw bounding boxes on the frame for detected persons
    height, width, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5 and class_ids[i] == 0:  # 0 is the class index for 'person'
            ymin, xmin, ymax, xmax = boxes[i]
            startX = int(xmin * width)
            startY = int(ymin * height)
            endX = int(xmax * width)
            endY = int(ymax * height)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start capturing MJPEG frames, detect persons, and send the modified frame to the output buffer
try:
    address = ('192.168.1.40', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on http://localhost:8000")

    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()

        # Detect persons in the captured frame
        frame_with_detections = detect_persons(frame)

        # Encode frame to JPEG and send to the streaming output buffer
        ret, jpeg_frame = cv2.imencode('.jpg', frame_with_detections)
        if ret:
            stream_output.write(jpeg_frame.tobytes())

    server.serve_forever()

except Exception as e:
    logging.error("Error starting server: %s", str(e))

finally:
    picam2.stop_recording()