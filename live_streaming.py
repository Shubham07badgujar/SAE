import io
import logging
import socketserver
from threading import Condition
from http import server
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput



PAGE = """
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
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
# Initialize the camera
picam2 = Picamera2()

# Configure camera for MJPEG streaming (not H.264)
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

stream_output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(stream_output))
# Start capturing MJPEG frames and sending them to the output buffer
#from picamera2.encoders import MJPEGEncoder
#from picamera2.outputs import FfmpegOutput
#output = FfmpegOutput("test.mp4", audio=True)
#menc = MJPEGEncoder()
#picam2.start_recording(menc, output)
#picam2.start_recording(output, encoder_name='mjpeg')  # Corrected parameter

try:
    address = ('192.168.1.40', 8000)
    server = StreamingServer(address, StreamingHandler)
    print("Starting server on http://localhost:8000")
    server.serve_forever()
except Exception as e:
    logging.error("Error starting server: %s", str(e))
finally:
    picam2.stop_recording()
