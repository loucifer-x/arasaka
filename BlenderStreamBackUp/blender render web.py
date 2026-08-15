import bpy
import threading
import time
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 8000
FPS = 10

# Output resolution — set this to whatever resolution you want the CCTV
# feed to be. The render.opengl camera render below will match it exactly.
RES_X = 640
RES_Y = 480

latest_frame = None
frame_lock = threading.Lock()

# Cross-platform temp path (works on Windows, macOS, Linux)
filepath = os.path.join(tempfile.gettempdir(), "blender_viewport.jpg")
print("Writing capture frames to:", filepath)

# Configure the scene to render at the target resolution and use the
# CCTV camera. Change "CCTV_01" below if your camera has a different name,
# or leave scene.camera as-is if it's already set.
scene = bpy.context.scene
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'JPEG'
scene.render.image_settings.quality = 85

if scene.camera is None:
    cam_obj = bpy.data.objects.get("CCTV_01")
    if cam_obj:
        scene.camera = cam_obj
    else:
        print("[setup] WARNING: no scene.camera set and 'CCTV_01' not found. "
              "Set bpy.context.scene.camera to your CCTV camera before running.")


def capture_viewport():
    global latest_frame
    window = bpy.context.window_manager.windows[0]
    area = None
    region = None
    for a in window.screen.areas:
        if a.type == 'VIEW_3D':
            area = a
            for r in a.regions:
                if r.type == 'WINDOW':
                    region = r
                    break
            if region:
                break
    if area is None or region is None:
        print("[capture] No VIEW_3D area found - open a 3D Viewport")
        return 1.0 / FPS
    try:
        scene.render.filepath = filepath
        with bpy.context.temp_override(window=window, area=area, region=region):
            # view_context=False renders through scene.camera at the
            # scene's render resolution, with no viewport UI/overlays -
            # just the clean camera image, like a fast preview render.
            bpy.ops.render.opengl(write_still=True, view_context=False)

        # Windows can briefly hold a lock on the file right after it's
        # written. Retry a few times instead of failing immediately.
        data = None
        for attempt in range(5):
            try:
                with open(filepath, "rb") as f:
                    data = f.read()
                break
            except PermissionError:
                time.sleep(0.02)

        if data:
            with frame_lock:
                latest_frame = data
            if not hasattr(capture_viewport, "_logged_first"):
                print(f"[capture] First frame captured OK ({len(data)} bytes)")
                capture_viewport._logged_first = True
        else:
            print("[capture] File locked after 5 retries, skipping this frame")
    except Exception as e:
        import traceback
        print("[capture] ERROR:", e)
        traceback.print_exc()
    return 1.0 / FPS


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = """
<!DOCTYPE html>
<html>
<head>
<title>Blender Live View</title>
<style>
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: #111;
    overflow: hidden;
}
img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
</style>
</head>
<body>
<img src="/stream">
</body>
</html>
"""
            data = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/stream":
            print("[server] Browser connected to /stream")
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(1.0 / FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def start_server():
    try:
        ThreadingHTTPServer.allow_reuse_address = True
        server = ThreadingHTTPServer((HOST, PORT), StreamHandler)
    except OSError as e:
        print()
        print(f"[server] FAILED TO START: {e}")
        print(f"[server] Port {PORT} is likely already in use by a previous")
        print("[server] run of this script. Restart Blender, or change PORT")
        print("[server] to a different number (e.g. 8080) and try again.")
        print()
        return
    print()
    print("==============================")
    print(" Blender viewport streaming")
    print("==============================")
    print()
    print("Open Chrome:")
    print(f"http://{HOST}:{PORT}")
    print()
    server.serve_forever()


# Start HTTP server
threading.Thread(target=start_server, daemon=True).start()

# Start Blender viewport capture
bpy.app.timers.register(capture_viewport, first_interval=0.1)

print("Streaming started!")
