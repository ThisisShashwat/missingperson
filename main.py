import time, json, threading

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
import streamlink
from imutils.video import VideoStream
from flask import Flask, Response, jsonify, render_template_string, request, render_template

TWITCH_CHANNEL = "plastuchino"

FACECAM_X, FACECAM_Y = 0, 0
FACECAM_W, FACECAM_H = 400, 240

missing_since = None

absent_frames = 0
present_frames = 0
DEBOUNCE_FRAMES = 5 * 60

is_missing = False

detection_paused = False
pause_started_at = None
was_paused = False

PERSIST_FILE = "missing_time.json"

latest_frame = None
current_missing_display = 0.0

REDUCE_STEP = 60
pending_reduction = 0.0
reset_requested = False

pause_until = None

try:
    with open(PERSIST_FILE) as f:
        total_missing = json.load(f)["total_missing"]
except FileNotFoundError:
    total_missing = 0.0


def get_stream_url(channel):
    streams = streamlink.streams(f"https://twitch.tv/{channel}")
    if not streams:
        raise RuntimeError("eee streams found")
    return streams["best"].to_url()

stream_url = get_stream_url(TWITCH_CHANNEL)

vs = VideoStream(stream_url).start()
time.sleep(1.0)

detector = FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path="face_detector.tflite"),
        running_mode=RunningMode.VIDEO,
    )
)

frame_count = 0
blackout = False

last_save_time = time.time()
SAVE_INTERVAL = 2

show_full = False


def detection_loop():
    global missing_since, absent_frames, present_frames, is_missing, total_missing
    global frame_count, last_save_time, latest_frame, current_missing_display
    global detection_paused, pause_started_at, was_paused, pending_reduction, reset_requested, show_full, pause_until

    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame_count += 1

        if pending_reduction:
            current = total_missing + (time.time() - missing_since if is_missing and missing_since else 0)
            total_missing = max(0.0, current - pending_reduction)
            if is_missing:
                missing_since = time.time()
            pending_reduction = 0.0

        if reset_requested:
            total_missing = 0.0
            missing_since = time.time() if is_missing else None
            reset_requested = False

        if not show_full:
            facecam = frame[FACECAM_Y:FACECAM_Y + FACECAM_H, FACECAM_X:FACECAM_X + FACECAM_W]
        else:
            facecam = frame

        if blackout:
            facecam[:] = 0

        if detection_paused and pause_until and time.time() >= pause_until:
            detection_paused = False
            pause_until = None

        if detection_paused:
            if not was_paused:
                pause_started_at = time.time()
                was_paused = True

            current_missing = total_missing + (pause_started_at - missing_since if is_missing else 0)
            current_missing_display = current_missing

            if time.time() - last_save_time >= SAVE_INTERVAL:
                with open(PERSIST_FILE, "w") as f:
                    json.dump({"total_missing": current_missing}, f)
                last_save_time = time.time()
        else:
            if was_paused:
                if is_missing and missing_since is not None:
                    missing_since += time.time() - pause_started_at
                was_paused = False

            rgb = cv2.cvtColor(facecam, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = detector.detect_for_video(mp_image, int(time.time() * 1000))
            face_present = len(result.detections) > 0

            if face_present:
                absent_frames = 0
                present_frames += 1
            else:
                present_frames = 0
                absent_frames += 1

            if is_missing and present_frames >= DEBOUNCE_FRAMES:
                is_missing = False
                total_missing += time.time() - missing_since
                missing_since = None
            elif not is_missing and absent_frames >= DEBOUNCE_FRAMES:
                is_missing = True
                missing_since = time.time()

            current_missing = total_missing + (time.time() - missing_since if is_missing else 0)
            current_missing_display = current_missing

            if time.time() - last_save_time >= SAVE_INTERVAL:
                with open(PERSIST_FILE, "w") as f:
                    json.dump({"total_missing": current_missing}, f)
                last_save_time = time.time()

            # print(f"missing: {current_missing:.1f}s")

            if result.detections:
                bb = result.detections[0].bounding_box
                cv2.rectangle(facecam, (bb.origin_x, bb.origin_y),
                              (bb.origin_x + bb.width, bb.origin_y + bb.height), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', facecam)
        latest_frame = jpeg.tobytes()


flask_app = Flask(__name__)


@flask_app.route('/')
def index():
    return render_template('index.html')


@flask_app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = latest_frame
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/api/status')
def api_status():
    remaining = max(0, pause_until - time.time()) if pause_until and detection_paused else 0
    return jsonify(missing=current_missing_display, paused=detection_paused, fullscreen=show_full,
                   pause_remaining=remaining)

@flask_app.route('/api/pause', methods=['POST'])
def api_pause():
    global detection_paused, pause_until
    data = request.get_json(silent=True) or {}
    minutes = float(data.get('minutes', 5))
    detection_paused = True
    pause_until = time.time() + minutes * 60
    return jsonify(paused=True)


@flask_app.route('/api/resume', methods=['POST'])
def api_resume():
    global detection_paused, pause_until
    detection_paused = False
    pause_until = None
    return jsonify(paused=False)


@flask_app.route('/api/fullscreen', methods=['POST'])
def api_fullscreen():
    global show_full
    show_full = not show_full
    return jsonify(fullscreen=show_full)


@flask_app.route('/api/reduce', methods=['POST'])
def api_reduce():
    global pending_reduction
    pending_reduction += REDUCE_STEP
    return jsonify(ok=True)


@flask_app.route('/api/reset', methods=['POST'])
def api_reset():
    global reset_requested
    reset_requested = True
    return jsonify(ok=True)


threading.Thread(target=detection_loop, daemon=True).start()
flask_app.run(host='0.0.0.0', port=8080, threaded=True)