import json
import threading
import time

import cv2
import mediapipe as mp
import streamlink
from flask import Flask, Response, jsonify, request, render_template
from imutils.video import VideoStream
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode

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
last_viewer_time = 0
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
    FaceDetectorOptions(base_options=BaseOptions(model_asset_path="face_detector.tflite"),
        running_mode=RunningMode.IMAGE, ))

frame_count = 0
blackout = False

last_save_time = time.time()
SAVE_INTERVAL = 2


def detection_loop():
    global missing_since, absent_frames, present_frames, is_missing, total_missing
    global frame_count, last_save_time, latest_frame, current_missing_display
    global detection_paused, pause_started_at, was_paused, pending_reduction, reset_requested
    global pause_until, last_viewer_time, blackout

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

        if detection_paused and pause_until and time.time() >= pause_until:
            detection_paused = False
            pause_until = None

        if blackout:
            frame[:] = 0

        crop = frame[FACECAM_Y:FACECAM_Y + FACECAM_H, FACECAM_X:FACECAM_X + FACECAM_W]
        crop_result = None

        if detection_paused:
            if not was_paused:
                pause_started_at = time.time()
                was_paused = True

            current_missing = total_missing + (pause_started_at - missing_since if is_missing and missing_since else 0)
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

            h, w = frame.shape[:2]
            corners = [
                (FACECAM_X, FACECAM_Y),
                (w - FACECAM_W, FACECAM_Y),
                (FACECAM_X, h - FACECAM_H),
                (w - FACECAM_W, h - FACECAM_H),
            ]

            face_present = False
            for cx, cy in corners:
                corner_crop = frame[cy:cy + FACECAM_H, cx:cx + FACECAM_W]
                rgb_crop = cv2.cvtColor(corner_crop, cv2.COLOR_BGR2RGB)
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                result = detector.detect(mp_crop)
                if (cx, cy) == (FACECAM_X, FACECAM_Y):
                    crop_result = result
                if len(result.detections) > 0:
                    face_present = True
                    break

            if not face_present:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)
                full_result = detector.detect(mp_full)
                face_present = len(full_result.detections) > 0

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

        if time.time() - last_viewer_time < 5:
            # if crop_result and crop_result.detections:
            #     bb = crop_result.detections[0].bounding_box
            #     cv2.rectangle(crop, (bb.origin_x, bb.origin_y), (bb.origin_x + bb.width, bb.origin_y + bb.height),
            #                   (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            latest_frame = jpeg.tobytes()

        # if frame_count % 30 == 0:  #     print(f"frame {frame_count}, shape {frame.shape}")


flask_app = Flask(__name__)


@flask_app.route('/dontshowtheurl')
def index():
    return render_template('index.html')


@flask_app.route('/video_feed')
def video_feed():
    def generate():
        global last_viewer_time
        while True:
            last_viewer_time = time.time()
            frame = latest_frame
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/api/status')
def api_status():
    global last_viewer_time
    last_viewer_time = time.time()
    remaining = max(0, pause_until - time.time()) if pause_until and detection_paused else 0
    return jsonify(missing=current_missing_display, paused=detection_paused,
                   pause_remaining=remaining, currently_missing=(is_missing and not detection_paused), blackout=blackout)


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

@flask_app.route('/overlay')
def overlay():
    return render_template('overlay.html')

@flask_app.route('/api/blackout', methods=['POST'])
def api_blackout():
    global blackout
    blackout = not blackout
    return jsonify(blackout=blackout)

threading.Thread(target=detection_loop, daemon=True).start()
flask_app.run(host='0.0.0.0', port=8080, threaded=True)