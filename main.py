import time, json

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
import streamlink

import threading
from fastapi import Response
from nicegui import app, ui


from imutils.video import VideoStream

TWITCH_CHANNEL = "plastuchino"

FACECAM_X, FACECAM_Y = 0,0
FACECAM_W, FACECAM_H = 400, 240

missing_since = None

absent_frames = 0
present_frames = 0
DEBOUNCE_FRAMES = 5*60

is_missing = False

detection_paused = False
pause_started_at = None
was_paused = False

PERSIST_FILE = "missing_time.json"

latest_frame = None

current_missing_display = 0.0

REDUCE_STEP = 60  # seconds
pending_reduction = 0.0
reset_requested = False


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
SAVE_INTERVAL = 2  # seconds

show_full = False

def detection_loop():
    global missing_since, absent_frames, present_frames, is_missing, total_missing
    global frame_count, last_save_time, latest_frame, current_missing_display
    global detection_paused, pause_started_at, was_paused, pending_reduction, reset_requested, show_full
    while True:
        frame = vs.read()
        if frame is None:
            continue

        frame_count += 1

        if pending_reduction:
            total_missing = max(0.0, total_missing - pending_reduction)
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

            print(f"missing: {current_missing:.1f}s")

            if result.detections:
                bb = result.detections[0].bounding_box
                cv2.rectangle(facecam, (bb.origin_x, bb.origin_y),
                              (bb.origin_x + bb.width, bb.origin_y + bb.height), (0, 255, 0), 2)

        if frame_count % 30 == 0:
            print(f"frame {frame_count}, shape {frame.shape}")

        _, jpeg = cv2.imencode('.jpg', facecam)
        latest_frame = jpeg.tobytes()

        # cv2.imshow("stream", facecam)
        #
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break
        # if key == ord('b'):
        #     blackout = not blackout
        # if key == ord('f'):
        #     show_full = not show_full



@app.get("/video/frame")
async def video_frame():
    if latest_frame is None:
        return Response(status_code=404)

    return Response(content=latest_frame, media_type='image/jpeg')


video_image = ui.interactive_image('/video/frame')
ui.timer(interval=0.1, callback=video_image.force_reload)

missing_label = ui.label()
ui.timer(interval=0.5, callback=lambda: missing_label.set_text(f'Missing: {current_missing_display:.1f}s'))

def toggle_pause():
    global detection_paused
    detection_paused = not detection_paused
    pause_button.text = 'Resume detection' if detection_paused else 'Pause detection'

pause_button = ui.button('Pause detection', on_click=toggle_pause)


def reduce_missing():
    global pending_reduction
    pending_reduction += REDUCE_STEP

reduce_button = ui.button(f'-{REDUCE_STEP}s', on_click=reduce_missing)

def toggle_fullscreen():
    global show_full
    show_full = not show_full
    fullscreen_button.text = 'Switch to facecam crop' if show_full else 'Switch to fullscreen'

fullscreen_button = ui.button('Switch to fullscreen', on_click=toggle_fullscreen)

def request_reset():
    global reset_requested
    reset_requested = True

reset_dialog = ui.dialog()
with reset_dialog:
    with ui.card():
        ui.label('Reset missing time to 0? This cannot be undone.')
        with ui.row():
            ui.button('Cancel', on_click=reset_dialog.close)
            ui.button('Reset', color='red', on_click=lambda: (request_reset(), reset_dialog.close()))

reset_button = ui.button('Reset missing time', on_click=reset_dialog.open)

threading.Thread(target=detection_loop, daemon=True).start()

ui.run(host='0.0.0.0', port=8080, reload=False)