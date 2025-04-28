
import cv2
import os
from trackers.yolo import YoloDetector
from trackers.sort_tracker import SimpleTracker

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

detector = YoloDetector()
tracker = SimpleTracker()

cap = cv2.VideoCapture(0)  # Use webcam or provide path to video
if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

object_ids_prev = set()
frame_count = 0

# New tracking flags
object_states = {}           # Tracks seen/gone states
saved_frames = set()         # Tracks saved appearances
empty_scene_saved = False    # Save empty scene only once

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    filtered_detections = []
    area_threshold = 5000
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        area = (x2 - x1) * (y2 - y1)
        if area > area_threshold:
             filtered_detections.append(det)


    tracks = tracker.update(filtered_detections)

    object_ids_now = set()

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        object_ids_now.add(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detect missing and new objects
    object_states = {}           # {object_id: {'appeared': True/False, 'missing': True/False}}
    saved_frames = set()         # IDs of objects already saved
    empty_scene_saved = False    # Empty frame saved or not

    new_objects = object_ids_now - object_ids_prev
    missing_objects = object_ids_prev - object_ids_now

    # 1️⃣ Save empty scene before any object appears
    if len(object_ids_now) == 0 and not empty_scene_saved:
        cv2.imwrite("output/scene_empty.jpg", frame)
        print("📸 Saved scene without any object")
        empty_scene_saved = True

    # 2️⃣ Save when new object appears
    for obj_id in new_objects:
        print(f"New object detected: {obj_id}")
        if object_states.get(obj_id) is None:
            object_states[obj_id] = {'appeared': False, 'missing': False}

        if not object_states[obj_id]['appeared']:
            filename = f"output/object_{obj_id}_appeared.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved object {obj_id} appearance frame: {filename}")
            object_states[obj_id]['appeared'] = True

    # 3️⃣ Save when object disappears
    for obj_id in missing_objects:
        print(f"Object missing: {obj_id}")
        if object_states.get(obj_id) and not object_states[obj_id]['missing']:
            filename = f"output/object_{obj_id}_missing.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved object {obj_id} missing frame: {filename}")
            object_states[obj_id]['missing'] = True

    object_ids_prev = object_ids_now
    frame_count += 1
    # print(f"Processing frame: {frame_count}")

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Quit requested.")
        break

cap.release()
cv2.destroyAllWindows()
