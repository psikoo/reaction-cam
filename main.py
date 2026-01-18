import cv2
import time
import mediapipe
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision

from landmarks import draw_landmarks_on_image_hand, draw_landmarks_on_image_face
from overlay import draw_overlay_image
from hand import count_fingers

DRAW_LANDMARKS    = True
DRAW_IMAGE        = True

DRAW_FACE_BOX     = True
DRAW_FACE_NUMBERS = False
DRAW_FACE_CIRCLES = True

DRAW_HAND_BOX     = True
DRAW_HAND_NUMBERS = False
DRAW_HAND_CIRCLES = True

number_fingers = 0
cooldown_end_time = 0
last_face_position = (0,0)

hand_options = vision.HandLandmarkerOptions(
    base_options=tasks.BaseOptions(model_asset_path='landmark/hand_landmarker.task'), 
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.9,
    num_hands=2, 
)
face_options = vision.FaceLandmarkerOptions(
    base_options=tasks.BaseOptions(model_asset_path="landmark/face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    num_faces=1,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    timestamp = int(time.time() * 1000)
    mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_result = hand_detector.detect_for_video(mp_image, timestamp)
    face_result = face_detector.detect_for_video(mp_image, timestamp)

    if time.time() > cooldown_end_time:
        # Check hand
        if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
            number_fingers = count_fingers(hand_result)
            cooldown_end_time = time.time() + 0.01
        # Display camera
        if DRAW_LANDMARKS:
            frame = draw_landmarks_on_image_hand(DRAW_HAND_BOX, DRAW_HAND_NUMBERS, DRAW_HAND_CIRCLES, frame, hand_result, number_fingers)
            frame = draw_landmarks_on_image_face(DRAW_FACE_BOX, DRAW_FACE_NUMBERS, DRAW_FACE_CIRCLES, frame, face_result)
        if DRAW_IMAGE: 
            frame, last_face_position = draw_overlay_image(frame, face_result, number_fingers, last_face_position)
        cv2.imshow("Webcam Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()