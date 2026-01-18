import os
import cv2
import numpy

IMAGE_FOLDER = 'img'
image_files = [file for file in os.listdir(IMAGE_FOLDER)]

def draw_overlay_image(rgb_image, face_result, number_fingers, last_face_position):
    annotated_image = numpy.copy(rgb_image)
    height, width, _ = annotated_image.shape
    if not face_result or not face_result.face_landmarks: 
        nose_x, nose_y = last_face_position[0], last_face_position[1]
    else:
        # Nose Position
        nose = face_result.face_landmarks[0][1]
        nose_x, nose_y = int(nose.x * width), int(nose.y * height)
        last_face_position = (nose_x, nose_y)
    # Select Image
    while number_fingers > len(image_files)-1: 
        number_fingers = number_fingers - (len(image_files)-1)
    overlay = cv2.imread(os.path.join(IMAGE_FOLDER, image_files[number_fingers]))
    overlay = cv2.resize(overlay, (300, 300))
    # Image coords
    overlay_height, overlay_width, _ = overlay.shape
    x1 = nose_x - (overlay_width // 2)
    x2 = x1     +  overlay_width
    y1 = nose_y - (overlay_height // 2)
    y2 = y1     +  overlay_height

    if x1 < 0: 
        x2 = x2 - x1
        x1 = 0
    if y1 < 0: 
        y2 = y2 - y1
        y1 = 0
    if x2 > width:
        x1 = x1 - (x2 - width) 
        x2 = width
    if y2 > height: 
        y1 = y1 - (y2 - height) 
        y2 = height

    if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
        annotated_image[y1:y2, x1:x2] = overlay

    return annotated_image, (nose_x, nose_y)